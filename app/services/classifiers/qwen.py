"""Backend real: Qwen2.5-3B-Instruct + adapter QLoRA ES+ECU (config 3 de la TIC).

Replica exactamente el protocolo de los scripts de evaluación del proyecto:

- Carga (`evaluar_qlora.py`, `evaluar_test_ecu_qlora.py`): tokenizer desde el adapter con
  padding a la derecha; `AutoModelForSequenceClassification(num_labels=2)` en 4-bit NF4 con
  doble cuantización y cómputo bf16; `pad_token_id` del tokenizer; `PeftModel` en modo eval.
- Entrada: el texto crudo, sin prompt ni plantilla de chat, truncado a `max_length=128`
  (valor de `qlora_entrenamiento_es_ecu.json`).
- Probabilidad: `softmax(logits)[:, 1]`; el índice 1 es la clase "odio".
- XAI (`xai_val_ecu.py`, Fase 7d): gradiente de (logit_odio - logit_no_odio) respecto a los
  embeddings de entrada, norma L2 por token, enmascarado con attention_mask y normalizado a
  suma 1; agregado a palabras con `offset_mapping`. Si el gradiente falla, oclusión por token
  (reemplazo por pad, delta de p_odio), igual que el fallback del script.
"""

from __future__ import annotations

import logging
import threading

from app.services.classifiers.base import ClassificationResult, HateSpeechClassifier
from app.services.xai import normalizar, palabras, top_palabras

logger = logging.getLogger(__name__)


class QwenQLoRAClassifier(HateSpeechClassifier):
    name = "qwen"

    def __init__(self, base_model: str, adapter_path: str, max_length: int = 128, top_k: int = 8):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_length = max_length
        self.top_k = top_k
        self.tok = None
        self.model = None
        # Una sola GPU: se serializan las inferencias (el gradiente usa estado del modelo).
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_BACKEND=qwen requiere una GPU NVIDIA con CUDA visible")
        tok = AutoTokenizer.from_pretrained(self.adapter_path)
        tok.padding_side = "right"
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,
            quantization_config=bnb,
            device_map={"": 0},
            dtype=torch.bfloat16,
        )
        base.config.pad_token_id = tok.pad_token_id
        self.model = PeftModel.from_pretrained(base, self.adapter_path).eval()
        self.tok = tok
        logger.info("Modelo %s + adapter %s cargado", self.base_model, self.adapter_path)

    # ------------------------------------------------------------------ inferencia
    def _p_odio(self, text: str) -> float:
        import torch

        with torch.no_grad():
            b = self.tok(
                [text], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
            ).to(0)
            logits = self.model(**b).logits.float()
            return float(torch.softmax(logits, -1)[0, 1].item())

    def _encode_offsets(self, text: str):
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = [tuple(int(v) for v in o) for o in enc.pop("offset_mapping")[0].tolist()]
        return enc["input_ids"].to(0), enc["attention_mask"].to(0), offsets

    def _saliencia_gradiente(self, text: str) -> tuple[list[float], list]:
        input_ids, attn, offsets = self._encode_offsets(text)
        emb_layer = self.model.get_input_embeddings()
        self.model.zero_grad(set_to_none=True)
        embeds_g = emb_layer(input_ids).detach().requires_grad_(True)
        logits = self.model(inputs_embeds=embeds_g, attention_mask=attn).logits.float()
        score = logits[0, 1] - logits[0, 0]
        score.backward()
        if embeds_g.grad is None:
            raise RuntimeError("embeds_g.grad es None")
        sal = embeds_g.grad[0].detach().float().norm(dim=-1).cpu().numpy()
        sal = sal * attn[0].detach().cpu().numpy().astype(bool)
        self.model.zero_grad(set_to_none=True)
        return normalizar(sal.tolist()), offsets

    def _saliencia_oclusion(self, text: str) -> tuple[list[float], list]:
        import torch

        input_ids, attn, offsets = self._encode_offsets(text)
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.inference_mode():
            logits0 = self.model(input_ids=input_ids, attention_mask=attn).logits.float()
            p0 = float(torch.softmax(logits0[0], dim=-1)[1].item())
            deltas = []
            for i in range(int(attn.sum().item())):
                ids2 = input_ids.clone()
                ids2[0, i] = pad_id
                logits = self.model(input_ids=ids2, attention_mask=attn).logits.float()
                deltas.append(p0 - float(torch.softmax(logits[0], dim=-1)[1].item()))
        deltas += [0.0] * (input_ids.shape[1] - len(deltas))
        return normalizar(deltas), offsets

    def predict(self, text: str, explain: bool = True) -> ClassificationResult:
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
        with self._lock:
            p_hate = self._p_odio(text)
            if not explain:
                return ClassificationResult(p_hate=round(p_hate, 6))
            metodo = "gradiente_embeddings_L2"
            try:
                sal, offsets = self._saliencia_gradiente(text)
            except Exception as exc:  # mismo fallback que xai_val_ecu.py
                logger.warning("Saliencia por gradiente falló (%s); se usa oclusión", exc)
                metodo = "oclusion_pad_delta_p_odio"
                sal, offsets = self._saliencia_oclusion(text)
        items = palabras(text, offsets, sal)
        return ClassificationResult(
            p_hate=round(p_hate, 6),
            explanation=top_palabras(items, self.top_k),
            explanation_method=metodo,
        )
