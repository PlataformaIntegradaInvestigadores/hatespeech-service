"""Autenticación y autorización.

Dos formas de llamar al servicio:

1. **Servicio a servicio** (social-service): header `X-Internal-Api-Key` con `INTERNAL_API_KEY`.
   social-service ya autenticó al usuario con su JWT y propaga la identidad en
   `X-Acting-User-Id` y `X-Acting-User-Is-Staff` (lo toma de su `IdentityPrincipal.is_staff`).
2. **JWT de identity-service** (`Authorization: Bearer ...`), HS256 firmado con `JWT_SIGNING_KEY`.
   El id sale del claim `user_id` (o `company_id`, o `sub`). Hoy identity-service NO incluye
   `is_staff` en el token, por eso la verificación de moderador es configurable
   (`STAFF_CHECK_MODE`): claim `is_staff` si algún día se agrega, o lista `STAFF_USER_IDS`.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings


@dataclass(frozen=True)
class Principal:
    via: str  # "internal" | "jwt"
    user_id: str | None
    is_staff: bool = False


def _staff_from_claims(claims: dict, user_id: str | None) -> bool:
    mode = settings.STAFF_CHECK_MODE
    by_claim = bool(claims.get("is_staff", False))
    by_list = user_id is not None and user_id in {str(u) for u in settings.STAFF_USER_IDS}
    if mode == "claim":
        return by_claim
    if mode == "allowlist":
        return by_list
    return by_claim or by_list


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_principal(
    authorization: str | None = Header(default=None),
    x_internal_api_key: str | None = Header(default=None),
    x_acting_user_id: str | None = Header(default=None),
    x_acting_user_is_staff: str | None = Header(default=None),
) -> Principal:
    if x_internal_api_key is not None:
        if not hmac.compare_digest(x_internal_api_key, settings.INTERNAL_API_KEY):
            raise _unauthorized("API key interna inválida")
        is_staff = (x_acting_user_is_staff or "").strip().lower() in {"1", "true", "yes"}
        return Principal(via="internal", user_id=x_acting_user_id or None, is_staff=is_staff)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise _unauthorized("Credenciales no provistas")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(
            token,
            settings.JWT_SIGNING_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp"]},
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized(f"Token inválido: {exc}") from exc
    if claims.get("token_type", "access") != "access":
        raise _unauthorized("Se requiere un access token")
    raw_id = claims.get("user_id") or claims.get("company_id") or claims.get("sub")
    user_id = str(raw_id) if raw_id is not None else None
    if user_id is None:
        raise _unauthorized("El token no identifica al usuario")
    return Principal(via="jwt", user_id=user_id, is_staff=_staff_from_claims(claims, user_id))


def require_internal_or_staff(principal: Principal = Depends(get_principal)) -> Principal:
    if principal.via == "internal" or principal.is_staff:
        return principal
    raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo social-service o moderadores")


def require_staff(principal: Principal = Depends(get_principal)) -> Principal:
    """Cola y decisiones HITL: moderador (is_staff) identificado."""
    if not principal.is_staff:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Se requiere rol de moderador (is_staff)")
    if not principal.user_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Falta X-Acting-User-Id del moderador")
    return principal
