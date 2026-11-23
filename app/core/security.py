"""Autenticación y autorización.

Dos formas de llamar al servicio:

1. **Servicio a servicio** (social-service): header `X-Internal-Api-Key`.
   social-service ya autenticó al usuario (reportes) o al admin del panel
   (cola HITL) y propaga esa identidad en `X-Acting-User-Id`. Para la cola,
   ese id es el usuario de la clave compartida de admin, no un usuario de
   identity-service.
2. **JWT de identity-service** (`Authorization: Bearer ...`), HS256 firmado con
   `JWT_SIGNING_KEY`. Sirve para reportar y consultar el estado. No autoriza
   la cola de revisión: eso solo lo hace social-service tras validar
   `X-Admin-Token`.
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
) -> Principal:
    if x_internal_api_key is not None:
        if not hmac.compare_digest(x_internal_api_key, settings.INTERNAL_API_KEY):
            raise _unauthorized("API key interna inválida")
        return Principal(via="internal", user_id=x_acting_user_id or None)

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
    return Principal(via="jwt", user_id=user_id)


def require_internal(principal: Principal = Depends(get_principal)) -> Principal:
    """Clasificación en segundo plano: solo social-service."""
    if principal.via != "internal":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo social-service puede clasificar")
    return principal


def require_acting_admin(principal: Principal = Depends(get_principal)) -> Principal:
    """Cola HITL. social-service ya validó el token de admin y manda su usuario."""
    if principal.via != "internal":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo social-service puede operar la cola")
    if not principal.user_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Falta X-Acting-User-Id del admin")
    return principal
