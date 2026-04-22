"""Microsoft Entra ID (Azure AD) JWT token verification.

Verifies ID tokens issued by Microsoft for browser-based SSO.
Works alongside the existing API key auth — requests can authenticate
with either ``X-API-Key`` header OR ``Authorization: Bearer <ms_token>``.

Configuration via environment variables:

    CDP_MS_CLIENT_ID     — Azure AD Application (client) ID.
    CDP_ALLOWED_DOMAINS  — Comma-separated list of allowed email domains
                           (e.g. "chubb.com,melon.sa"). Empty = allow all.
    CDP_MS_TENANT        — Azure AD tenant ID or "common" for multi-tenant.
                           Defaults to "common".
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_jwks_cache: dict[str, Any] = {}
_jwks_lock = threading.Lock()
_JWKS_TTL = 3600


def _get_jwks(tenant: str = "common") -> dict[str, Any]:
    """Fetch and cache Microsoft's OIDC signing keys."""
    now = time.time()
    cached = _jwks_cache.get(tenant)
    if cached and now - cached["fetched_at"] < _JWKS_TTL:
        return cached["keys"]

    with _jwks_lock:
        cached = _jwks_cache.get(tenant)
        if cached and now - cached["fetched_at"] < _JWKS_TTL:
            return cached["keys"]

        url = f"https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys"
        logger.info("Fetching Microsoft JWKS from %s", url)
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        _jwks_cache[tenant] = {"keys": data, "fetched_at": time.time()}
        return data


def verify_ms_token(token: str) -> dict[str, Any] | None:
    """Verify a Microsoft ID token. Returns decoded claims or None on failure.

    Requires ``PyJWT[crypto]`` (``pip install pyjwt[crypto]``).
    Returns None (instead of raising) so callers can fall through to
    API-key auth gracefully.
    """
    client_id = os.environ.get("CDP_MS_CLIENT_ID", "")
    if not client_id:
        return None

    tenant = os.environ.get("CDP_MS_TENANT", "common")

    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError:
        logger.warning("PyJWT not installed — Microsoft SSO disabled.")
        return None

    try:
        jwks_url = f"https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys"
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(token)

        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            options={"verify_exp": True, "verify_aud": True},
        )

        email = (claims.get("preferred_username") or claims.get("email") or claims.get("upn") or "").lower()

        allowed_raw = os.environ.get("CDP_ALLOWED_DOMAINS", "")
        if allowed_raw:
            allowed = {d.strip().lower() for d in allowed_raw.split(",") if d.strip()}
            domain = email.split("@")[-1] if "@" in email else ""
            if domain not in allowed:
                logger.warning("Token email domain %r not in allowed domains %s", domain, allowed)
                return None

        claims["_verified_email"] = email
        return claims

    except Exception as exc:
        logger.debug("Microsoft token verification failed: %s", exc)
        return None
