"""FastAPI dependencies — auth, database session, current user resolution."""

from __future__ import annotations

import hmac as _hmac_mod
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from pulse.api import jwt_utils
from pulse.config import settings
from pulse.database import get_db
from pulse.models import Organisation, User

security = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 72


def _sha256_hash(password: str) -> str:
    return sha256(password.encode()).hexdigest()


def hash_password(password: str) -> str:
    return _sha256_hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _hmac_mod.compare_digest(_sha256_hash(plain), hashed)


def create_access_token(user_id: str, org_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "org": org_id, "exp": expire.timestamp()}
    return jwt_utils.encode(payload, settings.secret_key)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Session = Depends(get_db),
) -> User:
    """Resolve current user from JWT token."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = jwt_utils.decode(credentials.credentials, settings.secret_key)
        user_id: str = payload.get("sub", "")
    except (jwt_utils.InvalidTokenError, jwt_utils.ExpiredSignatureError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_org_id(user: Annotated[User, Depends(get_current_user)]) -> str:
    """Shortcut to get current user's org_id."""
    return user.org_id


# Type aliases for cleaner route signatures
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentOrgId = Annotated[str, Depends(get_current_org_id)]
DB = Annotated[Session, Depends(get_db)]
