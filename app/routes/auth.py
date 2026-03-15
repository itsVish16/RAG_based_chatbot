from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from fastapi.security import OAuth2PasswordRequestForm


from app.db.base import get_db
from app.models.user import User
from app.services.auth_service import auth_service

router = APIRouter()

# ── Pydantic Schemas ─────────────────────────
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str

# ── Endpoints ─────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """Creates a new user account with hashed password."""
    # 1. Check if user already exists
    result = await db.execute(select(User).where(User.username == user_in.username))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # 2. Hash and save
    hashed = auth_service.hash_password(user_in.password)
    new_user = User(username=user_in.username, hashed_password=hashed)
    
    db.add(new_user)
    await db.commit()  # Async commit
    await db.refresh(new_user)
    
    return {"message": "User registered successfully", "user_id": str(new_user.id)}


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),  # ⬅️ Swapped to form data for Swagger
    db: AsyncSession = Depends(get_db)
):
    """Verifies credentials and issues a JWT token."""
    # 1. Fetch user using form_data.username
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalars().first()
    
    # 2. Verify using form_data.password
    if not user or not auth_service.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # 3. Create Token
    token = auth_service.create_access_token(user_id=str(user.id))
    
    return {"access_token": token, "token_type": "bearer"}

