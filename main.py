import os
from datetime import datetime, timedelta, timezone
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, EmailStr
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import Base, SessionLocal, engine
import models

app = FastAPI(title="Simple FastAPI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on application startup.
Base.metadata.create_all(bind=engine)


# Apply minimal schema migration for existing SQLite DBs created before auth fields existed.
def ensure_auth_columns() -> None:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "users" not in tables:
        return

    existing = {column["name"] for column in inspector.get_columns("users")}
    with engine.begin() as connection:
        if "password_hash" not in existing:
            connection.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR"))
        if "role" not in existing:
            connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR NOT NULL DEFAULT 'user'"))


ensure_auth_columns()

SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "change-this-secret-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class AuthRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class BootstrapAdmin(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "user"


class UserUpdate(BaseModel):
    name: str
    email: EmailStr
    password: str | None = None
    role: str | None = None


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AIRequest(BaseModel):
    prompt: str
    model: str = "gpt-4.1-mini"


class AIResponse(BaseModel):
    model: str
    output_text: str


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    return pwd_context.verify(plain_password, password_hash)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise unauthorized
    except JWTError:
        raise unauthorized

    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    if user is None:
        raise unauthorized
    return user


def require_admin(current_user: models.User = Depends(get_current_user)) -> models.User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def can_access_user(current_user: models.User, user_id: int) -> bool:
    return current_user.role == "admin" or current_user.id == user_id


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/auth/bootstrap-admin", response_model=UserOut, status_code=201)
def bootstrap_admin(payload: BootstrapAdmin, db: Session = Depends(get_db)):
    existing_admin = db.query(models.User).filter(models.User.role == "admin").first()
    if existing_admin is not None:
        raise HTTPException(status_code=403, detail="Admin already exists")

    admin = models.User(
        name=payload.name,
        email=payload.email,
        password_hash=get_password_hash(payload.password),
        role="admin",
    )
    db.add(admin)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists")

    db.refresh(admin)
    return admin


@app.post("/auth/register", response_model=UserOut, status_code=201)
def register(payload: AuthRegister, db: Session = Depends(get_db)):
    user = models.User(
        name=payload.name,
        email=payload.email,
        password_hash=get_password_hash(payload.password),
        role="user",
    )
    db.add(user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists")

    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
async def login(request: Request, db: Session = Depends(get_db)):
    content_type = request.headers.get("content-type", "").lower()
    username: str | None = None
    password: str | None = None

    if "application/json" in content_type:
        payload = await request.json()
        username = payload.get("username") or payload.get("email")
        password = payload.get("password")
    else:
        form = await request.form()
        username = form.get("username") or form.get("email")
        password = form.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=422,
            detail="Provide username/email and password",
        )

    # OAuth2 form uses `username`; in this app it represents email.
    user = db.query(models.User).filter(models.User.email == username).first()

    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(subject=str(user.id))
    return Token(access_token=access_token)


@app.get("/auth/me", response_model=UserOut)
def me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/ai/respond", response_model=AIResponse)
def ai_respond(payload: AIRequest, current_user: models.User = Depends(get_current_user)):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=payload.model,
            input=payload.prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API call failed: {exc}") from exc

    output_text = getattr(response, "output_text", None) or ""
    if not output_text:
        raise HTTPException(status_code=502, detail="OpenAI API returned empty output")

    return AIResponse(model=payload.model, output_text=output_text)


@app.post("/users", response_model=UserOut, status_code=201)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_admin),
):
    if payload.role not in {"user", "admin"}:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")

    user = models.User(
        name=payload.name,
        email=payload.email,
        password_hash=get_password_hash(payload.password),
        role=payload.role,
    )
    db.add(user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists")

    db.refresh(user)
    return user


@app.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_admin),
):
    return db.query(models.User).order_by(models.User.id.asc()).all()


@app.get("/users/{id}", response_model=UserOut)
def get_user(id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if not can_access_user(current_user, id):
        raise HTTPException(status_code=403, detail="Not enough permissions")

    user = db.query(models.User).filter(models.User.id == id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/users/{id}", response_model=UserOut)
def update_user(
    id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if not can_access_user(current_user, id):
        raise HTTPException(status_code=403, detail="Not enough permissions")

    user = db.query(models.User).filter(models.User.id == id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.name = payload.name
    user.email = payload.email

    if payload.password:
        user.password_hash = get_password_hash(payload.password)

    if payload.role is not None:
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Only admin can change roles")
        if payload.role not in {"user", "admin"}:
            raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
        user.role = payload.role

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists")

    db.refresh(user)
    return user


@app.delete("/users/{id}", status_code=204)
def delete_user(
    id: int,
    db: Session = Depends(get_db),
    _: models.User = Depends(require_admin),
):
    user = db.query(models.User).filter(models.User.id == id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return Response(status_code=204)
