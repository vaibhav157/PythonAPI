from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import Base, engine
from routers.person_account_match import router as person_account_match_router
from routers.prompt_configs import router as prompt_configs_router
from routers.users import router as users_router

app = FastAPI(title="Data Reconciliation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


@app.get("/")
def read_root():
    return {"message": "Data Reconciliation API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(users_router)
app.include_router(person_account_match_router)
app.include_router(prompt_configs_router)
