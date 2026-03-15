import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth
from contextlib import asynccontextmanager
from app.db.base import create_tables



from app.routes import upload, query

# ── Logging setup ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables automatically inside Postgres on startup
    logger.info("Verifying and Creating Database Tables...")
    await create_tables()
    yield


# ── App ────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="High-accuracy RAG chatbot with Hybrid Search (FAISS + BM25 + RRF)",
    version="1.0.0",
    lifespan=lifespan  # ⬅️ Add this trigger
)


# ── CORS (allow all for dev) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(auth.router, prefix="/auth", tags=["Auth"])



@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}


@app.get("/health", tags=["Health"])
async def health():
    from app.services.vector_store import vector_store
    return {
        "status": "healthy",
        "indexed_vectors": vector_store.total_vectors(),
    }
