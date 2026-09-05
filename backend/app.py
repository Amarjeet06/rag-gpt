# ==============================
# RAG-GPT Backend - Local Development
# ==============================
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
import os, json, uuid, time, base64, re, math

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# Auth
from jose import jwt, JWTError
from passlib.context import CryptContext

# LangChain (chat)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from starlette.concurrency import run_in_threadpool

# Optional Google GenAI (guarded so we don't crash if unset)
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# Hugging Face fallback (image gen)
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# PDF extraction
from pypdf import PdfReader

from . import vector_store

# ====== Answerability (embeddings + NLI) ======
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder  # for reranking
except Exception:
    CrossEncoder = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    hf_pipeline = None

# ====== Business KPI Forecasting ======
import pandas as pd
try:
    import pdfplumber  # optional for table extraction from PDFs
except Exception:
    pdfplumber = None
try:
    import pyarrow
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None


# ------------------------ ENV & APP ------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGO = "HS256"

# LLM choices (override in .env if you like)
TEXT_MODEL   = os.getenv("GEMINI_TEXT_MODEL",   "gemini-2.0-flash")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
IMAGEN_MODEL = os.getenv("IMAGEN_MODEL",        "imagen-3.0-fast")

# Provider switch: "auto" | "hf" | "google"
IMAGE_PROVIDER = (os.getenv("IMAGE_PROVIDER") or "auto").lower()

# Conversational memory window
CHAT_MEMORY_K = int(os.getenv("CHAT_MEMORY_K", "12"))

# Answerability configs
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NLI_MODEL_NAME = os.getenv("NLI_MODEL", "microsoft/deberta-base-mnli")

# Reranker + Router configs
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
ROUTER_ZS_MODEL = os.getenv("ROUTER_ZS_MODEL", "facebook/bart-large-mnli")  # zero-shot if available

_emb_model = None
_nli_tok = None
_nli_model = None
_reranker = None
_router_zs = None  # transformers zero-shot pipeline

# Local data directory
DATA_DIR = Path(os.getenv("DATA_DIR") or "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# KPI storage dir
KPI_DIR = DATA_DIR / "kpi_data"
KPI_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

app = FastAPI(title="RAG-GPT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy Google client — only if key present and lib available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_CLIENT = None
if GEMINI_API_KEY and genai is not None:
    try:
        GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("WARN: Google GenAI client init failed:", repr(e))
        GENAI_CLIENT = None


# ------------------------ MODEL DISCOVERY (Google) ------------------------
def list_model_names() -> List[str]:
    if GENAI_CLIENT is None:
        return []
    try:
        items = GENAI_CLIENT.models.list()
        names = []
        for m in (getattr(items, "models", None) or items):
            name = getattr(m, "name", None) or str(m)
            names.append(name.split("/")[-1])
        return names
    except Exception as e:
        print("list_model_names error:", repr(e))
        return []

def pick_imagen_model() -> Optional[str]:
    for n in list_model_names():
        if "imagen" in n:
            return n
    return None


# ------------------------ SIMPLE USER STORE ------------------------
USERS_FILE = DATA_DIR / "users.json"

def load_users() -> Dict[str, Dict[str, Any]]:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_users(users: Dict[str, Dict[str, Any]]):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

users_db = load_users()  # {username: {"password": <hashed>}}


# ------------------------ CHAT PERSISTENCE ------------------------
def user_file(username: str) -> Path:
    return DATA_DIR / f"{username}.json"

def load_user_chats(username: str) -> Dict[str, Any]:
    p = user_file(username)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"chats": {}}
    return {"chats": {}}

def save_user_chats(username: str, payload: Dict[str, Any]):
    user_file(username).write_text(json.dumps(payload, indent=2), encoding="utf-8")

# ========== personalization profile ==========
def profile_file(username: str) -> Path:
    return DATA_DIR / f"{username}_profile.json"

def load_profile(username: str) -> Dict[str, Any]:
    p = profile_file(username)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"source_affinity": {}}
    return {"source_affinity": {}}

def save_profile(username: str, prof: Dict[str, Any]):
    profile_file(username).write_text(json.dumps(prof, indent=2), encoding="utf-8")


# ------------------------ AUTH HELPERS ------------------------
def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(sub: str) -> str:
    payload = {"sub": sub, "exp": int(time.time()) + 60 * 60 * 24}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGO)

def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
        return payload.get("sub")
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from e

def auth_required(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> Dict[str, Any]:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    username = decode_token(creds.credentials)
    if not username or username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid user")
    return {"username": username}


# ------------------------ LANGCHAIN MEMORY (in-process) ------------------------
chains: Dict[str, ConversationChain] = {}

def chain_key(user: str, chat_id: str) -> str:
    return f"{user}:{chat_id}"

def get_chain(user: str, chat_id: str) -> ConversationChain:
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY for chat model.")
    key = chain_key(user, chat_id)
    if key not in chains:
        llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))
        memory = ConversationBufferWindowMemory(k=CHAT_MEMORY_K, return_messages=True)
        chains[key] = ConversationChain(llm=llm, memory=memory, verbose=False)
    return chains[key]

def reset_chain(user: str, chat_id: str):
    key = chain_key(user, chat_id)
    if key in chains:
        llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))
        chains[key] = ConversationChain(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=CHAT_MEMORY_K, return_messages=True),
        )

def rehydrate_memory_if_empty(user: str, chat_id: str, chain: ConversationChain):
    try:
        msgs = getattr(chain.memory.chat_memory, "messages", [])
    except Exception:
        msgs = []
    if msgs:
        return
    store = load_user_chats(user)
    history = store.get("chats", {}).get(chat_id, {}).get("messages", [])
    if not history:
        return
    for m in history[-(2 * CHAT_MEMORY_K):]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            chain.memory.chat_memory.add_user_message(content)
        else:
            chain.memory.chat_memory.add_ai_message(content)


# ------------------------ Pydantic Models ------------------------
class RegisterReq(BaseModel):
    username: str
    password: str

class LoginReq(BaseModel):
    username: str
    password: str

class ChatReq(BaseModel):
    user_input: str
    chat_id: Optional[str] = "default"

class ImageGenReq(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "1:1"
    count: int = 1
    provider: Optional[str] = None
    model: Optional[str] = None
    chat_id: Optional[str] = None


# ------------------------ ROUTES: HEALTH & MODELS ------------------------
@app.get("/")
def health():
    return {
        "ok": True,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "image_model": IMAGEN_MODEL,
        "image_provider": IMAGE_PROVIDER,
        "google_enabled": GENAI_CLIENT is not None,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "pdf_enabled": True,
        "answerability_enabled": True,
        "emb_model": EMB_MODEL_NAME,
        "nli_model": (NLI_MODEL_NAME if AutoTokenizer and AutoModelForSequenceClassification and torch is not None else None),
        "business_forecast_enabled": True,
        "reranker_enabled": bool(CrossEncoder is not None),
        "router_zero_shot_enabled": bool(hf_pipeline is not None and torch is not None),
    }

@app.get("/api/models")
def models(user=Depends(auth_required)):
    return {"models": list_model_names()}

@app.get("/api/image/status")
def image_status(user=Depends(auth_required)):
    return {
        "provider": IMAGE_PROVIDER,
        "imagen_env": IMAGEN_MODEL,
        "google_enabled": GENAI_CLIENT is not None,
        "google_visible_models": list_model_names(),
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "hf_model": os.getenv("HF_IMAGE_MODEL"),
    }


# ------------------------ ROUTES: AUTH ------------------------
@app.post("/auth/register")
def register(req: RegisterReq):
    if req.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password too short (min 6)")
    users_db[req.username] = {"password": hash_password(req.password)}
    save_users(users_db)
    token = create_token(req.username)
    return {"username": req.username, "token": token}

@app.post("/auth/login")
def login(req: LoginReq):
    user = users_db.get(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(req.username)
    return {"username": req.username, "token": token}

# ------------------------ ROUTES: CHAT MGMT ------------------------
@app.get("/api/chats")
def list_chats(user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    chats_meta = []
    for cid, c in store["chats"].items():
        chats_meta.append({
            "id": cid,
            "title": c.get("title") or "New chat",
            "updated_at": c.get("updated_at", int(time.time())),
        })
    chats_meta.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"chats": chats_meta}

@app.post("/api/chats/new")
def new_chat(user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    cid = uuid.uuid4().hex[:12]
    store["chats"][cid] = {
        "title": "New chat",
        "messages": [],
        "updated_at": int(time.time())
    }
    save_user_chats(u, store)
    return {"chat_id": cid}

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    chat = store["chats"].get(chat_id, {"messages": []})
    return {"messages": chat.get("messages", [])}

@app.post("/api/chats/{chat_id}/clear")
def clear_chat(chat_id: str, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    if chat_id in store["chats"]:
        store["chats"][chat_id]["messages"] = []
        store["chats"][chat_id]["updated_at"] = int(time.time())
        save_user_chats(u, store)
    reset_chain(u, chat_id)
    return {"ok": True}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    if chat_id in store["chats"]:
        del store["chats"][chat_id]
        save_user_chats(u, store)
    key = f"{u}:{chat_id}"
    if key in chains:
        del chains[key]
    return {"ok": True}

@app.post("/api/reset")
def reset_memory(req: ChatReq, user=Depends(auth_required)):
    """Reset the conversation memory for a chat (frontend compatibility)."""
    u = user["username"]
    chat_id = req.chat_id or "default"
    reset_chain(u, chat_id)
    return {"ok": True}

# ------------------------ ROUTES: CHAT ------------------------
@app.post("/api/chat")
async def chat(req: ChatReq, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)

    chat_id = req.chat_id or "default"
    if chat_id not in store["chats"]:
        store["chats"][chat_id] = {"title": "New chat", "messages": [], "updated_at": int(time.time())}

    # ------------ Answerability pre-check over indexed docs ------------
    chat_store = store["chats"][chat_id]
    doc_index = chat_store.get("doc_index") or []

    # Pre-score (no post-score here yet — we haven't answered)
    top_for_pre = retrieve_top(doc_index, req.user_input, k=8) if doc_index else []
    pre = combine_pre_answer_metrics(top_for_pre, req.user_input)
    action = decide_action(pre["pre_score"], None)

    # Optional tweak: if there are no docs, just chat normally
    if not doc_index:
        action = "ok"

    # Helper to persist a message pair
    def _persist(user_text: str, assistant_text: str):
        now_str = time.strftime("%I:%M %p")
        chat_store["messages"].append({"role": "user", "content": user_text, "time": now_str})
        chat_store["messages"].append({"role": "assistant", "content": assistant_text, "time": now_str})
        if chat_store.get("title", "New chat") == "New chat":
            chat_store["title"] = user_text[:40]
        chat_store["updated_at"] = int(time.time())
        save_user_chats(u, store)

    # ------------ Branch 1: Ask a clarifying question ------------
    if action == "clarify":
        clarify = (
            "I might be missing context. Could you clarify exactly what you need "
            "(e.g., section/page, metric, or timeframe)?"
        )
        _persist(req.user_input, clarify)
        return {
            "response": clarify,
            "answerability": {"pre_score": pre["pre_score"], "action": action},
            "used_context": False,
            "sources_used": []  # none yet
        }

    # ------------ Branch 2: Retrieve + rerank + personalized boost ------------
    if action == "more_retrieval":
        # Stage 1: shortlist by embeddings
        pool = 18
        shortlist = retrieve_top(doc_index, req.user_input, k=pool)

        # Stage 2: cross-encoder rerank (if available)
        reranker = get_reranker()
        if reranker:
            pairs = [(req.user_input, r["text"]) for r in shortlist]
            try:
                ce_scores = reranker.predict(pairs).tolist()
            except Exception as e:
                print("Rerank failed:", repr(e))
                ce_scores = [0.0] * len(shortlist)
        else:
            ce_scores = [0.0] * len(shortlist)

        # Normalize + combine with embedding + personalized boost
        def _norm01(vals):
            if not vals: return []
            lo, hi = min(vals), max(vals)
            return [0.5]*len(vals) if (hi-lo)<1e-8 else [(v-lo)/(hi-lo) for v in vals]

        ce_n = _norm01(ce_scores)
        emb_n = _norm01([r["score"] for r in shortlist])
        combined = []
        for i, r in enumerate(shortlist):
            boost = _personal_boost(u, r.get("source", ""))
            final = 0.65*ce_n[i] + 0.25*emb_n[i] + 0.10*boost
            combined.append({**r, "ce": ce_scores[i], "final": final, "boost": boost})

        combined.sort(key=lambda x: -x["final"])
        topk = combined[:6]  # 4–6 good; we use 6

        # Build grounded context with light citations
        ctx_lines = []
        for i, r in enumerate(topk, 1):
            src = r.get("source") or "document"
            sec = r.get("section_idx", -1)
            snippet = (r.get("text") or "").strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + " …"
            ctx_lines.append(f"[{i}] ({src} §{sec})\n{snippet}")

        ctx = "\n\n".join(ctx_lines)

        # Ask the LLM to answer strictly from context and include explicit citations
        if not os.getenv("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY for chat model.")
        llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))

        prompt = (
            "You are answering strictly from the provided context. If the answer is not present, say so.\n"
            "Include inline citations using the format `[source §section]` referring to the context lines below.\n"
            "Be concise and factual.\n\n"
            f"Question:\n{req.user_input}\n\n"
            f"Context:\n{ctx}\n\n"
            "Answer:"
        )
        try:
            reply = await run_in_threadpool(llm.predict, prompt)
        except Exception as e:
            # If anything fails, fall back to normal chat
            chain = get_chain(u, chat_id)
            rehydrate_memory_if_empty(u, chat_id, chain)
            reply = await run_in_threadpool(chain.predict, input=req.user_input)

        _persist(req.user_input, reply)

        # Also keep in-process memory synced (so follow-ups work)
        try:
            chain = get_chain(u, chat_id)
            rehydrate_memory_if_empty(u, chat_id, chain)
            chain.memory.chat_memory.add_user_message(req.user_input)
            chain.memory.chat_memory.add_ai_message(reply)
        except Exception:
            pass

        # Provide sources_used back to UI for 👍/👎 feedback
        srcs = []
        for r in topk:
            src = r.get("source")
            if src:
                srcs.append({"source": src, "section_idx": r.get("section_idx", -1)})

        return {
            "response": reply,
            "answerability": {"pre_score": pre["pre_score"], "action": action},
            "used_context": True,
            "sources_used": srcs
        }

    # ------------ Branch 3: Normal chat ------------
    try:
        chain = get_chain(u, chat_id)
        rehydrate_memory_if_empty(u, chat_id, chain)
        reply = await run_in_threadpool(chain.predict, input=req.user_input)
    except HTTPException as he:
        raise he
    except Exception as e:
        return {"response": f"Error: {e}", "answerability": {"pre_score": pre["pre_score"], "action": "ok"}, "used_context": False, "sources_used": []}

    # Persist & return
    now_str = time.strftime("%I:%M %p")
    chat_store["messages"].append({"role": "user", "content": req.user_input, "time": now_str})
    chat_store["messages"].append({"role": "assistant", "content": reply, "time": now_str})
    if chat_store.get("title", "New chat") == "New chat":
        chat_store["title"] = req.user_input[:40]
    chat_store["updated_at"] = int(time.time())
    save_user_chats(u, store)

    return {
        "response": reply,
        "answerability": {"pre_score": pre["pre_score"], "action": "ok"},
        "used_context": False,
        "sources_used": []
    }

# ------------------------ ROUTES: VISION (image -> text) ------------------------
@app.post("/api/vision")
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image"),
    chat_id: str = Form("default"),
    user=Depends(auth_required),
):
    if GENAI_CLIENT is None or types is None:
        return JSONResponse(status_code=400, content={"detail": "Vision requires GEMINI_API_KEY to be set on the server."})
    try:
        img_bytes = await image.read()
        img_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type=image.content_type or "image/jpeg",
        )
        result = GENAI_CLIENT.models.generate_content(
            model=VISION_MODEL,
            contents=[img_part, prompt],
        )
        text = getattr(result, "text", None) or "(no response)"

        u = user["username"]
        store = load_user_chats(u)
        if chat_id not in store["chats"]:
            store["chats"][chat_id] = {"title": "New chat", "messages": [], "updated_at": int(time.time())}
        now_str = time.strftime("%I:%M %p")
        store["chats"][chat_id]["messages"].append({"role": "user", "content": f"(uploaded image) {prompt}", "time": now_str})
        store["chats"][chat_id]["messages"].append({"role": "assistant", "content": text, "time": now_str})
        store["chats"][chat_id]["updated_at"] = int(time.time())
        save_user_chats(u, store)

        try:
            chain = get_chain(u, chat_id)
            rehydrate_memory_if_empty(u, chat_id, chain)
            chain.memory.chat_memory.add_user_message(f"(uploaded image) {prompt}")
            chain.memory.chat_memory.add_ai_message(text)
        except Exception:
            pass

        return {"response": text}
    except Exception as e:
        print("Vision error:", repr(e))
        return JSONResponse(status_code=400, content={"detail": str(e)})


# ------------------------ HF HELPERS (text -> image) ------------------------
def get_hf_client(model_override: Optional[str] = None):
    token = os.getenv("HF_TOKEN")
    model = model_override or os.getenv("HF_IMAGE_MODEL") or "black-forest-labs/FLUX.1-schnell"
    if not InferenceClient:
        return None, model, "huggingface_hub not installed."
    try:
        client = InferenceClient(model=model, token=token, timeout=60)
        return client, model, None
    except Exception as e:
        return None, model, f"Failed to create HF client: {type(e).__name__}: {e}"

def generate_with_hf(prompt: str, count: int = 1, model_override: Optional[str] = None):
    client, model, note = get_hf_client(model_override)
    if not client:
        return [], model, note
    n = max(1, min(int(count or 1), 4))
    images: List[str] = []
    for _ in range(n):
        try:
            try:
                pil_img = client.text_to_image(prompt, width=512, height=512)
            except TypeError:
                pil_img = client.text_to_image(prompt)
        except Exception as e:
            detail = (e.args[0] if e.args else repr(e))
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    sc = getattr(resp, "status_code", "?")
                    body = getattr(resp, "text", "")[:500]
                    detail += f" | HTTP {sc}: {body}"
                except Exception:
                    pass
            return [], model, f"Hugging Face error: {detail}"
        if pil_img is None:
            return [], model, "Hugging Face returned no image."
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))
    return images, model, None


# ------------------------ ROUTES: IMAGE GENERATION (text -> image) ------------------------
@app.post("/api/image/generate")
async def generate_image(req: ImageGenReq, user=Depends(auth_required)):
    provider = (req.provider or IMAGE_PROVIDER or "auto").lower()

    def _try_google(model_name: str):
        if GENAI_CLIENT is None or types is None:
            raise RuntimeError("Google Imagen not available (GEMINI_API_KEY not set or library missing).")
        cfg = types.GenerateImagesConfig(
            number_of_images=max(1, min(req.count, 4)),
            aspect_ratio=req.aspect_ratio or "1:1",
        )
        return GENAI_CLIENT.models.generate_images(
            model=model_name,
            prompt=req.prompt,
            config=cfg,
        )

    try:
        images: List[str] = []
        note: Optional[str] = None
        model_used: Optional[str] = None
        used_provider: Optional[str] = None

        if provider in ("google", "auto"):
            visible = list_model_names() if GENAI_CLIENT is not None else []
            google_model = None
            if IMAGEN_MODEL and any(IMAGEN_MODEL in m for m in visible):
                google_model = IMAGEN_MODEL
            else:
                google_model = pick_imagen_model()
            if google_model:
                try:
                    model_used = google_model
                    resp = _try_google(model_used)
                    if hasattr(resp, "images") and resp.images:
                        for img in resp.images:
                            raw = getattr(img, "bytes", None) or getattr(img, "image_bytes", None)
                            if raw:
                                images.append("data:image/png;base64," + base64.b64encode(raw).decode("utf-8"))
                    elif hasattr(resp, "generated_images") and resp.generated_images:
                        for gi in resp.generated_images:
                            blob = getattr(gi, "image", None)
                            raw = getattr(blob, "image_bytes", None) if blob else None
                            if raw:
                                images.append("data:image/png;base64," + base64.b64encode(raw).decode("utf-8"))
                    if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
                        note = str(resp.prompt_feedback)
                    if images:
                        used_provider = "google"
                except Exception as e:
                    print("Google Imagen error:", str(e))
            else:
                if provider == "google":
                    return JSONResponse(status_code=400, content={"detail": "No Google Imagen model available to your API key."})

        if not images and provider in ("auto", "hf"):
            imgs, hf_model, hf_note = generate_with_hf(req.prompt, req.count, model_override=req.model)
            if imgs:
                return {"images": imgs, "note": hf_note, "provider": "huggingface", "model_used": hf_model}
            else:
                return JSONResponse(status_code=400, content={"detail": hf_note or "No image returned by available providers."})

        if images:
            return {"images": images, "note": note, "provider": used_provider or "google", "model_used": model_used}

        return JSONResponse(status_code=400, content={"detail": "No image returned by available providers."})

    except Exception as e:
        print("Image generation error:", repr(e))
        return JSONResponse(status_code=400, content={"detail": str(e)})


# ------------------------ PDF HELPERS ------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t.strip())
        return "\n\n".join(parts)
    except Exception as e:
        print("PDF extract error:", repr(e))
        return ""

def split_text(text: str, max_chars: int = 2500, overlap: int = 250) -> List[str]:
    chunks: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if j >= n:
            break
    return chunks


# ------------------------ ROUTES: PDF READING ------------------------
@app.post("/api/pdf/read")
async def pdf_read(
    pdf: UploadFile = File(...),
    chat_id: str = Form("default"),
    question: Optional[str] = Form(None),
    mode: str = Form("auto"),
    user=Depends(auth_required),
):
    u = user["username"]
    raw = await pdf.read()
    text = extract_pdf_text(raw)
    if not text.strip():
        return JSONResponse(status_code=400, content={"detail": "Could not extract text from PDF."})

    chunks = split_text(text, max_chars=2500, overlap=250)[:24]

    store = load_user_chats(u)
    if chat_id not in store["chats"]:
        store["chats"][chat_id] = {"title": "New chat", "messages": [], "updated_at": int(time.time())}

    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    chunk_embs = emb_model.encode(chunks, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

    doc_index = []
    for i, ch in enumerate(chunks):
        doc_index.append({
            "text": ch,
            "embedding": chunk_embs[i].tolist(),
            "source": pdf.filename,
            "section_idx": i
        })
    store["chats"][chat_id]["doc_index"] = doc_index

    # Additionally persist to PostgreSQL/pgvector when DATABASE_URL is configured.
    # This runs alongside the JSON-file path above (not a replacement for it
    # yet) so existing behavior is unaffected when no database is configured,
    # while giving the app a real, tested, persistent vector store when one is.
    if vector_store.is_enabled():
        try:
            vector_store.init_db()
            vector_store.replace_chunks(u, chat_id, pdf.filename, chunks, chunk_embs.tolist())
        except Exception as e:
            print("WARN: pgvector persist failed (falling back to JSON-only):", repr(e))
    store["chats"][chat_id]["updated_at"] = int(time.time())
    save_user_chats(u, store)

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY for chat model.")
    llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))
    now_str = time.strftime("%I:%M %p")

    if question and question.strip():
        notes: List[str] = []
        for idx, ch in enumerate(chunks):
            prompt = (
                f"You are given part {idx+1}/{len(chunks)} of a PDF.\n"
                f"Question: {question}\n\n"
                "If this section contains facts that help answer, write concise bullet notes.\n"
                "If nothing relevant, reply exactly: NONE\n\n"
                f"SECTION:\n{ch}"
            )
            out = await run_in_threadpool(llm.predict, prompt)
            if out and "NONE" not in out.strip().upper():
                notes.append(out.strip())

        synth = (
            "Using ONLY the notes below, answer the question concisely and accurately.\n"
            "If notes are insufficient, say you couldn't find it in the PDF.\n\n"
            f"Question: {question}\n\nNotes:\n" + "\n\n".join(f"- {n}" for n in notes)
        )
        answer = await run_in_threadpool(llm.predict, synth)

        store = load_user_chats(u)
        store["chats"][chat_id]["messages"].append({"role":"user","content":f"(uploaded PDF) Q: {question} — {pdf.filename}","time":now_str})
        store["chats"][chat_id]["messages"].append({"role":"assistant","content":answer,"time":now_str})
        store["chats"][chat_id]["updated_at"] = int(time.time())
        save_user_chats(u, store)

        try:
            chain = get_chain(u, chat_id)
            rehydrate_memory_if_empty(u, chat_id, chain)
            chain.memory.chat_memory.add_user_message(f"(uploaded PDF) Q: {question} — {pdf.filename}")
            chain.memory.chat_memory.add_ai_message(answer)
        except Exception:
            pass

        return {"response": answer, "chunks": len(chunks), "notes_used": len(notes)}

    else:
        section_summaries: List[str] = []
        for ch in chunks:
            s = await run_in_threadpool(
                llm.predict,
                "Summarize this section in 3–6 bullets. Keep key facts, names, dates, numbers:\n\n" + ch
            )
            section_summaries.append(s.strip())

        final = await run_in_threadpool(
            llm.predict,
            "Combine these section summaries into a structured summary with:\n"
            "- Title\n- Brief outline\n- Bullet points grouped by headings\n- Key takeaways\n\n"
            "Avoid repetition; be faithful to the text.\n\n"
            "Section summaries:\n" + "\n\n".join(section_summaries)
        )

        store = load_user_chats(u)
        store["chats"][chat_id]["messages"].append({"role":"user","content":f"(uploaded PDF) Summarize: {pdf.filename}","time":now_str})
        store["chats"][chat_id]["messages"].append({"role":"assistant","content":final,"time":now_str})
        store["chats"][chat_id]["updated_at"] = int(time.time())
        save_user_chats(u, store)

        try:
            chain = get_chain(u, chat_id)
            rehydrate_memory_if_empty(u, chat_id, chain)
            chain.memory.chat_memory.add_user_message(f"(uploaded PDF) Summarize: {pdf.filename}")
            chain.memory.chat_memory.add_ai_message(final)
        except Exception:
            pass

        return {"response": final, "chunks": len(chunks)}


# ======================== Answerability / Hallucination Risk ========================
def get_emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return _emb_model

def get_nli():
    global _nli_tok, _nli_model
    if torch is None or AutoTokenizer is None:
        return None, None
    if _nli_tok is None or _nli_model is None:
        _nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _nli_model.eval()
    return _nli_tok, _nli_model

STOPWORDS = set("""
a an the and or of to for from in on at by with up out as is are was were be been being
this that those these which who whom whose what when where why how
i you he she it we they me him her us them my your his hers our their
""".split())
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def content_terms(text: str) -> set[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}

def keyword_coverage(q_terms: set[str], ctx: str) -> float:
    if not q_terms:
        return 0.0
    hits = sum(1 for t in q_terms if t in ctx.lower())
    return hits / max(1, len(q_terms))

def retrieve_top(doc_index: List[Dict[str, Any]], query: str, k: int = 8):
    if not doc_index:
        return []
    emb_model = get_emb_model()
    qv = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    mat = np.array([np.array(d["embedding"], dtype=np.float32) for d in doc_index])
    sims = (mat @ qv)  # cosine
    order = np.argsort(-sims)[:k]
    results = []
    for rank, i in enumerate(order):
        d = doc_index[int(i)]
        results.append({
            "rank": int(rank+1),
            "score": float(sims[i]),
            "text": d["text"],
            "source": d.get("source", ""),
            "section_idx": d.get("section_idx", -1),
        })
    return results

def nli_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    tok, model = get_nli()
    if tok is None or model is None:
        return {}
    with torch.no_grad():
        batch = tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        logits = model(**batch).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
    return {"contradiction": probs[0], "neutral": probs[1], "entailment": probs[2]}

def combine_pre_answer_metrics(top: List[Dict[str, Any]], query: str) -> Dict[str, float]:
    if not top:
        return {"coverage_max": 0.0, "coverage_mean": 0.0, "keyword_cov": 0.0, "pre_score": 1.0}
    coverage_max = max(r["score"] for r in top)
    coverage_mean = float(np.mean([r["score"] for r in top]))
    q_terms = content_terms(query)
    keyword_cov = float(np.mean([keyword_coverage(q_terms, r["text"]) for r in top[:4]]))
    risk_cov = 1.0 - max(coverage_max, 0.0) * 0.6 - max(coverage_mean, 0.0) * 0.4
    risk_kw  = 1.0 - keyword_cov
    pre_score = float(np.clip(0.6 * risk_cov + 0.4 * risk_kw, 0.0, 1.0))
    return {"coverage_max": float(coverage_max), "coverage_mean": float(coverage_mean),
            "keyword_cov": float(keyword_cov), "pre_score": pre_score}

def combine_post_answer_metrics(answer: str, top: List[Dict[str, Any]]) -> Dict[str, float]:
    tok, model = get_nli()
    if tok is None or model is None or not top or not answer:
        return {"entailment": None, "contradiction": None, "post_score": None}
    ents, conts = [], []
    for r in top[:4]:
        probs = nli_probs(premise=r["text"], hypothesis=answer)
        if probs:
            ents.append(probs["entailment"])
            conts.append(probs["contradiction"])
    if not ents:
        return {"entailment": None, "contradiction": None, "post_score": None}
    ent = max(ents); con = max(conts) if conts else 0.0
    post_score = float(np.clip(1.0 - 0.8*ent + 0.6*con, 0.0, 1.0))
    return {"entailment": float(ent), "contradiction": float(con), "post_score": post_score}

def decide_action(pre_score: float, post_score: Optional[float]) -> str:
    score = post_score if (post_score is not None) else pre_score
    if score < 0.30: return "ok"
    if score < 0.55: return "more_retrieval"
    return "clarify"

class AnswerabilityReq(BaseModel):
    question: str
    chat_id: Optional[str] = "default"
    k: Optional[int] = 8
    answer: Optional[str] = None

@app.post("/api/predict/answerability")
def predict_answerability(req: AnswerabilityReq, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    chat = store["chats"].get(req.chat_id or "default", {})
    doc_index = chat.get("doc_index") or []

    top = retrieve_top(doc_index, req.question, k=int(req.k or 8)) if doc_index else []
    pre = combine_pre_answer_metrics(top, req.question)

    post = combine_post_answer_metrics(req.answer, top) if (req.answer and req.answer.strip()) else {
        "entailment": None, "contradiction": None, "post_score": None
    }

    risk_score = post["post_score"] if post["post_score"] is not None else pre["pre_score"]
    action = decide_action(pre["pre_score"], post["post_score"])

    return {
        "ok": True,
        "top_used": len(top),
        "pre": pre,
        "post": post,
        "risk_score": float(risk_score),
        "action": action,
        "notes": ("NLI disabled (torch/transformers missing)" if (post["post_score"] is None and req.answer) else None)
    }


# ======================== RERANK / PERSONALIZATION ========================
def get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if CrossEncoder is None:
        return None
    try:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        return _reranker
    except Exception as e:
        print("Reranker init failed:", repr(e))
        return None

def _norm01(x_list: List[float]) -> List[float]:
    if not x_list:
        return []
    lo, hi = min(x_list), max(x_list)
    if hi - lo < 1e-8:
        return [0.5] * len(x_list)
    return [(x - lo) / (hi - lo) for x in x_list]

def _personal_boost(username: str, source: str) -> float:
    prof = load_profile(username)
    cnt = float(prof.get("source_affinity", {}).get(source or "", 0))
    # squash (0..inf) -> (0..0.25) approx
    return float(0.25 * (1 - math.exp(-cnt / 4.0)))

def _record_click(username: str, source: str, positive: bool = True):
    prof = load_profile(username)
    sa = prof.get("source_affinity", {})
    base = sa.get(source or "", 0)
    sa[source or ""] = max(0, base + (1 if positive else -1))
    prof["source_affinity"] = sa
    save_profile(username, prof)

class FeedbackClickReq(BaseModel):
    chat_id: str
    source: Optional[str] = ""
    positive: bool = True

@app.post("/api/feedback/click")
def feedback_click(req: FeedbackClickReq, user=Depends(auth_required)):
    u = user["username"]
    _record_click(u, req.source or "", bool(req.positive))
    return {"ok": True, "source": req.source, "positive": bool(req.positive)}

# Accept both "query" and "question" payloads
class SearchReq(BaseModel):
    chat_id: str = "default"
    question: Optional[str] = None
    query: Optional[str] = None
    k: int = 6
    pool: int = 18  # take top-N by embedding before rerank

@app.post("/api/search")
def personalized_search(req: SearchReq, user=Depends(auth_required)):
    """Retrieve with embeddings -> rerank with cross-encoder -> apply personal boost."""
    u = user["username"]
    store = load_user_chats(u)
    chat = store["chats"].get(req.chat_id or "default", {})
    doc_index = chat.get("doc_index") or []
    if not doc_index:
        return {"results": [], "note": "No indexed documents in this chat. Upload a PDF first."}

    q = (req.question or req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'query' (or 'question')")

    # Stage 1: embedding shortlist
    pool = max(req.k, min(req.pool, 50))
    short = retrieve_top(doc_index, q, k=pool)  # has "score" (cosine)

    # Stage 2: rerank with cross-encoder (optional)
    reranker = get_reranker()
    if reranker:
        pairs = [(q, r["text"]) for r in short]
        try:
            ce_scores = reranker.predict(pairs).tolist()
        except Exception as e:
            print("Rerank failed:", repr(e))
            ce_scores = [0.0] * len(short)
    else:
        ce_scores = [0.0] * len(short)

    # Normalize and combine (ce stronger)
    ce_n = _norm01(ce_scores)
    emb_n = _norm01([r["score"] for r in short])  # cosine -> 0..1
    combined = []
    for i, r in enumerate(short):
        boost = _personal_boost(u, r.get("source", ""))
        final = 0.65 * ce_n[i] + 0.25 * emb_n[i] + 0.10 * boost
        combined.append({**r, "ce": ce_scores[i], "ce_n": ce_n[i], "emb_n": emb_n[i], "boost": boost, "final": final})

    combined.sort(key=lambda x: -x["final"])
    return {"results": combined[:int(req.k)], "used_reranker": bool(reranker)}


# ---- RAG helper: reranked, personalized context for a question ----
def build_personalized_context(username: str, chat_id: str, question: str, k: int = 6, pool: int = 18):
    """
    Returns (topK_results, context_string).
    - topK_results: list[ {text, source, section_idx, final, ...} ]
    - context_string: concatenated, trimmed snippets with lightweight citations
    """
    store = load_user_chats(username)
    chat = store["chats"].get(chat_id or "default", {})
    doc_index = chat.get("doc_index") or []
    if not doc_index:
        return [], ""

    # 1) embed shortlist
    short = retrieve_top(doc_index, question, k=min(max(k, 6), pool))

    # 2) cross-encoder rerank (optional)
    reranker = get_reranker()
    if reranker:
        try:
            pairs = [(question, r["text"]) for r in short]
            ce_scores = reranker.predict(pairs).tolist()
        except Exception as e:
            print("Rerank failed:", repr(e))
            ce_scores = [0.0] * len(short)
    else:
        ce_scores = [0.0] * len(short)

    # 3) combine with personal boost
    ce_n  = _norm01(ce_scores)
    emb_n = _norm01([r["score"] for r in short])
    combined = []
    for i, r in enumerate(short):
        boost = _personal_boost(username, r.get("source", ""))
        final = 0.65 * ce_n[i] + 0.25 * emb_n[i] + 0.10 * boost
        combined.append({**r, "ce": ce_scores[i], "final": float(final), "boost": float(boost)})

    combined.sort(key=lambda x: -x["final"])
    topk = combined[:int(k)]

    # 4) make a compact CONTEXT block (trim each chunk a bit)
    parts = []
    for j, r in enumerate(topk, start=1):
        txt = (r["text"] or "").strip().replace("\n", " ")
        if len(txt) > 900:
            txt = txt[:900] + "…"
        src = r.get("source", "") or "context"
        sec = r.get("section_idx", -1)
        parts.append(f"[{j}] (source: {src}, sec {sec}) {txt}")
    context = "\n".join(parts)

    return topk, context


# ======================== Business KPI Ingest + Forecast ========================
def _user_kpi_path(username: str, ext: str) -> Path:
    return KPI_DIR / f"{username}_kpis.{ext}"

REQUIRED_COLS = {"date", "metric", "value"}

def _read_any_table_from_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(BytesIO(raw))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(BytesIO(raw))
    elif name.endswith(".pdf") and pdfplumber is not None:
        tables = []
        with pdfplumber.open(BytesIO(raw)) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_table()
                    if t and len(t) > 1:
                        hdr = [h.strip().lower() if isinstance(h,str) else h for h in t[0]]
                        rows = t[1:]
                        dfp = pd.DataFrame(rows, columns=hdr)
                        tables.append(dfp)
                except Exception:
                    pass
        if not tables:
            raise HTTPException(status_code=400, detail="No parseable tables found in PDF (try CSV/XLSX).")
        df = pd.concat(tables, ignore_index=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV/XLSX (or PDF with tables).")
    return df

def _canonicalize_kpi(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        for a in ["month","period","dt","timestamp"]:
            if a in df.columns: df = df.rename(columns={a:"date"}); break
    if "metric" not in df.columns:
        for a in ["kpi","measure","name"]:
            if a in df.columns: df = df.rename(columns={a:"metric"}); break
    if "value" not in df.columns:
        for a in ["amount","val","qty","quantity","units"]:
            if a in df.columns: df = df.rename(columns={a:"value"}); break

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date","metric","value"])
    return df

def _save_user_kpis(username: str, df: pd.DataFrame):
    df = df.sort_values("date")
    seg_cols = [c for c in df.columns if c not in {"date","metric","value"}]
    if seg_cols:
        df["__key__"] = df["date"].astype(str) + "||" + df["metric"] + "||" + df[seg_cols].astype(str).agg("|".join, axis=1)
    else:
        df["__key__"] = df["date"].astype(str) + "||" + df["metric"]

    if HAVE_PARQUET and _user_kpi_path(username, "parquet").exists():
        old = pd.read_parquet(_user_kpi_path(username, "parquet"))
    elif _user_kpi_path(username, "csv").exists():
        old = pd.read_csv(_user_kpi_path(username, "csv"))
    else:
        old = pd.DataFrame(columns=df.columns)

    if not old.empty and "__key__" not in old.columns:
        seg_cols_old = [c for c in old.columns if c not in {"date","metric","value"}]
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        if seg_cols_old:
            old["__key__"] = old["date"].astype(str) + "||" + old["metric"].astype(str) + "||" + old[seg_cols_old].astype(str).agg("|".join, axis=1)
        else:
            old["__key__"] = old["date"].astype(str) + "||" + old["metric"].astype(str)

    all_df = pd.concat([old, df], ignore_index=True)
    all_df = all_df.drop_duplicates("__key__", keep="last").drop(columns="__key__", errors="ignore")

    if HAVE_PARQUET:
        all_df.to_parquet(_user_kpi_path(username, "parquet"), index=False)
    else:
        all_df.to_csv(_user_kpi_path(username, "csv"), index=False)

def _load_user_kpis(username: str) -> pd.DataFrame:
    if HAVE_PARQUET and _user_kpi_path(username, "parquet").exists():
        return pd.read_parquet(_user_kpi_path(username, "parquet"))
    pcsv = _user_kpi_path(username, "csv")
    if pcsv.exists():
        return pd.read_csv(pcsv, parse_dates=["date"])
    raise HTTPException(status_code=404, detail="No KPI data uploaded yet.")

def _to_monthly(series: pd.Series) -> pd.Series:
    return series.resample("MS").sum()

def _snaive_forecast(y: pd.Series, steps: int, season: int = 12) -> pd.DataFrame:
    if len(y) < season:
        last = y.iloc[-1] if len(y) else 0.0
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        fc = pd.Series([last]*steps, index=idx)
    else:
        template = y.iloc[-season:]
        reps = int(np.ceil(steps/season))
        vals = np.tile(template.values, reps)[:steps]
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        fc = pd.Series(vals, index=idx)
    res = y - y.shift(12)
    s = float(np.nanstd(res.dropna())) if res.notna().sum() > 3 else float(np.nanstd(y.diff().dropna()))
    out = pd.DataFrame({"yhat": fc, "yhat_lower": fc - 1.28*s, "yhat_upper": fc + 1.28*s})
    return out

def _fit_sarimax_and_forecast(y: pd.Series, steps: int) -> pd.DataFrame:
    if SARIMAX is None:
        return _snaive_forecast(y, steps=steps, season=12)
    try:
        m = 12
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,m), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps)
        mean = pred.predicted_mean.clip(lower=0)
        ci = pred.conf_int(alpha=0.20)
        lower = ci.iloc[:,0].clip(lower=0)
        upper = ci.iloc[:,1].clip(lower=0)
        out = pd.DataFrame({"yhat": mean, "yhat_lower": lower, "yhat_upper": upper})
        return out
    except Exception:
        return _snaive_forecast(y, steps=steps, season=12)

class KPIIngestSummary(BaseModel):
    rows: int
    metrics: List[str]
    date_min: str
    date_max: str
    segments: List[str]

@app.post("/api/metrics/ingest")
async def ingest_metrics(file: UploadFile = File(...), user=Depends(auth_required)):
    u = user["username"]
    try:
        raw = await file.read()
        df = _read_any_table_from_bytes(raw, file.filename or "upload")
        df = _canonicalize_kpi(df)
        _save_user_kpis(u, df)
        segs = [c for c in df.columns if c not in {"date","metric","value"}]
        summary = KPIIngestSummary(
            rows = int(len(df)),
            metrics = sorted(df["metric"].unique().tolist()),
            date_min = df["date"].min().date().isoformat(),
            date_max = df["date"].max().date().isoformat(),
            segments = segs
        )
        return {"ok": True, "summary": summary.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Ingest failed: {e}"})

class ForecastReq(BaseModel):
    metric: str
    horizon_months: int = 6
    groupby: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None

@app.post("/api/forecast/kpi")
def forecast_kpi(req: ForecastReq, user=Depends(auth_required)):
    u = user["username"]
    df = _load_user_kpis(u)

    metric = req.metric.strip().lower()
    dff = df[df["metric"].str.lower() == metric].copy()
    if dff.empty:
        raise HTTPException(status_code=404, detail=f"No rows for metric='{metric}'")

    if req.filters:
        for k, v in req.filters.items():
            if k not in dff.columns:
                raise HTTPException(status_code=400, detail=f"Unknown filter field: {k}")
            dff = dff[dff[k].astype(str) == str(v)]
        if dff.empty:
            raise HTTPException(status_code=404, detail="No rows after applying filters.")

    seg_cols = [c for c in (req.groupby or []) if c in dff.columns]
    groups = dff.groupby(seg_cols, dropna=False) if seg_cols else [((), dff)]
    horizon = int(max(1, min(req.horizon_months, 36)))
    results = []

    for seg_key, gdf in groups:
        gdf = gdf[["date","value"]].dropna().sort_values("date")
        if gdf.empty:
            continue
        y = gdf.set_index("date")["value"].asfreq("D")
        y = _to_monthly(y).dropna()
        if len(y) == 0:
            continue

        fc = _fit_sarimax_and_forecast(y, steps=horizon)

        seg_dict = {}
        if seg_cols:
            if isinstance(seg_key, tuple):
                for k, v in zip(seg_cols, seg_key):
                    seg_dict[k] = None if (v is np.nan) else v
            else:
                seg_dict[seg_cols[0]] = seg_key

        results.append({
            "segment": seg_dict,
            "history_start": y.index.min().date().isoformat(),
            "history_end": y.index.max().date().isoformat(),
            "forecast": [
                {
                    "date": d.strftime("%Y-%m-01"),
                    "yhat": float(fc.loc[d,"yhat"]),
                    "yhat_lower": float(fc.loc[d,"yhat_lower"]),
                    "yhat_upper": float(fc.loc[d,"yhat_upper"]),
                }
                for d in fc.index
            ]
        })

    return {"ok": True, "metric": metric, "horizon_months": horizon, "segments": results}


# ======================== ROUTER (Intent Classification) ========================
def get_router_zs():
    """Zero-shot text classifier (if transformers+torch available)."""
    global _router_zs
    if _router_zs is not None:
        return _router_zs
    if hf_pipeline is None or torch is None:
        return None
    try:
        _router_zs = hf_pipeline("zero-shot-classification", model=ROUTER_ZS_MODEL)
        return _router_zs
    except Exception as e:
        print("Router pipeline init failed:", repr(e))
        return None

LABELS = ["chat", "pdfqa", "vision", "image"]

ROUTER_RULES = [
    (re.compile(r"\b(upload(ed)?|attach(ed)?)\b.*\b(pdf|doc|file|report)\b", re.I), "pdfqa"),
    (re.compile(r"\b(pdf|report|document)\b", re.I), "pdfqa"),
    (re.compile(r"\b(generate|make|create|draw|render|logo|poster|banner|image|picture|photo)\b", re.I), "image"),
    (re.compile(r"\b(analy[sz]e|describe)\b.*\b(image|photo|picture)\b", re.I), "vision"),
]

class RouteReq(BaseModel):
    # allow both {text: "..."} and {input: "..."} from different frontends
    text: Optional[str] = None
    input: Optional[str] = None
    has_pdf_index: Optional[bool] = None
    chat_id: Optional[str] = "default"

def _route_decide(text: str, has_pdf_index: Optional[bool]) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {"intent": "chat", "confidence": 0.0, "method": "default"}
    for pat, lab in ROUTER_RULES:
        if pat.search(t):
            return {"intent": lab, "confidence": 0.85, "method": "rules"}
    if has_pdf_index:
        return {"intent": "pdfqa", "confidence": 0.60, "method": "context_hint"}
    clf = get_router_zs()
    if clf:
        out = clf(t, candidate_labels=LABELS, multi_label=False)
        labels = out.get("labels", [])
        scores = out.get("scores", [])
        if labels:
            return {"intent": labels[0], "confidence": float(scores[0]), "method": "zero-shot"}
    return {"intent": "chat", "confidence": 0.50, "method": "default"}

# Frontend compatibility: accept both /api/route and /api/route/intent
@app.post("/api/route")
@app.post("/api/route/intent")
def route_intent(req: RouteReq, user=Depends(auth_required)):
    text = (req.text or req.input or "")
    return _route_decide(text, req.has_pdf_index)


# ======================== NEXT-QUESTION PREDICTION ========================
class NextQReq(BaseModel):
    chat_id: str = "default"
    n: int = 3

def _clean_and_pick_unique(items: List[str], k: int) -> List[str]:
    out, seen = [], set()
    for s in items:
        s = (s or "").strip()
        # remove fence artifacts / trailing commas / quotes
        s = re.sub(r"^[-*]\s*", "", s)
        s = re.sub(r"^\d+\.\s*", "", s)
        s = s.strip().strip('"').strip("'").rstrip(",").strip()
        if not s or len(s) < 3:  # skip empties
            continue
        if s.lower() in ("json", "[", "]"):
            continue
        if s.lower().startswith(("here are", "return only", "number of suggestions")):
            continue
        if s not in seen:
            out.append(s); seen.add(s)
        if len(out) >= k:
            break
    return out

def _extract_json_array(text: str, k: int) -> List[str]:
    """Try hard to pull a JSON array of strings from model output."""
    t = (text or "").strip()

    # 1) strip common fences like ```json ... ```
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```\s*$", "", t, flags=re.I)

    # 2) try direct JSON parse
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return _clean_and_pick_unique([str(x) for x in parsed], k)
        if isinstance(parsed, dict) and "suggestions" in parsed:
            return _clean_and_pick_unique([str(x) for x in parsed["suggestions"]], k)
    except Exception:
        pass

    # 3) find the first [...last] span and parse that
    l, r = t.find("["), t.rfind("]")
    if l != -1 and r != -1 and l < r:
        span = t[l:r+1]
        try:
            parsed = json.loads(span)
            if isinstance(parsed, list):
                return _clean_and_pick_unique([str(x) for x in parsed], k)
        except Exception:
            pass

    # 4) fallback: pull quoted strings
    quoted = re.findall(r'"([^"\n\r]{3,200})"', t)
    cleaned = _clean_and_pick_unique(quoted, k)
    if cleaned:
        return cleaned

    # 5) last resort: split lines
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return _clean_and_pick_unique(lines, k)

# Frontend compatibility: accept both /api/next-questions and /api/predict/next_questions
@app.post("/api/next-questions")
@app.post("/api/predict/next_questions")
async def next_questions(req: NextQReq, user=Depends(auth_required)):
    """Suggest next 2–3 follow-ups using the chat transcript."""
    u = user["username"]
    store = load_user_chats(u)
    chat = store.get("chats", {}).get(req.chat_id or "default", {})
    history = chat.get("messages", [])
    transcript = []
    for m in history[-12:]:
        who = "User" if m.get("role") == "user" else "Assistant"
        transcript.append(f"{who}: {m.get('content','')}")
    transcript_text = "\n".join(transcript) if transcript else "(no prior context)"

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY for chat model.")
    llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))

    k = max(2, min(5, int(req.n)))
    prompt = (
        "You are a helpful assistant that suggests follow-up questions.\n"
        "Given the conversation, produce ONLY a JSON array of 2–5 concise strings.\n"
        "Do NOT include any explanation, markdown, or code fences. "
        "Do NOT include the word 'json'. Return the array directly.\n\n"
        f"Conversation:\n{transcript_text}\n\n"
        f"Array length: {k}"
    )

    raw = await run_in_threadpool(llm.predict, prompt)
    text = (raw or "").strip()

    ideas = _extract_json_array(text, k)

    # generic fallback if still empty
    if not ideas:
        ideas = [
            "Summarize what we discussed so far",
            "What should I do next based on this?",
            "Any risks or blockers I should watch for?"
        ][:k]

    return {"suggestions": ideas}
