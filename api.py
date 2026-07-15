# api.py
import os
import time
import uuid
import json
import logging
import contextlib
from pathlib import Path
import joblib
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Depends, status, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from supabase import create_client, Client
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# =====================================================================
# 1. ENVIRONMENT & HARDWARE INITIALIZATION
# =====================================================================
load_dotenv()

# Setup hardware device for local AI models
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_label = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_label = "NVIDIA GPU (CUDA)"
else:
    device = torch.device("cpu")
    device_label = "CPU"

# Avoid OpenMP warnings/crashes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Supabase Credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase_client: Client = None
if url and key:
    try:
        supabase_client = create_client(url, key)
    except Exception as e:
        print(f"⚠️ Initial connection to Supabase failed: {e}")

# =====================================================================
# 2. STRUCTURED LOGGING CONFIGURATION
# =====================================================================
class JsonFormatter(logging.Formatter):
    """
    Custom formatter converting log records into single-line JSON strings.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "SYSTEM")
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

logger = logging.getLogger("fraud_pipeline")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# =====================================================================
# 3. API KEY SECURITY SCHEME
# =====================================================================
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(header_value: str = Depends(api_key_header)):
    expected_key = os.getenv("INFERENCE_API_KEY", "dev-secret-key-123")
    
    if not header_value:
        logger.warning("Access Denied: Request missing X-API-Key identification header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key validation header."
        )
        
    if header_value != expected_key:
        logger.warning("Access Denied: Unauthorized or mismatched key token supplied.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized API key."
        )
        
    return header_value

# =====================================================================
# 4. SCHEMAS & GLOBAL STATE
# =====================================================================
class TransactionPayload(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

class QuestionRequest(BaseModel):
    question: str

# Global registries for local inference models
ml_models_registry = {}
local_nlp_models = {}

# =====================================================================
# 5. ASYNC BACKGROUND TASKS & LIFESPAN
# =====================================================================
def persist_prediction_task(payload_data: dict, latency: float, prob: float, label: str, algo: str, strategy: str, threshold: float):
    if supabase_client is None:
        logger.error("❌ DATABASE INSERTION ABORTED: Supabase client is not initialized.")
        return
    try:
        record = {
            "algo": algo,
            "strategy": strategy,
            "amount": float(payload_data.get("Amount", 0.0)),
            "fraud_probability": prob,
            "label": label,
            "latency_ms": latency,
            "threshold": threshold
        }
        supabase_client.table("predictions").insert(record).execute()
        logger.info("🚀 Successfully pushed telemetry row to Supabase!")
    except Exception as e:
        logger.error(f"❌ Asynchronous Telemetry Write Failed: {e}")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # --- PHASE 1: Load Traditional Machine Learning Models ---
    logger.info("⏳ Warming up local ML models into RAM cache...")
    from model_io import ALGORITHMS, STRATEGIES, get_model_path
    for algo in ALGORITHMS:
        ml_models_registry[algo] = {}
        for strategy in STRATEGIES:
            path = get_model_path(algo, strategy)
            if path.exists():
                try:
                    ml_models_registry[algo][strategy] = joblib.load(path)
                except Exception as e:
                    logger.error(f"⚠️ Failed to load ML model file at {path}: {e}")

    # --- PHASE 2: Load Sovereign NLP Models (For RAG) ---
    logger.info(f"⏳ Booting Local RAG Models on device: {device_label}...")
    try:
        # Load embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        local_nlp_models["embeddings"] = embedding_model

        # Load tokenizers & model instances cleanly
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        ).to(device)

        # Register generator pipeline
        local_nlp_models["generator"] = pipeline(
            "text-generation", 
            model=llm_model, 
            tokenizer=tokenizer,
            device=device
        )
        local_nlp_models["tokenizer"] = tokenizer
        logger.info("🟢 Local RAG engine warmed up successfully.")
    except Exception as e:
        logger.error(f"🔴 Local RAG engine setup failed: {e}")

    yield
    # --- PHASE 3: Shutdown ---
    logger.info("🛑 Draining memory caches and shutting down API routing...")
    ml_models_registry.clear()
    local_nlp_models.clear()


# =====================================================================
# 6. FASTAPI SETUP & MIDDLEWARE
# =====================================================================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fraud Inference & Sovereign RAG Production Engine", version="1.3.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    extra = {"request_id": request_id}
    logger.info(f"Incoming request: {request.method} {request.url.path}", extra=extra)
    
    start_time = time.time()
    response = await call_next(request)
    latency = (time.time() - start_time) * 1000
    
    logger.info(f"Completed request with status {response.status_code} in {latency:.2f}ms", extra=extra)
    response.headers["X-Request-ID"] = request_id
    return response

# =====================================================================
# 7. ROUTING ENDPOINTS
# =====================================================================
@app.get("/")
def read_root():
    return {
        "message": "Unified Fraud API & Sovereign RAG is Live & Operational!",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    ml_count = sum(len(ml_models_registry[a]) for a in ml_models_registry)
    rag_warm = "generator" in local_nlp_models
    return {
        "status": "Healthy", 
        "models_cached_count": ml_count,
        "local_rag_active": rag_warm,
        "hardware_accelerator": str(device).upper()
    }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def predict_transaction(
    payload: TransactionPayload,
    request: Request, 
    background_tasks: BackgroundTasks,
    algo: str = Query("Random Forest"),
    strategy: str = Query("None"),
    threshold: float = Query(0.50, ge=0.0, le=1.0)
):
    start_process_time = time.time()
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    extra = {"request_id": req_id}

    if algo not in ml_models_registry or strategy not in ml_models_registry[algo]:
        logger.warning(f"Prediction aborted: Setup '{algo} ({strategy})' not found in RAM cache.", extra=extra)
        raise HTTPException(status_code=404, detail=f"Requested setup '{algo} ({strategy})' not cached.")
        
    cached = ml_models_registry[algo][strategy]
    input_df = pd.DataFrame([payload.model_dump()])
    model_obj = cached["model"] if isinstance(cached, dict) and "model" in cached else cached
    
    try:
        fraud_prob = float(model_obj.predict_proba(input_df)[0, 1])
    except Exception as e:
        logger.error(f"Scoring engine encountered an execution critical crash: {str(e)}", extra=extra)
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {str(e)}")
    
    prediction_label = "Fraud" if fraud_prob >= threshold else "Legit"
    elapsed_latency_ms = (time.time() - start_process_time) * 1000
    
    logger.info(
        f"Inference complete | Model: {algo} ({strategy}) | Amount: {payload.Amount} | "
        f"Prob: {fraud_prob:.4f} | Label: {prediction_label}", 
        extra=extra
    )
    
    # DISPATCH PERSISTENCE LOG ASYNC TASK
    background_tasks.add_task(
        persist_prediction_task,
        payload_data=payload.model_dump(),
        latency=elapsed_latency_ms,
        prob=fraud_prob,
        label=prediction_label,
        algo=algo,
        strategy=strategy,
        threshold=threshold
    )
    
    return {
        "prediction": prediction_label,
        "fraud_probability": fraud_prob,
        "applied_threshold": threshold,
        "model_used": f"{algo} ({strategy})"
    }


@app.get("/models")
def get_leaderboard_matrix():
    leaderboard = []
    for algo in ml_models_registry:
        for strategy, data in ml_models_registry[algo].items():
            metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
            leaderboard.append({
                "algorithm": algo,
                "strategy": strategy,
                "f1_score_fraud": metrics.get("f1_fraud", metrics.get("f1_score_fraud", 0.0)),
                "train_rows": metrics.get("train_rows", "N/A"),
                "test_rows": metrics.get("test_rows", "N/A")
            })
    return sorted(leaderboard, key=lambda x: x.get("f1_score_fraud", 0.0), reverse=True)


@app.post("/ask")
async def ask_rag(payload: QuestionRequest, request: Request):
    """
    Sovereign RAG endpoint utilizing local MiniLM + SmolLM2 and pgvector match_chunks
    """
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    extra = {"request_id": req_id}
    
    question = payload.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Validate model availability
    if "embeddings" not in local_nlp_models or "generator" not in local_nlp_models:
        raise HTTPException(
            status_code=503, 
            detail="Local RAG models are currently warm-booting or offline. Please try again in a moment."
        )

    try:
        # Step A: Local embedding generation
        query_vector = local_nlp_models["embeddings"].encode(question).tolist()

        # Step B: Match chunks in Supabase
        matched_res = supabase_client.rpc(
            "match_chunks",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.1,  # Lower similarity threshold for compact models
                "match_count": 3
            }
        ).execute()
        
        contexts = matched_res.data
    except Exception as e:
        logger.error(f"Error querying context: {e}", extra=extra)
        raise HTTPException(status_code=500, detail="Database RAG search failure.")

    if not contexts:
        return {
            "answer": "I found no documentation or matching log sections in our system database to answer this.",
            "citations": []
        }

    # Step C: Context Construction
    context_text = ""
    citations = []
    for idx, chunk in enumerate(contexts, 1):
        source = chunk["source_file"] if not chunk["source_url"] else chunk["source_url"]
        citations.append({
            "id": idx,
            "source": source,
            "excerpt": chunk["content"][:150] + "..."
        })
        context_text += f"\n[Context {idx}]: {chunk['content']}\n"

    # Step D: Render prompt template using SmolLM2 Chat markup
    tokenizer = local_nlp_models["tokenizer"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an operations assistant. Use the context blocks to answer the user's question concisely. "
                "Only output facts present in the context. Keep answers to under 3 paragraphs."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}"
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Step E: Inference execution
    try:
        outputs = local_nlp_models["generator"](
            prompt,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        raw_text = outputs[0]["generated_text"]
        answer = raw_text[len(prompt):].strip()
    except Exception as e:
        logger.error(f"Local LLM inference failure: {e}", extra=extra)
        raise HTTPException(status_code=500, detail="Local LLM inference engine failure.")

    return {
        "answer": answer,
        "citations": citations
    }