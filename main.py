import ast
import os
import time

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from supabase import create_client

app = FastAPI()

# =========================
# 🔹 CONFIG
# =========================


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Missing Supabase env variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SIMILARITY_THRESHOLD = 0.5


# =========================
# 🔹 UTILS
# =========================
def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    return xff.split(",")[0] if xff else request.client.host


def parse_embedding(embedding):
    if embedding is None:
        return None

    if isinstance(embedding, str):
        embedding = ast.literal_eval(embedding)

    return np.array(embedding, dtype=float)


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def extract_embedding(image_bytes: bytes):
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return None

        reps = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )

        if not reps or len(reps) != 1:
            return None

        return np.array(reps[0]["embedding"], dtype=float)

    except Exception as e:
        print("DeepFace error:", e)
        return None


# =========================
# 🔹 SERVICES
# =========================
def upload_image(file_bytes: bytes, email: str) -> str:
    file_name = f"avatars/{email}_{int(time.time())}.jpg"

    supabase.storage.from_("avatars").upload(
        file_name,
        file_bytes,
        {"content-type": "image/jpeg"}
    )

    return supabase.storage.from_("avatars").get_public_url(file_name)


def get_user_by_email(email: str):
    res = (
        supabase.table("users")
        .select("id, embedding")
        .eq("email", email)
        .single()
        .execute()
    )
    return res.data


# =========================
# 🔹 ENDPOINTS
# =========================
@app.post("/register-user")
async def register_user(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    file: UploadFile = File(...)
):
    start = time.time()

    image_bytes = await file.read()
    embedding = extract_embedding(image_bytes)

    if embedding is None:
        raise HTTPException(status_code=400, detail="Debe haber exactamente 1 rostro")

    try:
        image_url = upload_image(image_bytes, email)

        res = supabase.table("users").insert({
            "name": name,
            "email": email,
            "phone": phone,
            "profile_image_url": image_url,
            "embedding": embedding.tolist()
        }).execute()

        return {
            "success": True,
            "latency_ms": int((time.time() - start) * 1000)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-face")
async def match_face(
    request: Request,
    email: str = Form(...),
    file: UploadFile = File(...)
):
    start = time.time()
    client_ip = get_client_ip(request)

    image_bytes = await file.read()
    query_embedding = extract_embedding(image_bytes)

    if query_embedding is None:
        raise HTTPException(status_code=400, detail="No se detectó rostro")

    try:
        user = get_user_by_email(email)

        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        db_embedding = parse_embedding(user["embedding"])

        if db_embedding is None:
            raise HTTPException(status_code=400, detail="Usuario sin embedding")

        similarity = cosine_similarity(db_embedding, query_embedding)
        is_match = similarity >= SIMILARITY_THRESHOLD

        return {
            "success": is_match,
            "user_id": user["id"],
            "confidence": similarity,
            "model_latency_ms": int((time.time() - start) * 1000),
            "ip_address": client_ip
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def root():
    return {
        "service": "Face Recognition API",
        "status": "running",
        "version": "1.0.0"
    }