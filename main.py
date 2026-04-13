# AI tools used: Claude (Anthropic) assisted with FastAPI endpoint
# structure, CORS configuration, and static file serving setup.
"""
CineStyle — FastAPI backend entry point.

Endpoints:
  POST /identify   — receives an image crop, returns garment attributes + embedding
  POST /recommend  — receives an embedding, returns ranked product recommendations
  GET  /health     — liveness check
"""

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import numpy as np
from PIL import Image

from scripts.build_features import embed_image
from scripts.model import recommend

app = FastAPI(title="CineStyle", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cinestyle.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/static",
    StaticFiles(directory="data/raw/crops", check_directory=False),
    name="crops",
)

class GarmentResponse(BaseModel):
    garment_type: str
    color: str
    aesthetic: str
    embedding: list[float]


class RecommendRequest(BaseModel):
    embedding: list[float]
    top_k: int = 12
    price_min: float | None = None
    price_max: float | None = None


class ProductCard(BaseModel):
    id: str
    title: str
    brand: str
    price: float
    image_url: str
    product_url: str
    similarity: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/identify", response_model=GarmentResponse)
async def identify(file: UploadFile = File(...)):
    """
    Accept a cropped garment image.
    Returns FashionCLIP attributes and the 512-dim embedding.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    result = embed_image(image)
    return GarmentResponse(**result)


@app.post("/recommend", response_model=list[ProductCard])
def get_recommendations(req: RecommendRequest):
    """
    Given a garment embedding, run FAISS retrieval + NCF re-ranking.
    Returns ranked product cards.
    """
    embedding = np.array(req.embedding, dtype=np.float32)
    products = recommend(
        embedding,
        top_k=req.top_k,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    return products
