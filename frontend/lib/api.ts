/**
 * CineStyle API client — talks to the FastAPI backend.
 */

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export interface GarmentResult {
  garment_type: string;
  color: string;
  aesthetic: string;
  embedding: number[];
}

export interface ProductCard {
  id: string;
  title: string;
  brand: string;
  price: number;
  image_url: string;
  product_url: string;
  similarity: number;
}

export interface RecommendRequest {
  embedding: number[];
  top_k?: number;
  price_min?: number | null;
  price_max?: number | null;
}

/**
 * Send a cropped image blob to /identify.
 * Returns garment attributes + 512-dim embedding.
 */
export async function identify(blob: Blob): Promise<GarmentResult> {
  const form = new FormData();
  form.append("file", blob, "crop.jpg");
  const res = await fetch(`${API_BASE}/identify`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`identify failed (${res.status}): ${err}`);
  }
  return res.json();
}

/**
 * Send an embedding to /recommend.
 * Returns ranked product cards.
 */
export async function recommend(req: RecommendRequest): Promise<ProductCard[]> {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`recommend failed (${res.status}): ${err}`);
  }
  return res.json();
}
