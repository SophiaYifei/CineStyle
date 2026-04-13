"use client";

import { useState } from "react";
import type { ProductCard as ProductCardType } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Extract filename from image_url like "data/raw/crops/12345.jpg" → full static URL */
function toStaticUrl(imageUrl: string): string {
  const filename = imageUrl.split("/").pop() ?? imageUrl;
  return `${API_BASE}/static/${filename}`;
}

interface ProductCardProps {
  product: ProductCardType;
  index: number;
}

export default function ProductCard({ product, index }: ProductCardProps) {
  const [imgError, setImgError] = useState(false);
  const similarity = Math.round(product.similarity * 100);

  return (
    <div
      className="fade-up"
      style={{
        animationDelay: `${index * 0.06}s`,
        background: "var(--surface)",
        borderRadius: "10px",
        border: "1px solid var(--border)",
        overflow: "hidden",
        transition: "border-color 0.2s, transform 0.2s",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "var(--amber-dim)";
        e.currentTarget.style.transform = "translateY(-2px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "var(--border)";
        e.currentTarget.style.transform = "translateY(0)";
      }}
      onClick={() => {
        if (product.product_url) window.open(product.product_url, "_blank");
      }}
    >
      {/* Product image */}
      <div
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "3 / 4",
          background: "var(--surface2)",
          overflow: "hidden",
        }}
      >
        {!imgError ? (
          <img
            src={toStaticUrl(product.image_url)}
            alt={product.title}
            loading="lazy"
            onError={() => setImgError(true)}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              display: "block",
            }}
          />
        ) : (
          <div
            style={{
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-dim)",
              fontSize: "0.75rem",
              letterSpacing: "0.08em",
            }}
          >
            NO IMAGE
          </div>
        )}

        {/* Similarity badge */}
        <div
          style={{
            position: "absolute",
            top: "8px",
            right: "8px",
            background: "rgba(12,10,9,0.85)",
            border: "1px solid var(--amber-dim)",
            color: "var(--amber-light)",
            fontSize: "0.7rem",
            fontFamily: "var(--font-mono), monospace",
            padding: "2px 7px",
            borderRadius: "4px",
            letterSpacing: "0.05em",
          }}
        >
          {similarity}%
        </div>
      </div>

      {/* Info */}
      <div style={{ padding: "12px 14px 14px" }}>
        <p
          style={{
            color: "var(--text-muted)",
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            marginBottom: "4px",
          }}
        >
          {product.brand}
        </p>
        <p
          style={{
            color: "var(--text)",
            fontSize: "0.85rem",
            lineHeight: "1.3",
            marginBottom: "8px",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}
        >
          {product.title}
        </p>
        <p
          style={{
            color: "var(--amber)",
            fontSize: "0.95rem",
            fontFamily: "var(--font-mono), monospace",
            fontWeight: 600,
          }}
        >
          ${product.price.toFixed(2)}
        </p>
      </div>
    </div>
  );
}
