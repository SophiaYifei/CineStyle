"use client";

import { useState, useEffect, useCallback } from "react";
import type { GarmentResult, ProductCard as ProductCardType } from "@/lib/api";
import { recommend } from "@/lib/api";
import ProductCard from "./ProductCard";

interface ResultPanelProps {
  garment: GarmentResult;
  cropPreview: string;
  onClose: () => void;
}

export default function ResultPanel({
  garment,
  cropPreview,
  onClose,
}: ResultPanelProps) {
  const [products, setProducts] = useState<ProductCardType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [priceMin, setPriceMin] = useState<number>(0);
  const [priceMax, setPriceMax] = useState<number>(500);
  const [maxCeiling, setMaxCeiling] = useState(500);

  const fetchRecommendations = useCallback(
    async (pMin?: number, pMax?: number) => {
      setLoading(true);
      setError(null);
      try {
        const res = await recommend({
          embedding: garment.embedding,
          top_k: 12,
          price_min: pMin ?? null,
          price_max: pMax ?? null,
        });
        setProducts(res);
        // Set price ceiling from first unfiltered fetch
        if (pMin === undefined && res.length > 0) {
          const highest = Math.ceil(
            Math.max(...res.map((p) => p.price)) / 10
          ) * 10;
          setMaxCeiling(highest || 500);
          setPriceMax(highest || 500);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load recommendations");
      } finally {
        setLoading(false);
      }
    },
    [garment.embedding]
  );

  // Initial fetch (unfiltered)
  useEffect(() => {
    fetchRecommendations();
  }, [fetchRecommendations]);

  const handleFilter = () => {
    fetchRecommendations(priceMin, priceMax);
  };

  return (
    <div
      className="slide-in"
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: "min(480px, 100vw)",
        height: "100vh",
        background: "var(--bg)",
        borderLeft: "1px solid var(--border)",
        zIndex: 50,
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "20px 24px 0",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <h2
          style={{
            fontSize: "0.7rem",
            textTransform: "uppercase",
            letterSpacing: "0.2em",
            color: "var(--amber)",
            margin: 0,
          }}
        >
          Identification
        </h2>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "1px solid var(--border)",
            color: "var(--text-muted)",
            width: "28px",
            height: "28px",
            borderRadius: "6px",
            cursor: "pointer",
            fontSize: "1rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            lineHeight: 1,
          }}
        >
          &times;
        </button>
      </div>

      {/* Garment info */}
      <div style={{ padding: "20px 24px" }}>
        <div style={{ display: "flex", gap: "16px", marginBottom: "20px" }}>
          {/* Crop preview */}
          <div
            style={{
              width: "80px",
              height: "80px",
              borderRadius: "8px",
              overflow: "hidden",
              flexShrink: 0,
              border: "1px solid var(--border)",
            }}
          >
            <img
              src={cropPreview}
              alt="Cropped garment"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
          </div>

          {/* Labels */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", justifyContent: "center" }}>
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <Tag label={garment.garment_type} />
              <Tag label={garment.color} />
            </div>
            <p
              style={{
                color: "var(--text-muted)",
                fontSize: "0.8rem",
                margin: 0,
                fontStyle: "italic",
              }}
            >
              {garment.aesthetic}
            </p>
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: "1px", background: "var(--border)", margin: "0 0 20px" }} />

        {/* Price filter */}
        <div style={{ marginBottom: "20px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "12px",
            }}
          >
            <span
              style={{
                fontSize: "0.65rem",
                textTransform: "uppercase",
                letterSpacing: "0.15em",
                color: "var(--text-dim)",
              }}
            >
              Price Range
            </span>
            <span
              style={{
                fontSize: "0.8rem",
                fontFamily: "var(--font-mono), monospace",
                color: "var(--text-muted)",
              }}
            >
              ${priceMin} &ndash; ${priceMax}
            </span>
          </div>

          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <input
              type="range"
              min={0}
              max={maxCeiling}
              step={5}
              value={priceMin}
              onChange={(e) => {
                const v = Number(e.target.value);
                setPriceMin(Math.min(v, priceMax - 5));
              }}
              style={{ flex: 1 }}
            />
            <input
              type="range"
              min={0}
              max={maxCeiling}
              step={5}
              value={priceMax}
              onChange={(e) => {
                const v = Number(e.target.value);
                setPriceMax(Math.max(v, priceMin + 5));
              }}
              style={{ flex: 1 }}
            />
          </div>

          <button
            onClick={handleFilter}
            disabled={loading}
            style={{
              marginTop: "12px",
              width: "100%",
              padding: "8px",
              background: loading ? "var(--surface2)" : "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              color: loading ? "var(--text-dim)" : "var(--amber)",
              fontSize: "0.75rem",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              cursor: loading ? "default" : "pointer",
              transition: "border-color 0.2s",
            }}
            onMouseEnter={(e) => {
              if (!loading) e.currentTarget.style.borderColor = "var(--amber-dim)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--border)";
            }}
          >
            {loading ? "Loading..." : "Apply Filter"}
          </button>
        </div>

        {/* Divider */}
        <div style={{ height: "1px", background: "var(--border)", margin: "0 0 16px" }} />

        {/* Section header */}
        <h3
          style={{
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.2em",
            color: "var(--text-dim)",
            marginBottom: "16px",
          }}
        >
          Recommendations
          {!loading && (
            <span style={{ color: "var(--text-dim)", marginLeft: "8px" }}>
              ({products.length})
            </span>
          )}
        </h3>

        {/* Error */}
        {error && (
          <div
            style={{
              padding: "12px 16px",
              background: "rgba(220,38,38,0.08)",
              border: "1px solid rgba(220,38,38,0.2)",
              borderRadius: "8px",
              color: "#fca5a5",
              fontSize: "0.8rem",
              marginBottom: "16px",
            }}
          >
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, 1fr)",
              gap: "12px",
            }}
          >
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="shimmer"
                style={{
                  borderRadius: "10px",
                  aspectRatio: "3 / 5",
                }}
              />
            ))}
          </div>
        )}

        {/* Product grid */}
        {!loading && products.length > 0 && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, 1fr)",
              gap: "12px",
              paddingBottom: "32px",
            }}
          >
            {products.map((p, i) => (
              <ProductCard key={p.id} product={p} index={i} />
            ))}
          </div>
        )}

        {/* Empty state */}
        {!loading && products.length === 0 && !error && (
          <div
            style={{
              textAlign: "center",
              padding: "40px 20px",
              color: "var(--text-dim)",
              fontSize: "0.85rem",
            }}
          >
            No products found in this price range.
          </div>
        )}
      </div>
    </div>
  );
}

function Tag({ label }: { label: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "3px 10px",
        borderRadius: "4px",
        background: "var(--surface2)",
        border: "1px solid var(--border)",
        color: "var(--text)",
        fontSize: "0.75rem",
        letterSpacing: "0.05em",
        textTransform: "capitalize",
      }}
    >
      {label}
    </span>
  );
}
