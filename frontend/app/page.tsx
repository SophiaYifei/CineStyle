"use client";

import { useState, useCallback } from "react";
import FrameCapture from "./components/FrameCapture";
import GarmentHighlight from "./components/GarmentHighlight";
import ResultPanel from "./components/ResultPanel";
import { identify } from "@/lib/api";
import type { GarmentResult } from "@/lib/api";

type Stage = "upload" | "select" | "results";

export default function Home() {
  const [stage, setStage] = useState<Stage>("upload");
  const [frameDataUrl, setFrameDataUrl] = useState("");
  const [identifying, setIdentifying] = useState(false);
  const [garment, setGarment] = useState<GarmentResult | null>(null);
  const [cropPreview, setCropPreview] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFrame = useCallback((dataUrl: string) => {
    if (!dataUrl) {
      setStage("upload");
      setFrameDataUrl("");
      setGarment(null);
      setCropPreview("");
      return;
    }
    setFrameDataUrl(dataUrl);
    setStage("select");
    setGarment(null);
    setCropPreview("");
    setError(null);
  }, []);

  const handleCrop = useCallback(async (blob: Blob, previewUrl: string) => {
    setCropPreview(previewUrl);
    setIdentifying(true);
    setError(null);
    try {
      const result = await identify(blob);
      setGarment(result);
      setStage("results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Identification failed");
    } finally {
      setIdentifying(false);
    }
  }, []);

  const handleClosePanel = () => {
    setGarment(null);
    setStage("select");
  };

  const handleReset = () => {
    setStage("upload");
    setFrameDataUrl("");
    setGarment(null);
    setCropPreview("");
    setError(null);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Top bar */}
      <header
        style={{
          padding: "20px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: "10px" }}>
          <h1
            style={{
              fontSize: "1.1rem",
              fontWeight: 400,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: "var(--text)",
              margin: 0,
            }}
          >
            CineStyle
          </h1>
          <span
            style={{
              fontSize: "0.6rem",
              color: "var(--amber-dim)",
              letterSpacing: "0.2em",
              textTransform: "uppercase",
            }}
          >
            Film Fashion Intelligence
          </span>
        </div>

        {stage !== "upload" && (
          <button
            onClick={handleReset}
            style={{
              background: "none",
              border: "1px solid var(--border)",
              color: "var(--text-muted)",
              padding: "6px 16px",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "0.75rem",
              letterSpacing: "0.08em",
              transition: "border-color 0.2s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--text-dim)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--border)";
            }}
          >
            New Scene
          </button>
        )}
      </header>

      {/* Main content */}
      <main
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 32px",
          transition: "padding-right 0.35s ease",
          paddingRight: stage === "results" ? "min(512px, calc(100vw / 2))" : "32px",
        }}
      >
        <div style={{ width: "100%", maxWidth: "720px" }}>
          {/* Upload stage */}
          {stage === "upload" && (
            <div className="fade-up">
              <div style={{ textAlign: "center", marginBottom: "40px" }}>
                <h2
                  style={{
                    fontSize: "2rem",
                    fontWeight: 400,
                    color: "var(--text)",
                    marginBottom: "12px",
                    lineHeight: 1.3,
                  }}
                >
                  Identify any garment
                  <br />
                  <span style={{ color: "var(--amber)" }}>from the screen</span>
                </h2>
                <p
                  style={{
                    color: "var(--text-muted)",
                    fontSize: "0.9rem",
                    maxWidth: "420px",
                    margin: "0 auto",
                    lineHeight: 1.6,
                  }}
                >
                  Upload a film still or screenshot, select a garment, and
                  discover where to buy it.
                </p>
              </div>
              <FrameCapture onFrame={handleFrame} />
            </div>
          )}

          {/* Select / Results stage — show the image with overlay */}
          {(stage === "select" || stage === "results") && frameDataUrl && (
            <div className="fade-up">
              {/* Step indicator */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  marginBottom: "16px",
                }}
              >
                <span
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    background:
                      stage === "results" ? "var(--amber)" : "var(--text-dim)",
                  }}
                />
                <span
                  style={{
                    fontSize: "0.7rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.15em",
                    color:
                      stage === "results"
                        ? "var(--amber)"
                        : "var(--text-muted)",
                  }}
                >
                  {stage === "results"
                    ? "Garment identified — see results"
                    : "Draw a box around a garment"}
                </span>
              </div>

              <GarmentHighlight
                src={frameDataUrl}
                onCrop={handleCrop}
                loading={identifying}
              />

              {/* Error message */}
              {error && (
                <div
                  style={{
                    marginTop: "16px",
                    padding: "12px 16px",
                    background: "rgba(220,38,38,0.08)",
                    border: "1px solid rgba(220,38,38,0.2)",
                    borderRadius: "8px",
                    color: "#fca5a5",
                    fontSize: "0.8rem",
                  }}
                >
                  {error}
                </div>
              )}

              {/* Change image button */}
              <div style={{ marginTop: "16px", textAlign: "center" }}>
                <button
                  onClick={handleReset}
                  style={{
                    background: "none",
                    border: "1px solid var(--border)",
                    color: "var(--text-muted)",
                    padding: "6px 18px",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontSize: "0.8rem",
                    letterSpacing: "0.05em",
                    transition: "border-color 0.2s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--text-dim)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border)";
                  }}
                >
                  &#8592; Change image
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Result panel */}
      {stage === "results" && garment && (
        <ResultPanel
          garment={garment}
          cropPreview={cropPreview}
          onClose={handleClosePanel}
        />
      )}

      {/* Overlay backdrop for mobile */}
      {stage === "results" && garment && (
        <div
          onClick={handleClosePanel}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.4)",
            zIndex: 40,
            display: "none",
          }}
          className="panel-backdrop"
        />
      )}
    </div>
  );
}
