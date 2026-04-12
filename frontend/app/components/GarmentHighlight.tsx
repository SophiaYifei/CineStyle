"use client";

import { useRef, useState, useCallback, useEffect } from "react";

interface Point { x: number; y: number; }

interface GarmentHighlightProps {
  /** The frame data-URL to display */
  src: string;
  /** Called when the user finishes selecting a region. Blob is the cropped JPEG. */
  onCrop: (blob: Blob, previewUrl: string) => void;
  /** While true, show a loading indicator instead of enabling selection */
  loading?: boolean;
}

/**
 * GarmentHighlight — overlays the frame image with a click-and-drag
 * selection rectangle. On mouse-up it crops the selected region and
 * passes the Blob + preview URL to the parent.
 */
export default function GarmentHighlight({ src, onCrop, loading = false }: GarmentHighlightProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const [dragging, setDragging] = useState(false);
  const [start, setStart] = useState<Point | null>(null);
  const [end, setEnd] = useState<Point | null>(null);
  const [selectionBox, setSelectionBox] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

  // Draw amber dashed rectangle on overlay canvas while dragging
  useEffect(() => {
    const canvas = overlayRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (selectionBox) {
      const { left, top, width, height } = selectionBox;
      ctx.strokeStyle = "#d97706";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(left, top, width, height);
      ctx.fillStyle = "rgba(217,119,6,0.08)";
      ctx.fillRect(left, top, width, height);
    }
  }, [selectionBox]);

  const getRelativePos = useCallback((e: React.MouseEvent): Point => {
    const rect = imgRef.current!.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (loading) return;
    e.preventDefault();
    const pos = getRelativePos(e);
    setStart(pos);
    setEnd(pos);
    setDragging(true);
    setSelectionBox({ left: pos.x, top: pos.y, width: 0, height: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragging || !start) return;
    const pos = getRelativePos(e);
    setEnd(pos);
    setSelectionBox({
      left: Math.min(start.x, pos.x),
      top: Math.min(start.y, pos.y),
      width: Math.abs(pos.x - start.x),
      height: Math.abs(pos.y - start.y),
    });
  };

  const handleMouseUp = useCallback(async () => {
    if (!dragging || !start || !end || !selectionBox) return;
    setDragging(false);

    const { width, height } = selectionBox;
    if (width < 10 || height < 10) {
      setSelectionBox(null);
      return;
    }

    const img = imgRef.current!;
    const rect = img.getBoundingClientRect();

    // Scale from CSS pixels to natural image pixels
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;

    const cropX = selectionBox.left * scaleX;
    const cropY = selectionBox.top * scaleY;
    const cropW = width * scaleX;
    const cropH = height * scaleY;

    const canvas = document.createElement("canvas");
    canvas.width = cropW;
    canvas.height = cropH;
    const ctx = canvas.getContext("2d")!;

    const naturalImg = new Image();
    naturalImg.src = src;
    naturalImg.onload = () => {
      ctx.drawImage(naturalImg, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
      const previewUrl = canvas.toDataURL("image/jpeg", 0.92);
      canvas.toBlob(
        (blob) => { if (blob) onCrop(blob, previewUrl); },
        "image/jpeg",
        0.92
      );
    };
  }, [dragging, start, end, selectionBox, src, onCrop]);

  const clearSelection = () => setSelectionBox(null);

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", userSelect: "none", borderRadius: "8px", overflow: "hidden" }}
    >
      {/* The frame image */}
      <img
        ref={imgRef}
        src={src}
        alt="Frame"
        draggable={false}
        style={{ width: "100%", display: "block", borderRadius: "8px" }}
      />

      {/* Overlay canvas for selection rectangle */}
      <canvas
        ref={overlayRef}
        className={loading ? "" : "crosshair"}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { if (dragging) handleMouseUp(); }}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: loading ? "none" : "auto",
        }}
      />

      {/* Loading indicator */}
      {loading && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(12,10,9,0.6)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "12px",
          }}
        >
          <div style={{ display: "flex", gap: "6px" }}>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="pulse-amber"
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: "var(--amber)",
                  animationDelay: `${i * 0.2}s`,
                }}
              />
            ))}
          </div>
          <span
            style={{
              color: "var(--amber)",
              fontSize: "0.75rem",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
            }}
          >
            Identifying garment
          </span>
        </div>
      )}

      {/* Hint label when not loading and no selection */}
      {!loading && !selectionBox && (
        <div
          style={{
            position: "absolute",
            bottom: "12px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(12,10,9,0.75)",
            border: "1px solid var(--border)",
            color: "var(--text-muted)",
            fontSize: "0.75rem",
            padding: "4px 12px",
            borderRadius: "20px",
            letterSpacing: "0.08em",
            pointerEvents: "none",
            whiteSpace: "nowrap",
          }}
        >
          Drag to select a garment
        </div>
      )}

      {/* Clear selection button */}
      {selectionBox && !loading && (
        <button
          onClick={clearSelection}
          style={{
            position: "absolute",
            top: "8px",
            right: "8px",
            background: "rgba(12,10,9,0.8)",
            border: "1px solid var(--border)",
            color: "var(--text-muted)",
            padding: "3px 8px",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "0.75rem",
          }}
        >
          Clear
        </button>
      )}
    </div>
  );
}
