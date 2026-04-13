"use client";

import { useRef, useState, useCallback } from "react";

interface FrameCaptureProps {
  onFrame: (dataUrl: string) => void;
}

/**
 * FrameCapture — accepts a still image upload or a video file.
 * For video: renders a seek slider so the user can pick the exact frame,
 * then hands a data-URL of that frame up to the parent.
 */
export default function FrameCapture({ onFrame }: FrameCaptureProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [mode, setMode] = useState<"idle" | "image" | "video">("idle");
  const [videoSrc, setVideoSrc] = useState<string>("");
  const [seekVal, setSeekVal] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")!.drawImage(video, 0, 0);
    onFrame(canvas.toDataURL("image/jpeg", 0.92));
  }, [onFrame]);

  const handleFile = (file: File) => {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onFrame(e.target?.result as string);
        setMode("image");
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      setMode("video");
      setSeekVal(0);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleSeek = (val: number) => {
    setSeekVal(val);
    if (videoRef.current) {
      videoRef.current.currentTime = val;
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="w-full">
      {mode === "idle" && (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `2px dashed ${isDragging ? "var(--amber)" : "var(--border)"}`,
            background: isDragging ? "rgba(217,119,6,0.05)" : "var(--surface)",
            borderRadius: "12px",
            padding: "64px 40px",
            textAlign: "center",
            cursor: "pointer",
            transition: "border-color 0.2s, background 0.2s",
          }}
        >
          <div style={{ fontSize: "2.5rem", marginBottom: "16px", opacity: 0.5 }}>
            &#128247;
          </div>
          <p style={{ color: "var(--text)", fontSize: "1.05rem", marginBottom: "8px" }}>
            Drop a film still or screenshot here
          </p>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
            JPG, PNG, WEBP
          </p>
        </div>
      )}

      {mode === "image" && (
        <div style={{ textAlign: "center" }}>
          <button
            onClick={() => { setMode("idle"); onFrame(""); }}
            style={{
              marginBottom: "12px",
              background: "none",
              border: "1px solid var(--border)",
              color: "var(--text-muted)",
              padding: "4px 14px",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.8rem",
              letterSpacing: "0.05em",
            }}
          >
            &#8592; Change image
          </button>
        </div>
      )}

      {mode === "video" && (
        <div>
          {/* Hidden video + canvas for frame extraction */}
          <video
            ref={videoRef}
            src={videoSrc}
            style={{ display: "none" }}
            onLoadedMetadata={() => {
              setDuration(videoRef.current?.duration ?? 0);
            }}
            onSeeked={captureFrame}
          />
          <canvas ref={canvasRef} style={{ display: "none" }} />

          {/* Seek UI */}
          <div
            style={{
              background: "var(--surface)",
              borderRadius: "10px",
              padding: "20px",
              border: "1px solid var(--border)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
              <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                {formatTime(seekVal)}
              </span>
              <span style={{ color: "var(--amber)", fontSize: "0.75rem", letterSpacing: "0.1em" }}>
                DRAG TO SELECT FRAME
              </span>
              <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                {formatTime(duration)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={duration}
              step={0.033}
              value={seekVal}
              onChange={(e) => handleSeek(Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--amber)" }}
            />
            <div style={{ display: "flex", gap: "10px", marginTop: "14px" }}>
              <button
                onClick={() => { setMode("idle"); setVideoSrc(""); }}
                style={{
                  flex: 1,
                  background: "none",
                  border: "1px solid var(--border)",
                  color: "var(--text-muted)",
                  padding: "8px",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontSize: "0.8rem",
                }}
              >
                Change file
              </button>
              <button
                onClick={captureFrame}
                style={{
                  flex: 2,
                  background: "var(--amber)",
                  border: "none",
                  color: "#0c0a09",
                  padding: "8px",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: "bold",
                  fontSize: "0.85rem",
                  letterSpacing: "0.05em",
                }}
              >
                Use this frame
              </button>
            </div>
          </div>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
