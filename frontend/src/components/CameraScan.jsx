import React, { useState } from "react";
import Loader from "./loader.jsx";
import { apiScanCamera } from "../utils/api.js";

export default function CameraScan({ onScanned, onError }) {
  const [url, setUrl] = useState("http://10.150.114.196:8080/shot.jpg");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    try {
      setLoading(true);
      const res = await apiScanCamera(url.trim());
      onScanned({ ...res, _source: "camera", filename: res.filename || "camera.jpg" });
    } catch (err) {
      onError(err.message || "Camera scan failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-4">
      <input
        type="text"
        placeholder="http://ip:port/shot.jpg"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        className="w-full rounded-xl bg-white/10 text-white/90 px-4 py-2 placeholder:text-white/60"
      />
      <div className="flex items-center gap-3">
        <button className="btn-primary" type="submit" disabled={!url || loading}>
          {loading ? "Capturing…" : "Capture & Scan"}
        </button>
        {loading && <Loader label="Capturing from camera…" />}
      </div>
      <p className="text-white/70 text-xs">
        Enter an IP camera snapshot URL (e.g. from IP Webcam on Android). Backend will fetch and scan it.
      </p>
    </form>
  );
}
