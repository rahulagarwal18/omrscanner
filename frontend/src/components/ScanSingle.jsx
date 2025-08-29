import React, { useState } from "react";
import loader from "./loader.jsx";   // ✅ match the actual filename

import { apiScanSingle } from "../utils/api.js";

export default function ScanSingle({ onScanned, onError }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!file) return;
    try {
      setLoading(true);
      const res = await apiScanSingle(file);
      onScanned({ ...res, _source: "single", filename: res.filename || file.name });
    } catch (err) {
      onError(err.message || "Scan failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-4">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="block w-full text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-white/20 file:px-4 file:py-2 file:text-white hover:file:bg-white/30 text-white/90"
      />
      <div className="flex items-center gap-3">
        <button className="btn-primary" type="submit" disabled={!file || loading}>
          {loading ? "Scanning…" : "Scan"}
        </button>
        {loading && <Loader label="Scanning sheet…" />}
      </div>
      <p className="text-white/70 text-xs">
        Tip: Backend auto-loads <code>ans.json</code> if present. No need to upload answer key.
      </p>
    </form>
  );
}
