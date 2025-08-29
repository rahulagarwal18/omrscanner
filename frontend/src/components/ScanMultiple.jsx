import React, { useState } from "react";
import Loader from "./loader.jsx";
import { apiScanMultiple } from "../utils/api.js";

export default function ScanMultiple({ onScanned, onError }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!files?.length) return;
    try {
      setLoading(true);
      const res = await apiScanMultiple(files);
      // Expecting array
      onScanned(res.map((r, i) => ({ ...r, _source: "multi", filename: r.filename || files[i]?.name })));
    } catch (err) {
      onError(err.message || "Multi scan failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-4">
      <input
        type="file"
        accept="image/*"
        multiple
        onChange={(e) => setFiles(Array.from(e.target.files || []))}
        className="block w-full text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-white/20 file:px-4 file:py-2 file:text-white hover:file:bg-white/30 text-white/90"
      />
      <div className="flex items-center gap-3">
        <button className="btn-primary" type="submit" disabled={!files?.length || loading}>
          {loading ? "Scanning…" : `Scan ${files?.length || 0} Files`}
        </button>
        {loading && <Loader label="Scanning multiple sheets…" />}
      </div>
    </form>
  );
}
