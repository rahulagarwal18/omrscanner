import React from "react";

function downloadJSON(obj, filename = "results.json") {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Results({ data = [], clearAll }) {
  if (!data?.length) {
    return <div id="results-panel" className="text-white/80">No results yet.</div>;
  }

  const aggregate = () => {
    const arr = data;
    return {
      totalSheets: arr.length,
      totalBubbles: arr.reduce((s, r) => s + (r.bubbles_detected || 0), 0),
    };
  };

  return (
    <div id="results-panel" className="space-y-4 text-white/90">
      <div className="flex items-center justify-between gap-4">
        <div>
          <strong>{data.length}</strong> sheet(s) scanned • Bubbles: <strong>{aggregate().totalBubbles}</strong>
        </div>
        <div className="flex gap-2">
          <button
            className="btn-outline"
            onClick={() => downloadJSON(data, "omr_results.json")}
          >
            Download JSON
          </button>
          <button
            className="btn-outline"
            onClick={() => {
              // CSV simple export
              const header = ["filename", "score", "total", "percentage", "bubbles_detected", "rows_detected"];
              const rows = data.map((r) => [
                r.filename || "",
                r.score ?? "",
                r.total ?? "",
                r.percentage ?? "",
                r.bubbles_detected ?? "",
                r.rows_detected ?? "",
              ]);
              const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
              const blob = new Blob([csv], { type: "text/csv" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = "omr_results.csv";
              a.click();
              URL.revokeObjectURL(url);
            }}
          >
            Export CSV
          </button>
          <button className="btn-outline" onClick={clearAll}>Clear</button>
        </div>
      </div>

      <div className="grid gap-3">
        {data.map((r, idx) => (
          <div key={idx} className="bg-white/6 p-4 rounded-xl">
            <div className="flex justify-between items-start">
              <div>
                <div className="font-semibold">{r.filename || `sheet-${idx+1}`}</div>
                <div className="text-sm text-white/70">Score: {r.score}/{r.total} ({(r.percentage ?? 0).toFixed(1)}%)</div>
              </div>
              <div className="text-xs text-white/60">Source: {r._source || "unknown"}</div>
            </div>

            <details className="mt-3 text-sm text-white/80">
              <summary className="cursor-pointer">Show details</summary>
              <ul className="mt-2 list-disc ml-5">
                {r.details?.map((d, i) => (
                  <li key={i}>
                    Q{d.question}: Detected {d.detected} — Correct {d.correct} — {d.status}
                  </li>
                )) || <li>No details</li>}
              </ul>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
}
