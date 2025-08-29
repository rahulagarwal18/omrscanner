import React from "react";

export default function Header() {
  return (
    <header className="sticky top-0 z-30">
      <nav className="mx-auto max-w-6xl px-4 pt-6">
        <div className="glass rounded-2xl p-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-white/20 grid place-items-center text-white text-xl">🎯</div>
            <div>
              <h1 className="text-2xl font-extrabold text-white tracking-wide">OMR Web Scanner</h1>
              <p className="text-white/70 text-sm">Scan single, multiple & IP camera • Auto-key detection • Save results</p>
            </div>
          </div>
          <a
            className="btn-outline"
            href="#results"
            onClick={(e) => {
              e.preventDefault();
              document.getElementById("results-panel")?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            View Results
          </a>
        </div>
      </nav>
    </header>
  );
}
