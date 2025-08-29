import React from "react";

export function Button({ children, className = "", ...props }) {
  return (
    <button
      className={`px-6 py-3 rounded-lg bg-gradient-to-r from-purple-500 to-blue-500 text-white font-bold shadow-lg hover:shadow-xl transition ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
