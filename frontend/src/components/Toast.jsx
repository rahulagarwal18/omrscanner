import React, { useEffect } from "react";

export default function Toast({ message, onClose, ttl = 3500 }) {
  useEffect(() => {
    const id = setTimeout(() => onClose?.(), ttl);
    return () => clearTimeout(id);
  }, [message]);

  if (!message) return null;
  return (
    <div className="fixed right-6 bottom-6 z-50">
      <div className="glass p-3 rounded-lg text-white shadow">
        {message}
      </div>
    </div>
  );
}
