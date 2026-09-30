window.CYCLONE_CONFIG = {
  API_BASE: (typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"))
    ? window.location.origin
    : "https://cyclone-ai-47kh.onrender.com"
};
