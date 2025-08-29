// small helper to call backend endpoints
const base = window.__OMR_BASE_URL__ || ""; // if you host on same origin, leave empty

async function postForm(path, form) {
  const res = await fetch(base + path, { method: "POST", body: form });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function apiScanSingle(file) {
  const f = new FormData();
  f.append("file", file);
  const json = await postForm("/api/scan_single", f);
  return json;
}

export async function apiScanMultiple(files) {
  const f = new FormData();
  files.forEach((file, i) => f.append("files", file));
  const json = await postForm("/api/scan_multiple", f);
  return json; // expect array of results
}

export async function apiScanCamera(url) {
  const f = new FormData();
  f.append("camera_url", url);
  const json = await postForm("/api/scan_camera", f);
  return json;
}
