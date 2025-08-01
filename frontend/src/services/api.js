const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }
  return await response.json();
}

export async function buildIndex() {
  const response = await fetch(`${API_BASE_URL}/index`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Indexing failed: ${response.statusText}`);
  }
  return await response.json();
}

export async function searchDocuments(query) {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }
  return await response.json();
}
