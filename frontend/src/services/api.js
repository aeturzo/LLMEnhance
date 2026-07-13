const API_BASE_URL =
  (typeof process !== "undefined" &&
  process &&
  process.env &&
  process.env.REACT_APP_API_URL)
    ? process.env.REACT_APP_API_URL
    : "http://localhost:8000/api";

function normalizeBaseUrl(url) {
  return (url || "").replace(/\/+$/, "");
}

function candidateBaseUrls() {
  const base = normalizeBaseUrl(API_BASE_URL);
  if (!base) {
    return [""];
  }
  if (base.endsWith("/api")) {
    return [base, base.slice(0, -4)];
  }
  return [base, `${base}/api`];
}

async function parseError(response, fallbackMessage) {
  try {
    const data = await response.json();
    if (data && typeof data.detail === "string" && data.detail.trim()) {
      return new Error(data.detail);
    }
  } catch (err) {
    // Ignore JSON parsing failures and fall back to text/status.
  }

  try {
    const text = await response.text();
    if (text && text.trim()) {
      return new Error(text.trim());
    }
  } catch (err) {
    // Ignore text parsing failures and use the fallback.
  }

  return new Error(fallbackMessage);
}

async function requestJson(path, options = {}, { retryOnNotFound = true } = {}) {
  const bases = candidateBaseUrls();
  let lastError = null;

  for (let index = 0; index < bases.length; index += 1) {
    const baseUrl = bases[index];
    const url = `${baseUrl}${path}`;
    try {
      const response = await fetch(url, options);
      if (response.ok) {
        return await response.json();
      }
      lastError = await parseError(
        response,
        `${options.method || "GET"} ${url} failed: ${response.status} ${response.statusText}`
      );
      if (!(retryOnNotFound && response.status === 404 && index < bases.length - 1)) {
        throw lastError;
      }
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (index === bases.length - 1) {
        throw lastError;
      }
    }
  }

  throw lastError || new Error(`Request failed for ${path}`);
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson("/upload", {
    method: "POST",
    body: formData,
  });
}

export async function buildIndex() {
  return requestJson("/index", { method: "POST" });
}

export async function searchDocuments(query) {
  return requestJson("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
}

export async function solveQuery(query, options = {}) {
  const payload = {
    query,
    product: options.product || null,
    session: options.session || "frontend-session",
  };

  if (options.mode) {
    payload.mode = options.mode;
  }

  return requestJson("/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function solveAutoQuery(query, options = {}) {
  const payload = {
    query,
    product: options.product || null,
    domain: options.domain || "auto",
    session: options.session || "frontend-session",
  };

  if (options.topKSearch) {
    payload.top_k_search = options.topKSearch;
  }

  if (options.topKMemory) {
    payload.top_k_memory = options.topKMemory;
  }

  return requestJson("/solve_auto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function putMemory(sessionId, content, metadata = null) {
  return requestJson("/memory/put", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      content,
      metadata,
    }),
  });
}
