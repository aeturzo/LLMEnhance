import React, { useEffect, useMemo, useState } from "react";
import ResultsList from "./ResultsList";
import AdvancedAuditView from "./AdvancedAuditView";
import * as api from "../services/api";
import { theme } from "../theme";

const DOMAIN_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "battery", label: "Battery" },
  { value: "lexmark", label: "Lexmark" },
  { value: "viessmann", label: "Viessmann" },
];

const DOMAIN_DEFAULTS = {
  auto: {
    memoryNote: "For ProductA, the preferred packaging is recycled corrugated cardboard and the preferred supplier is GreenCells.",
    examples: [
      {
        label: "Memory preference",
        query: "What packaging did I say ProductA prefers?",
        product: "",
      },
      {
        label: "Symbolic compliance",
        query: "Name two compliance standards that apply to ProductA.",
        product: "",
      },
      {
        label: "Blended memory + symbolic",
        query: "What packaging did I say ProductA prefers, and name one compliance standard for ProductA.",
        product: "",
      },
    ],
  },
  battery: {
    memoryNote: "For ProductA, the preferred packaging is recycled corrugated cardboard and the preferred supplier is GreenCells.",
    examples: [
      {
        label: "Battery memory",
        query: "What did I record about ProductA packaging and supplier?",
        product: "",
      },
      {
        label: "Battery retrieval",
        query: "What is the canonical name for EPR?",
        product: "",
      },
      {
        label: "Battery symbolic",
        query: "Name two compliance standards that apply to ProductA.",
        product: "",
      },
    ],
  },
  lexmark: {
    memoryNote: "For PrinterL1, the preferred service approach is refurbish the printer before replacement and keep toner collection separate.",
    examples: [
      {
        label: "Lexmark memory",
        query: "What did I record about PrinterL1 service and toner handling?",
        product: "",
      },
      {
        label: "Lexmark retrieval",
        query: "What is the brand of Lexmark MS521dn?",
        product: "",
      },
      {
        label: "Lexmark symbolic",
        query: "Which compliance requirements apply to PrinterL1?",
        product: "",
      },
    ],
  },
  viessmann: {
    memoryNote: "For ProductV1, the preferred installation note is to keep a documented leak-check record and review the wireless module at commissioning.",
    examples: [
      {
        label: "Viessmann memory",
        query: "What did I record about ProductV1 installation and commissioning?",
        product: "",
      },
      {
        label: "Viessmann retrieval",
        query: "What is the website of Viessmann Climate Solutions SE?",
        product: "",
      },
      {
        label: "Viessmann symbolic",
        query: "Which compliance requirements apply to ProductV1?",
        product: "",
      },
    ],
  },
};

const styles = {
  shell: {
    display: "grid",
    gap: "1.35rem",
  },
  hero: {
    ...theme.glass.panelStrong,
    display: "grid",
    gap: "1rem",
    padding: "1.35rem 1.4rem",
    borderRadius: "28px",
  },
  eyebrow: {
    margin: 0,
    fontSize: "0.84rem",
    fontWeight: 800,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "rgba(255, 245, 251, 0.76)",
  },
  title: {
    margin: 0,
    fontSize: "clamp(2rem, 4vw, 3.4rem)",
    lineHeight: 0.98,
    fontWeight: 900,
    letterSpacing: "-0.03em",
    color: theme.colors.textOnDark,
    fontFamily: theme.fontFamily.display,
    textShadow: "0 12px 32px rgba(45, 11, 71, 0.22)",
  },
  subtitle: {
    margin: 0,
    maxWidth: "840px",
    fontSize: "1.02rem",
    lineHeight: 1.65,
    color: "rgba(255, 243, 251, 0.88)",
  },
  grid: {
    display: "grid",
    gap: "0.95rem",
  },
  card: {
    ...theme.glass.darkPanel,
    display: "grid",
    gap: "0.8rem",
    padding: "1.15rem",
    borderRadius: "24px",
  },
  label: {
    marginBottom: "0.35rem",
    fontSize: "0.82rem",
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "rgba(255, 232, 248, 0.82)",
  },
  field: {
    ...theme.glass.fieldDark,
    width: "100%",
    padding: "0.9rem 1rem",
    borderRadius: "16px",
    color: theme.colors.textOnDark,
    fontSize: "1rem",
    fontFamily: theme.fontFamily.body,
    outline: "none",
    boxSizing: "border-box",
  },
  textarea: {
    minHeight: "116px",
    resize: "vertical",
    lineHeight: 1.5,
  },
  compactTextarea: {
    minHeight: "92px",
  },
  row: {
    display: "grid",
    gap: "0.9rem",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
  },
  buttonRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.75rem",
    alignItems: "center",
  },
  button: {
    minWidth: "152px",
    height: "50px",
    padding: "0 1.2rem",
    borderRadius: "999px",
    border: "1px solid rgba(255, 255, 255, 0.3)",
    background: theme.gradients.button,
    color: "#361a4d",
    fontWeight: 800,
    fontSize: "0.98rem",
    cursor: "pointer",
    boxShadow: theme.shadows.button,
  },
  secondaryButton: {
    ...theme.glass.fieldDark,
    minWidth: "152px",
    height: "50px",
    padding: "0 1.2rem",
    borderRadius: "999px",
    color: theme.colors.textOnDark,
    fontWeight: 800,
    fontSize: "0.96rem",
    cursor: "pointer",
  },
  helper: {
    margin: 0,
    color: theme.colors.textOnDarkMuted,
    fontSize: "0.92rem",
    lineHeight: 1.55,
  },
  status: {
    margin: 0,
    color: "rgba(255, 236, 247, 0.94)",
    fontSize: "0.92rem",
    lineHeight: 1.55,
  },
  exampleRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.75rem",
  },
  exampleButton: {
    ...theme.glass.card,
    padding: "0.75rem 0.95rem",
    borderRadius: "18px",
    color: theme.colors.textPrimary,
    fontSize: "0.92rem",
    fontWeight: 700,
    cursor: "pointer",
  },
};

const OrchestratedWorkspace = () => {
  const [query, setQuery] = useState("");
  const [product, setProduct] = useState("");
  const [domain, setDomain] = useState("auto");
  const [session, setSession] = useState("frontend-session");
  const [memoryNote, setMemoryNote] = useState(DOMAIN_DEFAULTS.auto.memoryNote);
  const [memoryStatus, setMemoryStatus] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [showAdvancedAudit, setShowAdvancedAudit] = useState(false);

  const sessionLabel = useMemo(() => session.trim() || "frontend-session", [session]);
  const examples = useMemo(() => DOMAIN_DEFAULTS[domain]?.examples || DOMAIN_DEFAULTS.auto.examples, [domain]);

  useEffect(() => {
    setMemoryNote(DOMAIN_DEFAULTS[domain]?.memoryNote || DOMAIN_DEFAULTS.auto.memoryNote);
    setMemoryStatus("");
  }, [domain]);

  const runQuery = async (nextQuery = query, nextProduct = product) => {
    const trimmedQuery = nextQuery.trim();
    if (!trimmedQuery || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setHasSearched(true);
    setShowAdvancedAudit(false);

    try {
      const data = await api.solveAutoQuery(trimmedQuery, {
        product: nextProduct.trim() || undefined,
        domain,
        session: sessionLabel,
      });
      setQuery(trimmedQuery);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "The orchestrated query failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuery(example.query);
    setProduct(example.product);
    runQuery(example.query, example.product);
  };

  const handleSaveMemory = async () => {
    const note = memoryNote.trim();
    if (!note || memoryLoading) {
      return;
    }
    setMemoryLoading(true);
    setMemoryStatus("");
    try {
      await api.putMemory(sessionLabel, note);
      setMemoryStatus(`Saved to session ${sessionLabel}. Ask the memory example now.`);
    } catch (err) {
      setMemoryStatus(err instanceof Error ? err.message : "Saving memory failed.");
    } finally {
      setMemoryLoading(false);
    }
  };

  return (
    <div style={styles.shell}>
      <div style={styles.hero}>
        <p style={styles.eyebrow}>Orchestrated Workspace</p>
        <h1 style={styles.title}>Automatic Memory, Search, and Symbolic Composition</h1>
        <p style={styles.subtitle}>
          This workspace restores the behavior you described: it checks session memory, retrieval, and ontology-backed symbolic reasoning, filters noisy evidence, and composes one answer for the LLM. The carbon solve workspace remains separate.
        </p>

        <div style={styles.card}>
          <div style={styles.row}>
            <div>
              <div style={styles.label}>Question</div>
              <textarea
                style={{ ...styles.field, ...styles.textarea }}
                placeholder="Ask a memory, compliance, or blended product-passport question..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
          </div>

          <div style={styles.row}>
            <div>
              <div style={styles.label}>Domain</div>
              <select
                style={styles.field}
                value={domain}
                onChange={(event) => setDomain(event.target.value)}
              >
                {DOMAIN_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div style={styles.label}>Optional Product</div>
              <input
                style={styles.field}
                type="text"
                placeholder="Optional product, or let the system infer it"
                value={product}
                onChange={(event) => setProduct(event.target.value)}
              />
            </div>
            <div>
              <div style={styles.label}>Session</div>
              <input
                style={styles.field}
                type="text"
                placeholder="frontend-session"
                value={session}
                onChange={(event) => setSession(event.target.value)}
              />
            </div>
          </div>

          <p style={styles.helper}>
            Memory retrieval is session-scoped. If you seed memory here, keep the same session value when you ask the follow-up question. The selected domain steers symbolic reasoning, and product names like ProductA, PrinterL1, and ProductV1 can usually be inferred from the query text.
          </p>

          <div style={styles.buttonRow}>
            <button type="button" style={styles.button} onClick={() => runQuery()} disabled={loading}>
              {loading ? "Working..." : "Ask Orchestrator"}
            </button>
          </div>
        </div>

        <div style={styles.card}>
          <div>
            <div style={styles.label}>Seed Memory For This Session</div>
            <textarea
              style={{ ...styles.field, ...styles.textarea, ...styles.compactTextarea }}
              placeholder="Store a note in the current session so memory-sensitive questions can be demonstrated from the UI."
              value={memoryNote}
              onChange={(event) => setMemoryNote(event.target.value)}
            />
          </div>
          <div style={styles.buttonRow}>
            <button type="button" style={styles.secondaryButton} onClick={handleSaveMemory} disabled={memoryLoading}>
              {memoryLoading ? "Saving..." : "Save Memory"}
            </button>
            {memoryStatus ? <p style={styles.status}>{memoryStatus}</p> : null}
          </div>
        </div>

        <div style={styles.exampleRow}>
          {examples.map((example) => (
            <button
              key={example.label}
              type="button"
              style={styles.exampleButton}
              onClick={() => handleExampleClick(example)}
            >
              {example.label}
            </button>
          ))}
        </div>
      </div>

      {showAdvancedAudit ? (
        <AdvancedAuditView
          result={result}
          onBack={() => setShowAdvancedAudit(false)}
        />
      ) : (
        <ResultsList
          result={result}
          loading={loading}
          error={error}
          hasSearched={hasSearched}
          onOpenAdvancedAudit={() => setShowAdvancedAudit(true)}
          readyTitle="Ask through the orchestration pipeline"
          readyDescription="This workspace checks session memory, retrieval, and symbolic reasoning, filters weak evidence, and then composes one answer with a shared audit trail."
        />
      )}
    </div>
  );
};

export default OrchestratedWorkspace;
