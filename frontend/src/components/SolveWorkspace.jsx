import React, { useState } from "react";
import SearchBar from "./SearchBar";
import ResultsList from "./ResultsList";
import AdvancedAuditView from "./AdvancedAuditView";
import * as api from "../services/api";
import { theme } from "../theme";

const EXAMPLE_QUERIES = [
  "What is the carbon footprint of Lexmark MX431adn?",
  "Give me a stage breakdown for Lexmark MX431adn carbon footprint.",
  "What is the exact carbon footprint of Lexmark MX431adn with no estimate?",
];

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
    fontSize: "clamp(2rem, 4vw, 3.5rem)",
    lineHeight: 0.98,
    fontWeight: 900,
    letterSpacing: "-0.03em",
    color: theme.colors.textOnDark,
    fontFamily: theme.fontFamily.display,
    textShadow: "0 12px 32px rgba(45, 11, 71, 0.22)",
  },
  subtitle: {
    margin: 0,
    maxWidth: "780px",
    fontSize: "1.02rem",
    lineHeight: 1.65,
    color: "rgba(255, 243, 251, 0.88)",
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

const SolveWorkspace = () => {
  const [query, setQuery] = useState("");
  const [product, setProduct] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [showAdvancedAudit, setShowAdvancedAudit] = useState(false);

  const runQuery = async (nextQuery = query) => {
    const trimmedQuery = nextQuery.trim();
    if (!trimmedQuery || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setHasSearched(true);
    setShowAdvancedAudit(false);

    try {
      const data = await api.solveQuery(trimmedQuery, {
        product: product.trim() || undefined,
        session: "frontend-session",
      });
      setQuery(trimmedQuery);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "The query failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuery(example);
    runQuery(example);
  };

  return (
    <div style={styles.shell}>
      <div style={styles.hero}>
        <p style={styles.eyebrow}>Solve Workspace</p>
        <h1 style={styles.title}>Digital Product Passport and Carbon Intelligence</h1>
        <p style={styles.subtitle}>
          The current solve flow stays intact here. It handles general passport questions, carbon-footprint estimates, stage breakdowns, recyclability answers, and strict exact-only checks with honest disclosure.
        </p>

        <SearchBar
          query={query}
          product={product}
          onQueryChange={setQuery}
          onProductChange={setProduct}
          onSearch={() => runQuery()}
          loading={loading}
        />

        <div style={styles.exampleRow}>
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              type="button"
              style={styles.exampleButton}
              onClick={() => handleExampleClick(example)}
            >
              {example}
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
        />
      )}
    </div>
  );
};

export default SolveWorkspace;
