import React from "react";
import { theme } from "../theme";

const styles = {
  shell: {
    ...theme.glass.darkPanel,
    display: "grid",
    gap: "0.9rem",
    padding: "1.2rem",
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
  row: {
    display: "grid",
    gap: "0.9rem",
    gridTemplateColumns: "minmax(0, 1fr) auto",
    alignItems: "end",
  },
  helper: {
    margin: 0,
    color: theme.colors.textOnDarkMuted,
    fontSize: "0.92rem",
    lineHeight: 1.5,
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
};

const SearchBar = ({
  query,
  product,
  onQueryChange,
  onProductChange,
  onSearch,
  loading = false,
}) => {
  const handleQueryKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSearch();
    }
  };

  const handleProductKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      onSearch();
    }
  };

  return (
    <div style={styles.shell}>
      <div>
        <div style={styles.label}>Question</div>
        <textarea
          style={{ ...styles.field, ...styles.textarea }}
          placeholder="Ask about a product passport, carbon footprint, recyclability, or an exact-only footprint check..."
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={handleQueryKeyDown}
        />
      </div>

      <div style={styles.row}>
        <div>
          <div style={styles.label}>Optional Product</div>
          <input
            style={styles.field}
            type="text"
            placeholder="Example: lexmark_mx431adn"
            value={product}
            onChange={(event) => onProductChange(event.target.value)}
            onKeyDown={handleProductKeyDown}
          />
          <p style={styles.helper}>
            Press Enter to run the query. Use phrases like <strong>exact only</strong> or <strong>no estimate</strong> when you want strict official data only.
          </p>
        </div>

        <button type="button" style={styles.button} onClick={onSearch} disabled={loading}>
          {loading ? "Working..." : "Ask System"}
        </button>
      </div>
    </div>
  );
};

export default SearchBar;
