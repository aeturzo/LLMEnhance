import React from "react";
import { theme } from "../theme";

function formatNumber(value, decimals = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "Unavailable";
  }
  return Number(value).toFixed(decimals).replace(/\.?0+$/, "");
}

function titleize(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function jsonBlock(value) {
  return JSON.stringify(value, null, 2);
}

const styles = {
  shell: {
    display: "grid",
    gap: "1.1rem",
  },
  hero: {
    ...theme.glass.darkPanel,
    padding: "1.25rem 1.3rem",
    borderRadius: "24px",
    color: theme.colors.textOnDark,
  },
  eyebrow: {
    margin: 0,
    fontSize: "0.78rem",
    fontWeight: 800,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "rgba(255, 226, 245, 0.74)",
  },
  title: {
    margin: "0.4rem 0 0",
    fontSize: "1.6rem",
    lineHeight: 1.15,
    fontFamily: theme.fontFamily.display,
  },
  text: {
    margin: "0.8rem 0 0",
    fontSize: "0.98rem",
    lineHeight: 1.65,
    color: theme.colors.textOnDarkMuted,
  },
  buttonRow: {
    display: "flex",
    gap: "0.75rem",
    flexWrap: "wrap",
    marginTop: "1rem",
  },
  button: {
    height: "46px",
    padding: "0 1rem",
    borderRadius: "999px",
    border: "1px solid rgba(255, 255, 255, 0.28)",
    background: theme.gradients.button,
    color: "#361a4d",
    fontWeight: 800,
    cursor: "pointer",
    boxShadow: theme.shadows.button,
  },
  ghostButton: {
    ...theme.glass.fieldDark,
    height: "46px",
    padding: "0 1rem",
    borderRadius: "999px",
    color: theme.colors.textOnDark,
    fontWeight: 800,
    cursor: "pointer",
  },
  panel: {
    ...theme.glass.panel,
    padding: "1.15rem 1.2rem",
    borderRadius: "22px",
  },
  panelTitle: {
    margin: 0,
    fontSize: "0.84rem",
    fontWeight: 800,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "rgba(92, 66, 142, 0.82)",
  },
  panelText: {
    margin: "0.7rem 0 0",
    fontSize: "0.95rem",
    lineHeight: 1.6,
    color: theme.colors.textSecondary,
  },
  grid: {
    display: "grid",
    gap: "0.85rem",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    marginTop: "0.9rem",
  },
  card: {
    ...theme.glass.card,
    padding: "0.95rem 1rem",
    borderRadius: "18px",
  },
  code: {
    ...theme.glass.code,
    margin: "0.75rem 0 0",
    padding: "0.95rem",
    borderRadius: "16px",
    color: theme.colors.textOnDark,
    overflowX: "auto",
    fontSize: "0.84rem",
    lineHeight: 1.6,
    fontFamily: theme.fontFamily.mono,
  },
  metric: {
    margin: "0.42rem 0 0",
    fontSize: "0.92rem",
    lineHeight: 1.55,
    color: theme.colors.textSecondary,
  },
};

function collectTraceEntries(result) {
  const carbon = result?.carbon;
  const stageResults = carbon?.stage_results || {};
  const entries = [];

  Object.entries(stageResults).forEach(([stageName, stage]) => {
    const traces = Array.isArray(stage?.traces) ? stage.traces : [];
    traces.forEach((trace) => {
      entries.push({
        stageName,
        ...trace,
      });
    });
  });

  return entries;
}

const AdvancedAuditView = ({ result, onBack }) => {
  const steps = Array.isArray(result?.steps) ? result.steps : [];
  const traceEntries = collectTraceEntries(result);
  const ontologySidecar = result?.carbon?.ontology_sidecar || result?.ontology_sidecar || null;
  const answerTrace = result?.answer_trace || null;

  return (
    <div style={styles.shell}>
      <div style={styles.hero}>
        <p style={styles.eyebrow}>Advanced Audit</p>
        <h2 style={styles.title}>Raw execution details</h2>
        <p style={styles.text}>
          This page exposes the raw `steps` object, trace-level formulas and calculation items, and any ontology or debug payload returned with the response.
        </p>
        <div style={styles.buttonRow}>
          <button type="button" style={styles.button} onClick={onBack}>
            Back to answer
          </button>
          <button
            type="button"
            style={styles.ghostButton}
            onClick={() => navigator.clipboard && navigator.clipboard.writeText(jsonBlock(result))}
          >
            Copy full response JSON
          </button>
        </div>
      </div>

      <div style={styles.panel}>
        <p style={styles.panelTitle}>Answer Synthesis Trail</p>
        <p style={styles.panelText}>
          This block shows whether a live LLM was used for the final answer, which provider and model were configured, which API path was used, and what fallback path applied if the answer stayed deterministic.
        </p>
        {answerTrace ? (
          <pre style={styles.code}>{jsonBlock(answerTrace)}</pre>
        ) : (
          <p style={styles.panelText}>No answer synthesis trace was returned for this response.</p>
        )}
      </div>

      <div style={styles.panel}>
        <p style={styles.panelTitle}>Raw Steps Object</p>
        <p style={styles.panelText}>
          This is the solve pipeline step metadata exactly as returned by the backend.
        </p>
        {steps.length > 0 ? (
          <pre style={styles.code}>{jsonBlock(steps)}</pre>
        ) : (
          <p style={styles.panelText}>No step metadata was returned for this response.</p>
        )}
      </div>

      <div style={styles.panel}>
        <p style={styles.panelTitle}>Trace-Level Calculation Items</p>
        <p style={styles.panelText}>
          Each trace shows the stage, item id, formula, activity, factor, emissions, source refs, and notes when those fields are available.
        </p>
        {traceEntries.length > 0 ? (
          <div style={styles.grid}>
            {traceEntries.map((trace, index) => (
              <div key={`${trace.stageName}-${trace.item_id || index}`} style={styles.card}>
                <p style={styles.panelTitle}>
                  {titleize(trace.stageName)} {trace.label ? `· ${trace.label}` : ""}
                </p>
                <p style={styles.metric}>Item: {trace.item_id || "Unavailable"}</p>
                <p style={styles.metric}>Status: {titleize(trace.status)}</p>
                <p style={styles.metric}>Formula: {trace.formula || "Unavailable"}</p>
                <p style={styles.metric}>
                  Activity: {trace.activity_value !== null && trace.activity_value !== undefined ? `${formatNumber(trace.activity_value)} ${trace.activity_unit || ""}` : "Unavailable"}
                </p>
                <p style={styles.metric}>
                  Factor: {trace.factor_value !== null && trace.factor_value !== undefined ? `${formatNumber(trace.factor_value)} ${trace.factor_unit || ""}` : "Unavailable"}
                </p>
                <p style={styles.metric}>
                  Emissions: {trace.emissions_kg_co2e !== null && trace.emissions_kg_co2e !== undefined ? `${formatNumber(trace.emissions_kg_co2e)} kg CO2e` : "Unavailable"}
                </p>
                {Array.isArray(trace.source_refs) && trace.source_refs.length > 0 ? (
                  <pre style={styles.code}>{jsonBlock(trace.source_refs)}</pre>
                ) : null}
                {Array.isArray(trace.notes) && trace.notes.length > 0 ? (
                  <pre style={styles.code}>{jsonBlock(trace.notes)}</pre>
                ) : null}
              </div>
            ))}
          </div>
        ) : (
          <p style={styles.panelText}>No trace items were returned for this response.</p>
        )}
      </div>

      <div style={styles.panel}>
        <p style={styles.panelTitle}>Ontology and Debug Payloads</p>
        <p style={styles.panelText}>
          These fields appear only when the backend includes ontology sidecar or debug payloads in the response.
        </p>
        {ontologySidecar ? (
          <pre style={styles.code}>{jsonBlock(ontologySidecar)}</pre>
        ) : (
          <p style={styles.panelText}>No ontology or debug sidecar payload was returned for this response.</p>
        )}
      </div>
    </div>
  );
};

export default AdvancedAuditView;
