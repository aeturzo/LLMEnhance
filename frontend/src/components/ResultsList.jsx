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

const styles = {
  shell: {
    display: "grid",
    gap: "1.15rem",
  },
  panel: {
    ...theme.glass.panel,
    padding: "1.2rem 1.25rem",
    borderRadius: "24px",
  },
  darkPanel: {
    ...theme.glass.darkPanel,
    padding: "1.2rem 1.25rem",
    borderRadius: "24px",
    color: theme.colors.textOnDark,
  },
  eyebrow: {
    margin: 0,
    fontSize: "0.78rem",
    fontWeight: 800,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "rgba(92, 66, 142, 0.82)",
  },
  darkEyebrow: {
    margin: 0,
    fontSize: "0.78rem",
    fontWeight: 800,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "rgba(255, 226, 245, 0.74)",
  },
  title: {
    margin: "0.35rem 0 0",
    fontSize: "1.55rem",
    lineHeight: 1.2,
    color: theme.colors.textPrimary,
    fontFamily: theme.fontFamily.display,
  },
  darkTitle: {
    margin: "0.35rem 0 0",
    fontSize: "1.45rem",
    lineHeight: 1.2,
    color: theme.colors.textOnDark,
    fontFamily: theme.fontFamily.display,
  },
  answer: {
    margin: "1rem 0 0",
    fontSize: "1.03rem",
    lineHeight: 1.7,
    color: theme.colors.textSecondary,
  },
  darkAnswer: {
    margin: "1rem 0 0",
    fontSize: "1.03rem",
    lineHeight: 1.7,
    color: theme.colors.textOnDarkMuted,
  },
  chipRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.55rem",
    marginTop: "1rem",
  },
  chip: {
    padding: "0.4rem 0.75rem",
    borderRadius: "999px",
    background: theme.gradients.chip,
    border: "1px solid rgba(255, 255, 255, 0.18)",
    color: theme.colors.textPrimary,
    fontSize: "0.84rem",
    fontWeight: 700,
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
  },
  darkChip: {
    padding: "0.4rem 0.75rem",
    borderRadius: "999px",
    background: theme.gradients.darkChip,
    border: "1px solid rgba(255, 255, 255, 0.16)",
    color: theme.colors.textOnDark,
    fontSize: "0.84rem",
    fontWeight: 700,
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
  },
  metricGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
    gap: "0.8rem",
    marginTop: "1rem",
  },
  metricCard: {
    ...theme.glass.card,
    padding: "0.95rem",
    borderRadius: "18px",
  },
  metricLabel: {
    margin: 0,
    fontSize: "0.8rem",
    fontWeight: 800,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "rgba(92, 66, 142, 0.74)",
  },
  metricValue: {
    margin: "0.35rem 0 0",
    fontSize: "1.35rem",
    fontWeight: 800,
    color: theme.colors.textPrimary,
  },
  metricNote: {
    margin: "0.4rem 0 0",
    fontSize: "0.88rem",
    lineHeight: 1.5,
    color: theme.colors.textSecondary,
  },
  sectionGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: "0.9rem",
    marginTop: "1rem",
  },
  card: {
    ...theme.glass.card,
    padding: "1rem",
    borderRadius: "18px",
  },
  cardTitle: {
    margin: 0,
    fontSize: "0.92rem",
    fontWeight: 800,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "rgba(92, 66, 142, 0.82)",
  },
  cardValue: {
    margin: "0.45rem 0 0",
    fontSize: "1.15rem",
    fontWeight: 800,
    color: theme.colors.textPrimary,
  },
  cardText: {
    margin: "0.5rem 0 0",
    fontSize: "0.92rem",
    lineHeight: 1.55,
    color: theme.colors.textSecondary,
  },
  list: {
    display: "grid",
    gap: "0.75rem",
    marginTop: "0.9rem",
  },
  listCard: {
    ...theme.glass.card,
    padding: "0.95rem 1rem",
    borderRadius: "16px",
  },
  mono: {
    fontFamily: theme.fontFamily.mono,
  },
  placeholder: {
    margin: 0,
    fontSize: "0.98rem",
    lineHeight: 1.7,
    color: theme.colors.textSecondary,
  },
  actionButton: {
    ...theme.glass.fieldDark,
    height: "44px",
    padding: "0 0.95rem",
    borderRadius: "999px",
    color: theme.colors.textOnDark,
    fontSize: "0.9rem",
    fontWeight: 800,
    cursor: "pointer",
  },
};

function renderChip(text, dark = false) {
  return (
    <span key={text} style={dark ? styles.darkChip : styles.chip}>
      {text}
    </span>
  );
}

function MetricCard({ label, value, note }) {
  return (
    <div style={styles.metricCard}>
      <p style={styles.metricLabel}>{label}</p>
      <p style={styles.metricValue}>{value}</p>
      {note ? <p style={styles.metricNote}>{note}</p> : null}
    </div>
  );
}

function StageCard({ stageName, stage }) {
  return (
    <div style={styles.card}>
      <p style={styles.cardTitle}>{titleize(stageName)}</p>
      <p style={styles.cardValue}>{formatNumber(stage.total_kg_co2e)} kg CO2e</p>
      <p style={styles.cardText}>
        Status: {titleize(stage.status)}. Quality: {titleize(stage.quality_status)}.
        {stage.uncertainty_pct !== null && stage.uncertainty_pct !== undefined
          ? ` Approximate uncertainty ${formatNumber(stage.uncertainty_pct, 1)}%.`
          : ""}
      </p>
      {stage.estimated_inputs && stage.estimated_inputs.length > 0 ? (
        <div style={styles.chipRow}>
          {stage.estimated_inputs.map((item) => renderChip(item))}
        </div>
      ) : null}
      {stage.missing_inputs && stage.missing_inputs.length > 0 ? (
        <p style={styles.cardText}>
          Missing inputs: {stage.missing_inputs.slice(0, 4).join("; ")}
        </p>
      ) : null}
    </div>
  );
}

function hasAdvancedAuditData(result) {
  if (!result) {
    return false;
  }

  const hasSteps = Array.isArray(result.steps) && result.steps.length > 0;
  const stageResults = result?.carbon?.stage_results || {};
  const hasTraces = Object.values(stageResults).some(
    (stage) => Array.isArray(stage?.traces) && stage.traces.length > 0
  );
  const hasOntology = Boolean(result?.carbon?.ontology_sidecar || result?.ontology_sidecar);

  return hasSteps || hasTraces || hasOntology;
}

const ResultsList = ({
  result,
  loading,
  error,
  hasSearched,
  onOpenAdvancedAudit,
  readyTitle = "Ask through the solve pipeline",
  readyDescription = "This page now uses the backend solve flow instead of the old flat search list. Carbon questions will surface totals, quality labels, uncertainty, stage breakdowns, provenance, and cited sources.",
}) => {
  if (loading) {
    return (
      <div style={styles.darkPanel}>
        <p style={styles.darkEyebrow}>Working</p>
        <h2 style={styles.darkTitle}>Composing an answer</h2>
        <p style={styles.darkAnswer}>
          The system is resolving the query, checking available evidence, and packaging the answer for the UI.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.panel}>
        <p style={styles.eyebrow}>Request Error</p>
        <h2 style={styles.title}>The query did not complete</h2>
        <p style={styles.answer}>{error}</p>
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div style={styles.panel}>
        <p style={styles.eyebrow}>Ready</p>
        <h2 style={styles.title}>{readyTitle}</h2>
        <p style={styles.placeholder}>
          {readyDescription}
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div style={styles.panel}>
        <p style={styles.eyebrow}>No Answer</p>
        <h2 style={styles.title}>The system returned an empty response</h2>
      </div>
    );
  }

  const carbon = result.carbon;
  const primaryStep = Array.isArray(result.steps) && result.steps.length > 0 ? result.steps[0] : null;
  const sources = Array.isArray(result.sources) ? result.sources : [];
  const showAdvancedAuditButton = hasAdvancedAuditData(result) && typeof onOpenAdvancedAudit === "function";
  const answerTrace = result?.answer_trace || null;

  return (
    <div style={styles.shell}>
      <div style={styles.darkPanel}>
        <p style={styles.darkEyebrow}>Answer</p>
        <h2 style={styles.darkTitle}>
          {carbon ? "Carbon Result" : "Solve Result"}
        </h2>
        <p style={styles.darkAnswer}>{result.answer || "No answer text returned."}</p>
        <div style={styles.chipRow}>
          {renderChip(`Mode: ${result.mode || "Unknown"}`, true)}
          {result.domain ? renderChip(`Domain: ${result.domain}`, true) : null}
          {result.product ? renderChip(`Product: ${result.product}`, true) : null}
          {typeof result.confidence === "number"
            ? renderChip(`Confidence: ${formatNumber(result.confidence, 2)}`, true)
            : null}
          {answerTrace
            ? renderChip(
              answerTrace.llm_used && answerTrace.model
                ? `LLM: ${answerTrace.model}`
                : "LLM: Not used",
              true
            )
            : null}
          {answerTrace && answerTrace.provider
            ? renderChip(`Provider: ${titleize(answerTrace.provider)}`, true)
            : null}
          {answerTrace && answerTrace.api
            ? renderChip(`API: ${titleize(answerTrace.api)}`, true)
            : null}
          {answerTrace && answerTrace.path
            ? renderChip(`Answer Path: ${titleize(answerTrace.path)}`, true)
            : null}
          {carbon && carbon.quality_status
            ? renderChip(`Quality: ${titleize(carbon.quality_status)}`, true)
            : null}
          {primaryStep && primaryStep.estimate_fallback_used
            ? renderChip("Estimate fallback used", true)
            : null}
        </div>
        {showAdvancedAuditButton ? (
          <div style={styles.chipRow}>
            <button type="button" style={styles.actionButton} onClick={onOpenAdvancedAudit}>
              Open advanced audit
            </button>
          </div>
        ) : null}
      </div>

      {carbon ? (
        <div style={styles.panel}>
          <p style={styles.eyebrow}>Carbon Summary</p>
          <h2 style={styles.title}>{carbon.product_name || "Product carbon profile"}</h2>

          <div style={styles.metricGrid}>
            <MetricCard
              label="Total Footprint"
              value={carbon.total_kg_co2e !== null ? `${formatNumber(carbon.total_kg_co2e)} kg CO2e` : "Partial"}
              note={carbon.total_kg_co2e === null ? "Official inputs are still incomplete." : null}
            />
            <MetricCard
              label="Quality"
              value={titleize(carbon.quality_status || "unknown")}
              note={carbon.used_bootstrap_estimates ? "Estimate defaults contributed to the total." : "Calculated from current official normalized inputs."}
            />
            <MetricCard
              label="Uncertainty"
              value={
                carbon.uncertainty_pct !== null && carbon.uncertainty_pct !== undefined
                  ? `${formatNumber(carbon.uncertainty_pct, 1)}%`
                  : "Unavailable"
              }
              note={
                carbon.uncertainty_kg_co2e !== null && carbon.uncertainty_kg_co2e !== undefined
                  ? `+/- ${formatNumber(carbon.uncertainty_kg_co2e)} kg CO2e`
                  : null
              }
            />
            <MetricCard
              label="Recyclability"
              value={
                carbon.recyclability && carbon.recyclability.recyclability_pct !== null
                  ? `${formatNumber(carbon.recyclability.recyclability_pct, 1)}%`
                  : "Unavailable"
              }
              note={
                carbon.recyclability && carbon.recyclability.recoverable_mass_kg !== null
                  ? `${formatNumber(carbon.recyclability.recoverable_mass_kg)} kg recoverable mass`
                  : null
              }
            />
          </div>

          {carbon.stage_results && Object.keys(carbon.stage_results).length > 0 ? (
            <div style={styles.sectionGrid}>
              {Object.entries(carbon.stage_results).map(([stageName, stage]) => (
                <StageCard key={stageName} stageName={stageName} stage={stage} />
              ))}
            </div>
          ) : null}

          {carbon.estimated_fields && carbon.estimated_fields.length > 0 ? (
            <div style={styles.list}>
              <div style={styles.listCard}>
                <p style={styles.cardTitle}>Estimated Inputs</p>
                <div style={styles.chipRow}>
                  {carbon.estimated_fields.map((item) => renderChip(item))}
                </div>
              </div>
            </div>
          ) : null}

          {carbon.provenance && carbon.provenance.length > 0 ? (
            <div style={styles.list}>
              <div style={styles.listCard}>
                <p style={styles.cardTitle}>Provenance</p>
                <div style={styles.list}>
                  {carbon.provenance.map((item, index) => (
                    <div key={`${item.field_name}-${index}`} style={styles.card}>
                      <p style={styles.cardTitle}>{item.label}</p>
                      <p style={styles.cardValue}>
                        {typeof item.value === "object" && item.value !== null
                          ? JSON.stringify(item.value)
                          : `${item.value ?? "Unavailable"}${item.unit ? ` ${item.unit}` : ""}`}
                      </p>
                      <p style={styles.cardText}>
                        Status: {titleize(item.status)}. Method: {titleize(item.method)}.
                        {item.uncertainty_pct !== null && item.uncertainty_pct !== undefined
                          ? ` Uncertainty ${formatNumber(item.uncertainty_pct, 1)}%.`
                          : ""}
                      </p>
                      {item.source_refs && item.source_refs.length > 0 ? (
                        <p style={{ ...styles.cardText, ...styles.mono }}>
                          Sources: {item.source_refs.join(" | ")}
                        </p>
                      ) : null}
                      {item.notes && item.notes.length > 0 ? (
                        <p style={styles.cardText}>{item.notes.join(" ")}</p>
                      ) : null}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : null}

          {carbon.missing_inputs && carbon.missing_inputs.length > 0 ? (
            <div style={styles.list}>
              <div style={styles.listCard}>
                <p style={styles.cardTitle}>Missing Inputs</p>
                <p style={styles.cardText}>{carbon.missing_inputs.join("; ")}</p>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      <div style={styles.panel}>
        <p style={styles.eyebrow}>Sources</p>
        <h2 style={styles.title}>Supporting evidence</h2>
        {sources.length === 0 ? (
          <p style={styles.answer}>No source snippets were returned for this answer.</p>
        ) : (
          <div style={styles.list}>
            {sources.map((source, index) => (
              <div key={`${source.id || source.title || "source"}-${index}`} style={styles.listCard}>
                <p style={styles.cardTitle}>{source.title || source.id || `Source ${index + 1}`}</p>
                {source.id ? <p style={{ ...styles.cardText, ...styles.mono }}>ID: {source.id}</p> : null}
                {source.snippet ? <p style={styles.cardText}>{source.snippet}</p> : null}
                {typeof source.score === "number" ? (
                  <p style={styles.cardText}>Score: {formatNumber(source.score, 3)}</p>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsList;
