import React, { useState } from "react";
import SolveWorkspace from "../components/SolveWorkspace";
import OrchestratedWorkspace from "../components/OrchestratedWorkspace";
import { theme } from "../theme";

const styles = {
  page: {
    minHeight: "100vh",
    padding: "2.5rem 1.2rem 3.5rem",
    position: "relative",
    overflow: "hidden",
    background: theme.gradients.page,
    color: theme.colors.textPrimary,
    fontFamily: theme.fontFamily.body,
    boxSizing: "border-box",
  },
  backdrop: {
    position: "absolute",
    inset: 0,
    overflow: "hidden",
    pointerEvents: "none",
  },
  orbLarge: {
    position: "absolute",
    top: "-120px",
    right: "-80px",
    width: "420px",
    height: "420px",
    borderRadius: "50%",
    background: "radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.38), rgba(255, 255, 255, 0.08) 58%, transparent 74%)",
    border: "1px solid rgba(255, 255, 255, 0.2)",
    backdropFilter: "blur(20px)",
    WebkitBackdropFilter: "blur(20px)",
  },
  orbMid: {
    position: "absolute",
    left: "-90px",
    top: "26%",
    width: "260px",
    height: "260px",
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(255, 255, 255, 0.24), rgba(255, 255, 255, 0.08) 58%, transparent 72%)",
    filter: "blur(8px)",
  },
  orbBottom: {
    position: "absolute",
    right: "8%",
    bottom: "-100px",
    width: "340px",
    height: "340px",
    borderRadius: "50%",
    background: "radial-gradient(circle, rgba(255, 192, 124, 0.34), rgba(255, 143, 55, 0.14) 52%, transparent 74%)",
    filter: "blur(10px)",
  },
  frame: {
    maxWidth: "1180px",
    margin: "0 auto",
    display: "grid",
    gap: "1.35rem",
    position: "relative",
    zIndex: 1,
  },
  switcher: {
    display: "flex",
    flexWrap: "wrap",
    gap: "0.75rem",
    alignItems: "center",
  },
  switchLabel: {
    margin: 0,
    fontSize: "0.82rem",
    fontWeight: 800,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "rgba(255, 244, 252, 0.84)",
    textShadow: "0 1px 12px rgba(41, 14, 73, 0.22)",
  },
  switchButton: {
    ...theme.glass.panel,
    padding: "0.78rem 1rem",
    borderRadius: "999px",
    color: theme.colors.textPrimary,
    fontSize: "0.92rem",
    fontWeight: 800,
    cursor: "pointer",
  },
  switchButtonActive: {
    background: theme.gradients.buttonActive,
    color: theme.colors.textOnDark,
    border: "1px solid rgba(255, 255, 255, 0.34)",
    boxShadow: "0 18px 42px rgba(92, 27, 156, 0.34)",
  },
};

const HomePage = () => {
  const [workspace, setWorkspace] = useState("solve");

  return (
    <div style={styles.page}>
      <div aria-hidden="true" style={styles.backdrop}>
        <div style={styles.orbLarge} />
        <div style={styles.orbMid} />
        <div style={styles.orbBottom} />
      </div>
      <div style={styles.frame}>
        <div style={styles.switcher}>
          <p style={styles.switchLabel}>Workspace</p>
          <button
            type="button"
            style={{
              ...styles.switchButton,
              ...(workspace === "solve" ? styles.switchButtonActive : {}),
            }}
            onClick={() => setWorkspace("solve")}
          >
            Carbon Solve
          </button>
          <button
            type="button"
            style={{
              ...styles.switchButton,
              ...(workspace === "orchestrated" ? styles.switchButtonActive : {}),
            }}
            onClick={() => setWorkspace("orchestrated")}
          >
            Auto Orchestrator
          </button>
        </div>

        {workspace === "solve" ? <SolveWorkspace /> : <OrchestratedWorkspace />}
      </div>
    </div>
  );
};

export default HomePage;
