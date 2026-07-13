import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { theme } from "./theme";

document.documentElement.style.background = "#3017c8";
document.body.style.margin = "0";
document.body.style.fontFamily = theme.fontFamily.body;
document.body.style.background = "#3017c8";
document.body.style.color = theme.colors.textPrimary;

const container = document.getElementById("root");
const root = createRoot(container);
root.render(<App />);
