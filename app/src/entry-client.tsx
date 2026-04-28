import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import "./index.css";
import App from "./App.tsx";

const baseUrl = import.meta.env.BASE_URL;
const basename = baseUrl === "/" ? "" : baseUrl;

const container = document.getElementById("root")!;

createRoot(container).render(
  <StrictMode>
    <BrowserRouter basename={basename}>
      <App />
    </BrowserRouter>
  </StrictMode>
);
