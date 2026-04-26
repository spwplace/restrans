import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import "./index.css";
import App from "./App.tsx";

const baseUrl = import.meta.env.BASE_URL;
const basename = baseUrl === "/" ? "" : baseUrl;

const container = document.getElementById("root")!;

if (container.hasChildNodes()) {
  hydrateRoot(
    container,
    <StrictMode>
      <BrowserRouter basename={basename}>
        <App />
      </BrowserRouter>
    </StrictMode>
  );
} else {
  createRoot(container).render(
    <StrictMode>
      <BrowserRouter basename={basename}>
        <App />
      </BrowserRouter>
    </StrictMode>
  );
}
