import ReactDOMServer from "react-dom/server";
import { StaticRouter } from "react-router";
import App from "./App";

const baseUrl = import.meta.env.BASE_URL;
const basename = baseUrl === "/" ? "" : baseUrl;

export function render(url: string) {
  return ReactDOMServer.renderToString(
    <StaticRouter location={url} basename={basename}>
      <App />
    </StaticRouter>
  );
}
