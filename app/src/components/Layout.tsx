import { Link, useLocation } from "react-router";
import {
  Database,
  FlaskConical,
  Github,
  Home,
  Layers3,
  Microscope,
  Terminal,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { path: "/", label: "Home", icon: Home },
  { path: "/architecture", label: "Architecture", icon: Layers3 },
  { path: "/experiments", label: "Experiments", icon: FlaskConical },
  { path: "/tasks", label: "Tasks", icon: Database },
  { path: "/interpretability", label: "Interp", icon: Microscope },
  { path: "/reproducibility", label: "Repro", icon: Terminal },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/75">
        <div className="layout-header">
          <Link to="/" className="brand-link">
            <FlaskConical aria-hidden="true" className="h-5 w-5 text-primary" />
            <span>Resonance Research</span>
          </Link>
          <nav className="nav-scroll" aria-label="Primary navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "nav-link",
                    active
                      ? "nav-link-active"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent"
                  )}
                >
                  <Icon aria-hidden="true" className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
            <a
              href="https://github.com/spwplace/restrans"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link text-muted-foreground hover:text-foreground hover:bg-accent"
            >
              <Github aria-hidden="true" className="h-4 w-4" />
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        {children}
      </main>

      <footer className="border-t py-8 mt-12">
        <div className="container mx-auto px-4 text-sm text-muted-foreground footer-grid">
          <p className="mb-2">
            Resonance Transformer research dossier ·{" "}
            <a href="https://github.com/spwplace/restrans" className="underline hover:text-foreground">GitHub</a>
          </p>
          <p>
            Built to summarize current evidence, failure modes, open questions, and reproduction paths.
          </p>
        </div>
      </footer>
    </div>
  );
}
