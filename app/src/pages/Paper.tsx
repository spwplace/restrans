import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BookOpen, Loader2 } from "lucide-react";

const sections = [
  { file: "resonance_paper_sec00.md", title: "Abstract" },
  { file: "resonance_paper_sec01.md", title: "1. Introduction" },
  { file: "resonance_paper_sec02.md", title: "2. Related Work" },
  { file: "resonance_paper_sec03.md", title: "3. Methodology" },
  { file: "resonance_paper_sec04.md", title: "4. Experiments & Results" },
  { file: "resonance_paper_sec05.md", title: "5. Discussion" },
  { file: "resonance_paper_sec06.md", title: "6. Conclusion" },
];

export default function Paper() {
  const [contents, setContents] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [activeSection, setActiveSection] = useState("");

  useEffect(() => {
    async function load() {
      const results: Record<string, string> = {};
      for (const sec of sections) {
        try {
          const res = await fetch(`./${sec.file}`);
          results[sec.file] = await res.text();
        } catch {
          results[sec.file] = `*Failed to load ${sec.file}*`;
        }
      }
      setContents(results);
      setLoading(false);
    }
    load();
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        }
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );

    for (const sec of sections) {
      const el = document.getElementById(sec.file);
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, [loading]);

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-20 flex items-center justify-center">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading paper sections...
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="flex items-center gap-3 mb-8">
        <BookOpen className="h-6 w-6 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">
          Resonance Transformers: Dual-Stream Architectures with Phase-Structured Attention
        </h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* TOC Sidebar */}
        <div className="hidden lg:block">
          <div className="sticky top-20">
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground uppercase tracking-wider">
              Contents
            </h3>
            <ScrollArea className="h-[calc(100vh-8rem)]">
              <nav className="space-y-1">
                {sections.map((sec) => (
                  <button
                    key={sec.file}
                    onClick={() => scrollTo(sec.file)}
                    className={`block w-full text-left px-3 py-1.5 rounded-md text-sm transition-colors ${
                      activeSection === sec.file
                        ? "bg-primary text-primary-foreground font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent"
                    }`}
                  >
                    {sec.title}
                  </button>
                ))}
              </nav>
            </ScrollArea>
          </div>
        </div>

        {/* Content */}
        <div className="lg:col-span-3 space-y-8">
          {sections.map((sec) => (
            <section key={sec.file} id={sec.file}>
              <Card>
                <CardContent className="pt-6">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({ children }) => (
                        <h1 className="text-2xl font-bold mt-2 mb-4">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-xl font-semibold mt-6 mb-3">{children}</h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-lg font-semibold mt-4 mb-2">{children}</h3>
                      ),
                      h4: ({ children }) => (
                        <h4 className="text-base font-semibold mt-3 mb-2">{children}</h4>
                      ),
                      p: ({ children }) => (
                        <p className="text-sm text-muted-foreground leading-relaxed mb-4">{children}</p>
                      ),
                      ul: ({ children }) => (
                        <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 mb-4 ml-2">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-1 mb-4 ml-2">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-2 border-primary pl-4 italic text-sm text-muted-foreground mb-4">
                          {children}
                        </blockquote>
                      ),
                      code: ({ children, className }) => {
                        const inline = !className;
                        return inline ? (
                          <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono">
                            {children}
                          </code>
                        ) : (
                          <pre className="bg-muted rounded-lg p-4 overflow-x-auto text-xs font-mono mb-4">
                            <code>{children}</code>
                          </pre>
                        );
                      },
                      table: ({ children }) => (
                        <div className="overflow-x-auto mb-4">
                          <table className="w-full text-sm border-collapse">
                            {children}
                          </table>
                        </div>
                      ),
                      thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
                      th: ({ children }) => (
                        <th className="border px-3 py-2 text-left font-semibold text-xs">{children}</th>
                      ),
                      td: ({ children }) => (
                        <td className="border px-3 py-2 text-xs text-muted-foreground">{children}</td>
                      ),
                      img: ({ src, alt }) => (
                        <figure className="my-6">
                          <img
                            src={src || ""}
                            alt={alt || ""}
                            className="rounded-lg border max-w-full mx-auto"
                          />
                          {alt && (
                            <figcaption className="text-center text-xs text-muted-foreground mt-2">
                              {alt}
                            </figcaption>
                          )}
                        </figure>
                      ),
                    }}
                  >
                    {contents[sec.file] || ""}
                  </ReactMarkdown>
                </CardContent>
              </Card>
              <Separator className="my-8" />
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
