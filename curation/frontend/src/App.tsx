import { useEffect, useState } from "react";
import { ListChecks, Scale, Send } from "lucide-react";
import { fetchSamples, Sample } from "./api";
import { QueuePage } from "./pages/QueuePage";
import { AnnotatePage } from "./pages/AnnotatePage";
import { AdjudicationPage } from "./pages/AdjudicationPage";
import "./style.css";

type Page = "queue" | "annotate" | "adjudication";

export default function App() {
  const [page, setPage] = useState<Page>("queue");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<Sample | null>(null);

  useEffect(() => {
    fetchSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  const nav = [
    ["queue", ListChecks, "Queue"],
    ["annotate", Send, "Annotate"],
    ["adjudication", Scale, "Adjudication"],
  ] as const;

  return (
    <main className="shell">
      <aside className="rail">
        <div className="brand">mprisk curation</div>
        {nav.map(([key, Icon, label]) => (
          <button className={page === key ? "active" : ""} onClick={() => setPage(key)} key={key}>
            <Icon size={18} />
            <span>{label}</span>
          </button>
        ))}
      </aside>
      <section className="workspace">
        {page === "queue" && (
          <QueuePage
            samples={samples}
            onSelect={(sample) => {
              setSelected(sample);
              setPage("annotate");
            }}
          />
        )}
        {page === "annotate" && <AnnotatePage sample={selected ?? samples[0] ?? null} />}
        {page === "adjudication" && <AdjudicationPage />}
      </section>
    </main>
  );
}
