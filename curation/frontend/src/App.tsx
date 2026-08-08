import { useCallback, useEffect, useState } from "react";
import { ListChecks, Scale, Send } from "lucide-react";
import { fetchProgress, fetchSamples, PaginatedSamples, Progress, Sample } from "./api";
import { QueuePage } from "./pages/QueuePage";
import { AnnotatePage } from "./pages/AnnotatePage";
import { AdjudicationPage } from "./pages/AdjudicationPage";
import "./style.css";

type Page = "queue" | "annotate" | "adjudication";

export type QueueFilters = {
  candidateType: string;
  llmType: string;
  humanType: string;
  protocol: string;
  annotatedFilter: "all" | "only_mine" | "hide_mine";
  disagreementOnly: boolean;
};

const annotatorStorageKey = "mprisk.annotator_id";
const PAGE_SIZE = 50;

export default function App() {
  const [page, setPage] = useState<Page>("queue");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [pagination, setPagination] = useState<PaginatedSamples | null>(null);
  const [pageNum, setPageNum] = useState(1);
  const [selected, setSelected] = useState<Sample | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [filters, setFilters] = useState<QueueFilters>({
    candidateType: "",
    llmType: "",
    humanType: "",
    protocol: "",
    annotatedFilter: "all",
    disagreementOnly: false,
  });

  const reloadProgress = useCallback(() => {
    fetchProgress().then(setProgress).catch(() => setProgress(null));
  }, []);

  useEffect(() => {
    const annotatorId = window.localStorage.getItem(annotatorStorageKey)?.trim() ?? "";
    fetchSamples({
      candidateType: filters.candidateType || undefined,
      llmType: filters.llmType || undefined,
      humanType: filters.humanType || undefined,
      protocol: filters.protocol || undefined,
      onlyAnnotator: filters.annotatedFilter === "only_mine" && annotatorId ? annotatorId : undefined,
      excludeAnnotator: filters.annotatedFilter === "hide_mine" && annotatorId ? annotatorId : undefined,
      disagreementOnly: filters.disagreementOnly || undefined,
      page: pageNum,
      pageSize: PAGE_SIZE,
    })
      .then((res) => {
        setSamples(res.items);
        setPagination(res);
      })
      .catch(() => {
        setSamples([]);
        setPagination(null);
      });
  }, [filters, pageNum]);

  useEffect(() => {
    reloadProgress();
  }, [reloadProgress]);

  // reset to page 1 when filters change
  const onFiltersChange = useCallback((next: QueueFilters) => {
    setFilters(next);
    setPageNum(1);
  }, []);

  const selectNext = useCallback(() => {
    reloadProgress();
    setSelected((current) => {
      if (!current) return samples[0] ?? null;
      const index = samples.findIndex((item) => item.sample_id === current.sample_id);
      if (index + 1 < samples.length) return samples[index + 1];
      // last on page → try next page
      if (pagination && pageNum < pagination.total_pages) {
        setPageNum(pageNum + 1);
      }
      return samples[index + 1] ?? null;
    });
  }, [samples, reloadProgress, pagination, pageNum]);

  const navigate = useCallback((direction: "prev" | "next") => {
    setSelected((current) => {
      if (samples.length === 0) return current;
      if (!current) return samples[0] ?? null;
      const index = samples.findIndex((item) => item.sample_id === current.sample_id);
      if (direction === "next") {
        if (index + 1 < samples.length) return samples[index + 1];
        if (pagination && pageNum < pagination.total_pages) {
          setPageNum(pageNum + 1);
          return null;
        }
        return current;
      } else {
        if (index > 0) return samples[index - 1];
        if (pagination && pageNum > 1) {
          setPageNum(pageNum - 1);
          return null;
        }
        return current;
      }
    });
  }, [samples, pagination, pageNum]);

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
            filters={filters}
            onFiltersChange={onFiltersChange}
            progress={progress}
            pagination={pagination}
            pageNum={pageNum}
            onPageChange={setPageNum}
            onSelect={(sample) => {
              setSelected(sample);
              setPage("annotate");
            }}
          />
        )}
        {page === "annotate" && (
          <AnnotatePage
            sample={selected ?? samples[0] ?? null}
            onSaved={selectNext}
            onNavigate={navigate}
          />
        )}
        {page === "adjudication" && <AdjudicationPage />}
      </section>
    </main>
  );
}
