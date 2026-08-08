import { PaginatedSamples, Progress, Sample } from "../api";
import { QueueFilters } from "../App";
import { SampleMeta } from "../components/SampleMeta";

const typeOptions = [
  { value: "", label: "全部" },
  { value: "Conflict", label: "Conflict" },
  { value: "Aligned", label: "Aligned" },
  { value: "Ambiguous", label: "Ambiguous" },
];
const protoOptions = [
  { value: "", label: "全部" },
  { value: "VT", label: "VT" },
  { value: "VA", label: "VA" },
];

function ToggleGroup({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="toggleGroup">
      {options.map((opt) => (
        <button
          key={opt.value}
          className={"toggleBtn" + (value === opt.value ? " active" : "")}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function ProgressBars({ progress }: { progress: Progress | null }) {
  if (!progress || progress.groups.length === 0) return null;
  return (
    <div className="progressBars">
      {progress.groups.map((group) => {
        const percent = group.total ? Math.round((group.annotated_twice / group.total) * 100) : 0;
        return (
          <div className="progressRow" key={`${group.protocol}:${group.candidate_type}`}>
            <span className="progressLabel">
              {group.protocol} / {group.candidate_type}
            </span>
            <div className="progressTrack">
              <div className="progressFill" style={{ width: `${percent}%` }} />
            </div>
            <span className="progressNums">
              {group.annotated_twice}×2 / {group.annotated_once}×1 / {group.total}
            </span>
          </div>
        );
      })}
      <div className="progressTotal">{progress.total_annotations} annotations saved</div>
    </div>
  );
}

function Pagination({
  pagination,
  pageNum,
  onPageChange,
}: {
  pagination: PaginatedSamples | null;
  pageNum: number;
  onPageChange: (page: number) => void;
}) {
  if (!pagination) return null;
  const { total, total_pages, page_size } = pagination;
  if (total === 0) return null;
  const start = (pageNum - 1) * page_size + 1;
  const end = Math.min(pageNum * page_size, total);
  return (
    <div className="pagination">
      <button disabled={pageNum <= 1} onClick={() => onPageChange(1)} title="First">«</button>
      <button disabled={pageNum <= 1} onClick={() => onPageChange(pageNum - 1)} title="Prev">‹</button>
      <span className="pageInfo">
        Page <strong>{pageNum}</strong> / {total_pages} · {start}-{end} of {total}
      </span>
      <button disabled={pageNum >= total_pages} onClick={() => onPageChange(pageNum + 1)} title="Next">›</button>
      <button disabled={pageNum >= total_pages} onClick={() => onPageChange(total_pages)} title="Last">»</button>
    </div>
  );
}

export function QueuePage({
  samples,
  filters,
  onFiltersChange,
  progress,
  pagination,
  pageNum,
  onPageChange,
  onSelect,
}: {
  samples: Sample[];
  filters: QueueFilters;
  onFiltersChange: (filters: QueueFilters) => void;
  progress: Progress | null;
  pagination: PaginatedSamples | null;
  pageNum: number;
  onPageChange: (page: number) => void;
  onSelect: (sample: Sample) => void;
}) {
  return (
    <div className="page">
      <header className="pageHeader">
        <h1>Sample Queue</h1>
        <div className="count">{pagination ? `${pagination.total} items` : `${samples.length} items`}</div>
      </header>
      <ProgressBars progress={progress} />
      <div className="filters">
        <div className="filterRow">
          <span className="filterLabel">初筛</span>
          <ToggleGroup
            value={filters.candidateType}
            options={typeOptions}
            onChange={(v) => onFiltersChange({ ...filters, candidateType: v })}
          />
        </div>
        <div className="filterRow">
          <span className="filterLabel">LLM</span>
          <ToggleGroup
            value={filters.llmType}
            options={typeOptions}
            onChange={(v) => onFiltersChange({ ...filters, llmType: v })}
          />
        </div>
        <div className="filterRow">
          <span className="filterLabel">人工</span>
          <ToggleGroup
            value={filters.humanType}
            options={typeOptions}
            onChange={(v) => onFiltersChange({ ...filters, humanType: v })}
          />
        </div>
        <div className="filterRow">
          <span className="filterLabel">协议</span>
          <ToggleGroup
            value={filters.protocol}
            options={protoOptions}
            onChange={(v) => onFiltersChange({ ...filters, protocol: v })}
          />
        </div>
        <div className="filterRow">
          <span className="filterLabel">我的标注</span>
          <ToggleGroup
            value={filters.annotatedFilter}
            options={[
              { value: "all", label: "全部" },
              { value: "only_mine", label: "只看我已标的" },
              { value: "hide_mine", label: "隐藏我已标的" },
            ]}
            onChange={(v) => onFiltersChange({ ...filters, annotatedFilter: v as "all" | "only_mine" | "hide_mine" })}
          />
        </div>
        <div className="filterRow">
          <span className="filterLabel">其他</span>
          <button
            className={"toggleBtn" + (filters.disagreementOnly ? " active" : "")}
            onClick={() => onFiltersChange({ ...filters, disagreementOnly: !filters.disagreementOnly })}
          >
            只看初筛/LLM分歧
          </button>
        </div>
      </div>
      <Pagination pagination={pagination} pageNum={pageNum} onPageChange={onPageChange} />
      <div className="queueGrid">
        {samples.map((sample) => (
          <button className="sampleRow" key={sample.sample_id} onClick={() => onSelect(sample)}>
            <SampleMeta sample={sample} />
          </button>
        ))}
      </div>
      <Pagination pagination={pagination} pageNum={pageNum} onPageChange={onPageChange} />
    </div>
  );
}
