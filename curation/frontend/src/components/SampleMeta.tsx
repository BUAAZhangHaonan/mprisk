import { Sample } from "../api";

function typePillClass(types: string[]): string {
  if (types.length === 0) return "pill pillMuted";
  if (types.length > 1) return "pill pillWarn"; // disputed
  switch (types[0]) {
    case "Conflict":
      return "pill pillConflict";
    case "Aligned":
      return "pill pillAligned";
    case "Ambiguous":
      return "pill pillInfo";
    default:
      return "pill pillMuted";
  }
}

function TypePill({ types }: { types: string[] }) {
  if (types.length === 0) return <span className="pill pillMuted">—</span>;
  const label = types.length > 1 ? `${types.join(" / ")} (disputed)` : types[0];
  return <span className={typePillClass(types)}>{label}</span>;
}

export function SampleMeta({ sample }: { sample: Sample }) {
  const suggestion = sample.llm_sample_type_suggestion
    ? [sample.llm_sample_type_suggestion]
    : [];
  const count = sample.annotation_count ?? 0;
  const humanTypes = sample.human_types ?? [];

  return (
    <div className="sampleMeta">
      <div className="metaLeft">
        <span>{sample.source_dataset}</span>
        <span className="sourceId">{sample.source_id}</span>
        <span className="pill pillProto">{sample.protocol}</span>
      </div>
      <div className="metaRight">
        {sample.review_priority === "high" && <span className="pill pillWarn">priority</span>}
        <span className="verdictBlock">
          <span className="verdictLabel">初筛</span>
          <TypePill types={[sample.candidate_type]} />
        </span>
        <span className="verdictBlock">
          <span className="verdictLabel">LLM</span>
          <TypePill types={suggestion} />
        </span>
        <span className="verdictBlock">
          <span className="verdictLabel">人工</span>
          <TypePill types={humanTypes} />
          {count > 0 && (
            <span className="annotCount">
              ×{count}
              {sample.annotators && sample.annotators.length > 0
                ? ` (${sample.annotators.join(", ")})`
                : ""}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}
