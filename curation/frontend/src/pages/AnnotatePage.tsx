import { useEffect, useState } from "react";
import { AnnotationPayload, fetchSample, saveAnnotation, Sample, ViewSuggestion } from "../api";
import { LabelForm } from "../components/LabelForm";
import { MediaPanel } from "../components/MediaPanel";
import { KeyboardShortcuts } from "../components/KeyboardShortcuts";

const annotatorStorageKey = "mprisk.annotator_id";

function initialAnnotatorId() {
  const params = new URLSearchParams(window.location.search);
  const urlAnnotator = params.get("annotator_id")?.trim();
  if (urlAnnotator) return urlAnnotator;
  return window.localStorage.getItem(annotatorStorageKey)?.trim() ?? "";
}

const emptyPayload = (sampleId: string, annotatorId: string): AnnotationPayload => ({
  sample_id: sampleId,
  annotator_id: annotatorId,
  m1_label: "uncertain",
  m2_label: "uncertain",
  joint_label: "uncertain",
  m1_specific_affect: "",
  m2_specific_affect: "",
  joint_specific_affect: "",
  m1_is_clear: false,
  m2_is_clear: false,
  joint_is_clear: false,
  m1_confidence: 0.5,
  m2_confidence: 0.5,
  joint_confidence: 0.5,
  sample_type: "Ambiguous",
  dominant_modality: "unclear",
  quality_flags: [],
  notes: "",
});

function SuggestionRow({ view, suggestion }: { view: string; suggestion: ViewSuggestion }) {
  return (
    <div className="llmRow">
      <span className="llmView">{view}</span>
      <span className="pill pillLlm">{suggestion.label}</span>
      <span className="llmConfidence">{Math.round(suggestion.confidence * 100)}%</span>
      <span className="llmEvidence">
        {suggestion.specific_affect ? `${suggestion.specific_affect} — ` : ""}
        {suggestion.evidence}
      </span>
    </div>
  );
}

function LlmSuggestions({ sample }: { sample: Sample }) {
  const screening = sample.llm_screening;
  if (!screening) return null;
  return (
    <details className="llmPanel">
      <summary>
        Gemini suggestion: {screening.sample_type_suggestion} / dominant{" "}
        {screening.dominant_modality_suggestion}
        {sample.llm_agrees === false ? " (disagrees with initial screen!)" : ""}
        <span className="llmHint"> — make your own judgment first</span>
      </summary>
      {(["M1", "M2", "M12"] as const).map((view) =>
        screening.view_outputs?.[view] ? (
          <SuggestionRow key={view} view={view} suggestion={screening.view_outputs[view]} />
        ) : null,
      )}
      {screening.quality_flags?.length > 0 && (
        <div className="llmRow">
          <span className="llmView">flags</span>
          <span>{screening.quality_flags.join(", ")}</span>
        </div>
      )}
    </details>
  );
}

function HistoryRow({ ann }: { ann: AnnotationPayload }) {
  return (
    <div className="llmRow">
      <span className="llmView">{ann.annotator_id}</span>
      <span className="pill pillLlm">{ann.sample_type}</span>
      <span className="pill">{ann.dominant_modality}</span>
      <span className="llmEvidence">
        M1 {ann.m1_label}{ann.m1_is_clear ? "" : "?"} /{" "}
        M2 {ann.m2_label}{ann.m2_is_clear ? "" : "?"} /{" "}
        M12 {ann.joint_label}{ann.joint_is_clear ? "" : "?"}
      </span>
    </div>
  );
}

function HistoryPanel({ annotations }: { annotations: AnnotationPayload[] }) {
  if (!annotations || annotations.length === 0) return null;
  return (
    <details className="llmPanel" open>
      <summary>
        Human annotations: {annotations.length} (distinct annotators:{" "}
        {new Set(annotations.map((a) => a.annotator_id)).size})
      </summary>
      {annotations.map((ann, idx) => (
        <HistoryRow key={`${ann.annotator_id}-${idx}`} ann={ann} />
      ))}
    </details>
  );
}

export function AnnotatePage({
  sample,
  onSaved,
  onNavigate,
}: {
  sample: Sample | null;
  onSaved?: () => void;
  onNavigate?: (direction: "prev" | "next") => void;
}) {
  const [view, setView] = useState<"M1" | "M2" | "M12">("M1");
  const [annotatorId, setAnnotatorId] = useState(initialAnnotatorId);
  const [status, setStatus] = useState("");
  const [payload, setPayload] = useState<AnnotationPayload | null>(
    sample ? emptyPayload(sample.sample_id, annotatorId) : null,
  );
  const [history, setHistory] = useState<AnnotationPayload[]>(sample?.annotations ?? []);

  useEffect(() => {
    if (annotatorId.trim()) {
      window.localStorage.setItem(annotatorStorageKey, annotatorId.trim());
    }
  }, [annotatorId]);

  useEffect(() => {
    setPayload((current) => {
      if (!sample) return null;
      if (!current || current.sample_id !== sample.sample_id) {
        return emptyPayload(sample.sample_id, annotatorId);
      }
      if (current.annotator_id !== annotatorId) {
        return { ...current, annotator_id: annotatorId };
      }
      return current;
    });
    setStatus("");
    // fetch fresh detail to populate history
    if (sample) {
      fetchSample(sample.sample_id)
        .then((detail) => setHistory(detail.annotations ?? []))
        .catch(() => setHistory([]));
    } else {
      setHistory([]);
    }
  }, [sample, annotatorId]);

  if (!sample || !payload) {
    return <div className="empty">No sample selected.</div>;
  }

  const saveCurrent = async () => {
    const trimmed = annotatorId.trim();
    if (!trimmed) {
      setStatus("annotator_id is required");
      return;
    }
    const currentPayload = { ...payload, annotator_id: trimmed };
    setPayload(currentPayload);
    window.localStorage.setItem(annotatorStorageKey, trimmed);
    try {
      await saveAnnotation(currentPayload);
      setStatus("saved");
      // refresh history so the annotator sees their own just-saved record
      fetchSample(currentPayload.sample_id)
        .then((detail) => setHistory(detail.annotations ?? []))
        .catch(() => {});
      onSaved?.();
    } catch (error) {
      setStatus(String(error));
    }
  };

  return (
    <div className="page annotate">
      <KeyboardShortcuts onSave={saveCurrent} />
      <div className="navBar">
        <button onClick={() => onNavigate?.("prev")} title="上一条">‹ 上一条</button>
        <span className="navSampleId">{sample.sample_id}</span>
        <button onClick={() => onNavigate?.("next")} title="下一条">下一条 ›</button>
      </div>
      <label className="annotatorBar">
        <span>Annotator</span>
        <input
          value={annotatorId}
          onChange={(event) => setAnnotatorId(event.target.value)}
          placeholder="annotator_id"
        />
        {status && <span className="saveStatus">{status}</span>}
      </label>
      <div className="segmented">
        {(["M1", "M2", "M12"] as const).map((key) => (
          <button className={view === key ? "active" : ""} onClick={() => setView(key)} key={key}>
            {key}
          </button>
        ))}
      </div>
      <MediaPanel sample={sample} view={view} />
      <LlmSuggestions sample={sample} />
      <HistoryPanel annotations={history} />
      <LabelForm payload={payload} onChange={setPayload} onSave={saveCurrent} />
    </div>
  );
}
