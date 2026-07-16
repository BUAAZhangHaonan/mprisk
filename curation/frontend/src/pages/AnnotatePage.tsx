import { useEffect, useState } from "react";
import { AnnotationPayload, saveAnnotation, Sample } from "../api";
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

export function AnnotatePage({ sample }: { sample: Sample | null }) {
  const [view, setView] = useState<"M1" | "M2" | "M12">("M1");
  const [annotatorId, setAnnotatorId] = useState(initialAnnotatorId);
  const [payload, setPayload] = useState<AnnotationPayload | null>(
    sample ? emptyPayload(sample.sample_id, annotatorId) : null,
  );

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
  }, [sample, annotatorId]);

  if (!sample || !payload) {
    return <div className="empty">No sample selected.</div>;
  }

  const saveCurrent = () => {
    const currentPayload = { ...payload, annotator_id: annotatorId.trim() };
    setPayload(currentPayload);
    if (currentPayload.annotator_id) {
      window.localStorage.setItem(annotatorStorageKey, currentPayload.annotator_id);
    }
    return saveAnnotation(currentPayload);
  };

  return (
    <div className="page annotate">
      <KeyboardShortcuts onSave={saveCurrent} />
      <label className="annotatorBar">
        <span>Annotator</span>
        <input
          value={annotatorId}
          onChange={(event) => setAnnotatorId(event.target.value)}
          placeholder="annotator_id"
        />
      </label>
      <div className="segmented">
        {(["M1", "M2", "M12"] as const).map((key) => (
          <button className={view === key ? "active" : ""} onClick={() => setView(key)} key={key}>
            {key}
          </button>
        ))}
      </div>
      <MediaPanel sample={sample} view={view} />
      <LabelForm payload={payload} onChange={setPayload} onSave={saveCurrent} />
    </div>
  );
}
