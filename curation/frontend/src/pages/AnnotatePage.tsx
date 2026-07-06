import { useState } from "react";
import { AnnotationPayload, saveAnnotation, Sample } from "../api";
import { LabelForm } from "../components/LabelForm";
import { MediaPanel } from "../components/MediaPanel";
import { KeyboardShortcuts } from "../components/KeyboardShortcuts";

const emptyPayload = (sampleId: string): AnnotationPayload => ({
  sample_id: sampleId,
  annotator_id: "annotator",
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
  const [payload, setPayload] = useState<AnnotationPayload | null>(
    sample ? emptyPayload(sample.sample_id) : null,
  );

  if (!sample || !payload) {
    return <div className="empty">No sample selected.</div>;
  }

  return (
    <div className="page annotate">
      <KeyboardShortcuts onSave={() => saveAnnotation(payload)} />
      <div className="segmented">
        {(["M1", "M2", "M12"] as const).map((key) => (
          <button className={view === key ? "active" : ""} onClick={() => setView(key)} key={key}>
            {key}
          </button>
        ))}
      </div>
      <MediaPanel sample={sample} view={view} />
      <LabelForm payload={payload} onChange={setPayload} onSave={() => saveAnnotation(payload)} />
    </div>
  );
}
