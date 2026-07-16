import { AnnotationPayload } from "../api";

const labels = ["positive", "negative", "neutral", "uncertain", "invalid"];
const sampleTypes = ["Conflict", "Ambiguous", "Aligned"];
const dominant = ["M1", "M2", "balanced", "unclear"];
const keys = {
  m1: {
    label: "m1_label",
    specific: "m1_specific_affect",
    clear: "m1_is_clear",
    confidence: "m1_confidence",
  },
  m2: {
    label: "m2_label",
    specific: "m2_specific_affect",
    clear: "m2_is_clear",
    confidence: "m2_confidence",
  },
  joint: {
    label: "joint_label",
    specific: "joint_specific_affect",
    clear: "joint_is_clear",
    confidence: "joint_confidence",
  },
} as const;

export function LabelForm({
  payload,
  onChange,
  onSave,
}: {
  payload: AnnotationPayload;
  onChange: (payload: AnnotationPayload) => void;
  onSave: () => void;
}) {
  const set = (key: keyof AnnotationPayload, value: string | number | boolean | string[]) =>
    onChange({ ...payload, [key]: value });

  return (
    <form className="labelForm" onSubmit={(event) => { event.preventDefault(); onSave(); }}>
      {(["m1", "m2", "joint"] as const).map((prefix) => (
        <fieldset key={prefix}>
          <legend>{prefix.toUpperCase()}</legend>
          <select value={String(payload[keys[prefix].label])} onChange={(event) => set(keys[prefix].label, event.target.value)}>
            {labels.map((label) => <option key={label}>{label}</option>)}
          </select>
          <input
            value={String(payload[keys[prefix].specific])}
            onChange={(event) => set(keys[prefix].specific, event.target.value)}
            placeholder="specific affect"
          />
          <label>
            <input
              type="checkbox"
              checked={Boolean(payload[keys[prefix].clear])}
              onChange={(event) => set(keys[prefix].clear, event.target.checked)}
            />
            clear
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={Number(payload[keys[prefix].confidence])}
            onChange={(event) => set(keys[prefix].confidence, Number(event.target.value))}
          />
        </fieldset>
      ))}
      <div className="formRow">
        <select value={payload.sample_type} onChange={(event) => set("sample_type", event.target.value)}>
          {sampleTypes.map((value) => <option key={value}>{value}</option>)}
        </select>
        <select value={payload.dominant_modality} onChange={(event) => set("dominant_modality", event.target.value)}>
          {dominant.map((value) => <option key={value}>{value}</option>)}
        </select>
      </div>
      <textarea value={payload.notes} onChange={(event) => set("notes", event.target.value)} />
      <button className="saveButton" type="submit">Save</button>
    </form>
  );
}
