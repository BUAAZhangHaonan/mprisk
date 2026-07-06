import { AnnotationPayload } from "../api";

const labels = ["positive", "negative", "neutral", "uncertain", "invalid"];
const sampleTypes = ["Conflict", "Ambiguous", "Aligned"];
const dominant = ["M1", "M2", "balanced", "unclear"];

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
          <select value={payload[`${prefix}_label`]} onChange={(event) => set(`${prefix}_label`, event.target.value)}>
            {labels.map((label) => <option key={label}>{label}</option>)}
          </select>
          <input
            value={payload[`${prefix}_specific_affect`]}
            onChange={(event) => set(`${prefix}_specific_affect`, event.target.value)}
            placeholder="specific affect"
          />
          <label>
            <input
              type="checkbox"
              checked={Boolean(payload[`${prefix}_is_clear`])}
              onChange={(event) => set(`${prefix}_is_clear`, event.target.checked)}
            />
            clear
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={Number(payload[`${prefix}_confidence`])}
            onChange={(event) => set(`${prefix}_confidence`, Number(event.target.value))}
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
