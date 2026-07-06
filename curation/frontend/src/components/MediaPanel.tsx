import { Sample } from "../api";

export function MediaPanel({ sample, view }: { sample: Sample; view: "M1" | "M2" | "M12" }) {
  const media = sample.media_paths ?? {};
  return (
    <section className="mediaPanel">
      <div className="mediaTitle">{sample.sample_id} · {view}</div>
      {(view === "M1" || view === "M12") && <div className="mediaBox">{media.vision || media.image || "visual view"}</div>}
      {(view === "M2" || view === "M12") && <div className="textBox">{media.text || media.audio || "text/audio view"}</div>}
    </section>
  );
}
