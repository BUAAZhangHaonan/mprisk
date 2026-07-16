import { Sample } from "../api";

export function SampleMeta({ sample }: { sample: Sample }) {
  return (
    <>
      <span className="sampleId">{sample.sample_id}</span>
      <span>{sample.source_dataset}</span>
      <span>{sample.protocol}</span>
      <span className="pill">{sample.candidate_type}</span>
    </>
  );
}
