import { Sample } from "../api";
import { SampleMeta } from "../components/SampleMeta";

export function QueuePage({ samples, onSelect }: { samples: Sample[]; onSelect: (sample: Sample) => void }) {
  return (
    <div className="page">
      <header className="pageHeader">
        <h1>Sample Queue</h1>
        <div className="count">{samples.length} items</div>
      </header>
      <div className="queueGrid">
        {samples.map((sample) => (
          <button className="sampleRow" key={sample.sample_id} onClick={() => onSelect(sample)}>
            <SampleMeta sample={sample} />
          </button>
        ))}
      </div>
    </div>
  );
}
