export type Sample = {
  sample_id: string;
  source_dataset: string;
  protocol: string;
  candidate_type: string;
  media_paths?: Record<string, string>;
};

export type AnnotationPayload = {
  sample_id: string;
  annotator_id: string;
  m1_label: string;
  m2_label: string;
  joint_label: string;
  m1_specific_affect: string;
  m2_specific_affect: string;
  joint_specific_affect: string;
  m1_is_clear: boolean;
  m2_is_clear: boolean;
  joint_is_clear: boolean;
  m1_confidence: number;
  m2_confidence: number;
  joint_confidence: number;
  sample_type: string;
  dominant_modality: string;
  quality_flags: string[];
  notes: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchSamples(candidateType = ""): Promise<Sample[]> {
  const suffix = candidateType ? `?candidate_type=${encodeURIComponent(candidateType)}` : "";
  const response = await fetch(`${API_BASE}/samples${suffix}`);
  const data = await response.json();
  return data.items;
}

export async function fetchSample(sampleId: string): Promise<Sample> {
  const response = await fetch(`${API_BASE}/samples/${encodeURIComponent(sampleId)}`);
  return response.json();
}

export async function saveAnnotation(payload: AnnotationPayload): Promise<void> {
  await fetch(`${API_BASE}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
