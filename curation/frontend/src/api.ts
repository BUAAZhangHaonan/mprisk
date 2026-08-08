export type ViewSuggestion = {
  label: string;
  specific_affect: string;
  is_clear: boolean;
  confidence: number;
  evidence: string;
  quality_flags: string[];
};

export type LlmScreening = {
  sample_id: string;
  view_outputs: Record<"M1" | "M2" | "M12", ViewSuggestion>;
  sample_type_suggestion: string;
  dominant_modality_suggestion: string;
  quality_flags: string[];
};

export type Sample = {
  sample_id: string;
  source_dataset: string;
  source_id?: string;
  protocol: string;
  m1_modality?: string;
  m2_modality?: string;
  candidate_type: string;
  media_paths?: Record<string, string>;
  text_content?: string;
  llm_sample_type_suggestion?: string;
  llm_screening?: LlmScreening | null;
  llm_agrees?: boolean;
  review_priority?: string;
  annotation_count?: number;
  annotators?: string[];
  human_types?: string[];
  annotations?: AnnotationPayload[];
  source_metadata?: Record<string, string>;
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

export type ProgressGroup = {
  protocol: string;
  candidate_type: string;
  total: number;
  annotated_once: number;
  annotated_twice: number;
};

export type Progress = {
  groups: ProgressGroup[];
  total_annotations: number;
  annotators: { annotator_id: string; count: number }[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function mediaUrl(pathOrUrl: string, audioOnly = false): string {
  if (/^(?:https?:|data:|blob:)/i.test(pathOrUrl)) {
    return pathOrUrl;
  }
  const assetPath = pathOrUrl.split("/").map(encodeURIComponent).join("/");
  const audioSuffix = audioOnly ? "?audio=true" : "";
  return `${API_BASE}/media/${assetPath}${audioSuffix}`;
}

export type SampleFilters = {
  candidateType?: string;
  llmType?: string;
  humanType?: string;
  protocol?: string;
  excludeAnnotator?: string;
  onlyAnnotator?: string;
  disagreementOnly?: boolean;
  page?: number;
  pageSize?: number;
};

export type PaginatedSamples = {
  items: Sample[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
};

export async function fetchSamples(filters: SampleFilters = {}): Promise<PaginatedSamples> {
  const params = new URLSearchParams();
  if (filters.candidateType) params.set("candidate_type", filters.candidateType);
  if (filters.llmType) params.set("llm_type", filters.llmType);
  if (filters.humanType) params.set("human_type", filters.humanType);
  if (filters.protocol) params.set("protocol", filters.protocol);
  if (filters.excludeAnnotator) params.set("exclude_annotator", filters.excludeAnnotator);
  if (filters.onlyAnnotator) params.set("only_annotator", filters.onlyAnnotator);
  if (filters.disagreementOnly) params.set("disagreement_only", "true");
  params.set("page", String(filters.page ?? 1));
  params.set("page_size", String(filters.pageSize ?? 50));
  const response = await fetch(`${API_BASE}/samples?${params.toString()}`);
  const data = await response.json();
  return {
    items: data.items,
    page: data.page,
    page_size: data.page_size,
    total: data.total,
    total_pages: data.total_pages,
  };
}

export async function fetchSample(sampleId: string): Promise<Sample> {
  const response = await fetch(`${API_BASE}/samples/${encodeURIComponent(sampleId)}`);
  return response.json();
}

export async function fetchProgress(): Promise<Progress> {
  const response = await fetch(`${API_BASE}/samples/progress`);
  return response.json();
}

export async function saveAnnotation(payload: AnnotationPayload): Promise<void> {
  const response = await fetch(`${API_BASE}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`save failed: ${response.status}`);
  }
}
