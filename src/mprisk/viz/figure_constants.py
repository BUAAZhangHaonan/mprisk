"""Locked constants for the final ten-figure bundle."""

FIGURE_SCHEMA = "mprisk_bundle_figure_map_v1"
STATUS_READY = "Ready"
STATUS_PENDING = "Pending"
LOCKED_TERMS = {
    "conflict": "Conflict",
    "aligned": "Aligned",
    "misread": "Misread",
    "non_misread": "Non-misread",
    "vision_lean": "V lean",
    "text_audio_lean": "T/A lean",
}
FORBIDDEN_PDF_TEXT = (
    "illustrative",
    "placeholder",
    "[xx]",
    "wrong-answer",
    "state consistency",
    "divergence",
    "arbitration",
)
CONCEPTUAL_KEYS = {
    "fig01_problem_protocol",
    "fig02_representation_pipeline",
    "fig03_spherical_sdr",
    "figB1_representation_details",
}
MODEL_SPECS = (
    ("qwen2_5_omni_7b", "Qwen2.5-Omni-7B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
    ("internvl3_5_8b", "InternVL3.5-8B"),
)
MODEL_LABELS = tuple(label for _, label in MODEL_SPECS)
UMAP_CONFIG = {
    "random_state": 20260716,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine",
}
FULL_MODEL_LABELS = (
    "Gemma-3-4B",
    "Gemma-3-12B",
    "GLM-4.6V-Flash",
    "InternVL3.5-8B",
    "LLaVA-v1.5-7B",
    "LLaVA-OneVision-7B",
    "MiniCPM-V-2.6",
    "MiniCPM-V-4.5",
    "Phi-3.5-Vision",
    "Qwen2.5-VL-7B",
    "Qwen3-VL-8B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Gemma-4-12B",
    "Phi-4-Multimodal",
    "Qwen2.5-Omni-7B",
)
