# LLM Screening JSON

The LLM response must fit this shape:

```json
{
  "label": "positive",
  "specific_affect": "sadness",
  "is_clear": true,
  "confidence": 0.87,
  "evidence": "short phrase",
  "quality_flags": []
}
```

For `M12`, also include:

```json
{
  "sample_type_suggestion": "Conflict",
  "dominant_modality_suggestion": "M2",
  "needs_human_review": true
}
```
