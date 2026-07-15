# Prefill smoke manifest v1

Each model manifest freezes one `Conflict` and one `Aligned` `use_in_main` row
from the matching protocol manifest. The batch runner expands the two rows over
three conditions and the fixed P=8 prompt set, yielding exactly 48 cache tasks
per model. `Misread` is not an extraction input.
