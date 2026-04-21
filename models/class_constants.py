"""Torch-free class vocabularies for the 3-layer pipeline.

Data-prep scripts (ingestors, validators) import from here instead of
``models.layer{1,2,3}_*`` so they don't pull in torch/timm just to
read a list of strings.
"""

from __future__ import annotations

DEFAULT_PART_CLASSES: list[str] = [
    "bumper",
    "hood",
    "fender",
    "door",
    "windshield",
    "headlight",
    "taillight",
    "mirror",
    "trunk",
    "roof",
    "quarter_panel",
    "grille",
    "wheel",
]

DEFAULT_DAMAGE_CLASSES: list[str] = [
    "dent",
    "scratch",
    "crack",
    "shatter",
    "tear",
    "deformation",
    "paint_loss",
    "puncture",
    "misalignment",
]

# V2 vocabulary: prepends the explicit "no_damage" class so the multi-label
# classifier can learn what *undamaged* looks like and stops false-positiving
# damaged labels on clean parts. The existing 9-class L2 keeps working;
# deployments that retrain on v2 data will pick up this list.
NO_DAMAGE_CLASS = "no_damage"
DEFAULT_DAMAGE_CLASSES_V2: list[str] = [NO_DAMAGE_CLASS, *DEFAULT_DAMAGE_CLASSES]

DEFAULT_SEVERITY_GRADES: list[str] = ["minor", "moderate", "severe", "total_loss"]
