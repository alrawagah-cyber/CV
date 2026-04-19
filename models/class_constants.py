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

DEFAULT_SEVERITY_GRADES: list[str] = ["minor", "moderate", "severe", "total_loss"]
