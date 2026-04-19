"""Model architectures for the 3-layer car damage assessment pipeline.

Submodules are intentionally NOT eagerly imported here. The torch-heavy
modules (``heads``, ``layer2_damage``, ``layer3_severity``, ``registry``)
only load when their specific paths are imported. This lets data-prep
scripts use ``models.class_constants`` without installing torch.
"""
