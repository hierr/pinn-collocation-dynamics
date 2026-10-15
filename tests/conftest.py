"""Select noninteractive plotting and the PyTorch backend for validation tests."""

import os

os.environ["DDE_BACKEND"] = "pytorch"
os.environ.setdefault("MPLBACKEND", "Agg")
