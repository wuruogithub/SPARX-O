"""Run CPU assertions even when pytest is not installed."""
from pathlib import Path
import runpy
import torch

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
for filename in ("test_model.py", "test_training.py"):
    namespace = runpy.run_path(str(root / "tests" / filename))
    for name, function in sorted(namespace.items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}", flush=True)
