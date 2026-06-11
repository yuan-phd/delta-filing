"""ONE-TIME: build a Kaggle dataset that holds Qwen3.5-4B, so the matrix kernels
read the base model from the mount with ZERO downloads.

Run this in a Kaggle NOTEBOOK with: Accelerator = None/CPU, Internet = ON.
Then: Save Version (Save & Run All) -> when it finishes, on the output use
"Create Dataset" / "New Dataset from Notebook Output". Name it e.g. qwen3-5-4b.

The model lands in /kaggle/working/Qwen3.5-4B, so in the dataset it is at
/kaggle/input/<slug>/Qwen3.5-4B/ and train_pref.resolve_model_source auto-detects
it (normalized name match), no DELTA_MODEL_DIR needed.
"""
import os

from huggingface_hub import snapshot_download

OUT = "/kaggle/working/Qwen3.5-4B"
path = snapshot_download(
    "Qwen/Qwen3.5-4B",
    local_dir=OUT,
    ignore_patterns=["*.pth", "*.gguf", "original/*", "*.bin.index.json.tmp"],
)
print("downloaded to:", path)
print("contents:")
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f)) / 1e6
    print(f"  {f:<40} {sz:8.1f} MB")
assert os.path.exists(os.path.join(OUT, "config.json")), "config.json missing!"
print("\nOK -> Save Version, then create a dataset from this notebook's output.")
