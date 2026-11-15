import os
from huggingface_hub import HfApi
from datasets import load_dataset

api = HfApi()

models_to_check = [
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "tiiuae/Falcon3-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "GSAI-ML/LLaDA-8B-Instruct",
    "answerdotai/ModernBERT-large",
    "microsoft/deberta-v3-large",
    "desklib/ai-text-detector-v1.01",
    "TrustSafeAI/RADAR-Vicuna-7B",
]

print("=== CHECKING MODELS ===")
for m in models_to_check:
    try:
        info = api.model_info(m)
        print(f"OK: {m} | gated={info.gated} | pipeline={info.pipeline_tag}")
    except Exception as e:
        print(f"ERR: {m} -> {e}")

print("\n=== CHECKING DATASETS ===")
ds_to_check = [
    ("HuggingFaceH4/no_robots", "train"),
    ("toloka/beemo", "train"),
    ("ONE-Lab/MixSet", "train"),
    ("yaful/MAGE", "test"),
]
for name, split in ds_to_check:
    try:
        ds = load_dataset(name, split=split, streaming=True)
        row = next(iter(ds))
        print(f"OK: {name} [{split}] | keys={list(row.keys())}")
    except Exception as e:
        print(f"ERR: {name} -> {e}")

try:
    ds_raid = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
    row = next(iter(ds_raid))
    print(f"OK: liamdugan/raid [train] | keys={list(row.keys())}")
except Exception as e:
    print(f"ERR: liamdugan/raid -> {e}")
