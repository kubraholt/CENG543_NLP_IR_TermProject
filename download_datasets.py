# download_datasets.py
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

datasets = ["scifact", "arguana", "nq"]
data_dir = "./datasets"

os.makedirs(data_dir, exist_ok=True)

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Downloading {dataset.upper()}...")
    print(f"{'='*60}")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    try:
        data_path = util.download_and_unzip(url, data_dir)
        print(f"SUCCESS: {dataset} -> {data_path}")

        # Verify
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        print(f"   Verified: {len(queries)} queries, {len(corpus)} docs")

    except Exception as e:
        print(f"ERROR: {dataset} - {e}")

print("\nAll datasets downloaded!")