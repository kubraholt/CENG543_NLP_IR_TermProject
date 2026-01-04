import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

# 1. Download datasets
datasets = ["nq", "scifact", "arguana"]
data_dir = "/Users/kubraholt/CENG543_NLP_IR_TermProject/datasets"

for dataset in datasets:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, data_dir)
    print(f"SUCCESS: {dataset} downloaded: {data_path}")