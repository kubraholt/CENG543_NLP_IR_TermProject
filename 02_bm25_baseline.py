# 02_bm25_baseline.py
import json
import os
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search.lucene import LuceneSearcher
import subprocess

def create_pyserini_corpus(corpus, output_dir):
    """Convert BEIR corpus to Pyserini JSONL format"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "corpus.jsonl")
    
    with open(output_file, "w") as f:
        for doc_id, doc in corpus.items():
            record = {
                "id": doc_id,
                "contents": f"{doc.get('title', '')} {doc.get('text', '')}"
            }
            f.write(json.dumps(record) + "\n")
    
    return output_file

def build_index(corpus_dir, index_dir):
    """Build Lucene index using Pyserini"""
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(cmd, check=True)

def run_bm25_retrieval(dataset_name, k=100):
    """Run BM25 retrieval for a dataset"""
    print(f"\n{'='*50}")
    print(f"🚀 PROCESSING: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    # Load dataset
    data_path = f"./datasets/{dataset_name}"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"✅ Loaded -> Corpus: {len(corpus):,} | Queries: {len(queries):,}")
    
    # Prepare directories
    corpus_dir = f"./indices/{dataset_name}/corpus"
    index_dir = f"./indices/{dataset_name}/index"
    os.makedirs("./results", exist_ok=True)
    
    # Create corpus and build index (if not exists)
    if not os.path.exists(index_dir):
        print("📝 Creating corpus file...")
        create_pyserini_corpus(corpus, corpus_dir)
        
        print("🔨 Building index (this may take a while for NQ)...")
        build_index(corpus_dir, index_dir)
    else:
        print("✅ Index already exists, skipping build...")
    
    # Search
    print(f"🔍 Running BM25 search (Top-{k})...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)  # Standard BM25 parameters
    
    results = {}
    for qid, query_text in queries.items():
        hits = searcher.search(query_text, k=k)
        results[qid] = {hit.docid: float(hit.score) for hit in hits}
    
    # Save results
    output_file = f"./results/{dataset_name}_bm25_top{k}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved: {output_file}")
    
    # Evaluate
    print(f"\n📊 BM25 Performance on {dataset_name.upper()}:")
    evaluator = EvaluateRetrieval()
    ndcg, map_score, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])
    
    print(f"   NDCG@10:    {ndcg['NDCG@10']:.4f}")
    print(f"   NDCG@100:   {ndcg['NDCG@100']:.4f}")
    print(f"   Recall@100: {recall['Recall@100']:.4f}")
    print(f"   P@3:        {precision['P@3']:.4f}")
    
    return {
        "dataset": dataset_name,
        "ndcg@10": ndcg['NDCG@10'],
        "ndcg@100": ndcg['NDCG@100'],
        "recall@100": recall['Recall@100'],
        "p@3": precision['P@3']
    }

if __name__ == "__main__":
    import os
    import json
    from datetime import datetime
    
    # Set Java environment
    os.environ["JAVA_HOME"] = os.popen("/usr/libexec/java_home").read().strip()
    
    # Run for all datasets
    all_results = []
    
    for dataset in ["scifact", "arguana", "nq"]:
        try:
            result = run_bm25_retrieval(dataset, k=100)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
    
    # Save summary to JSON
    summary = {
        "experiment": "BM25 Baseline",
        "date": datetime.now().isoformat(),
        "results": all_results
    }
    
    with open("./results/bm25_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save summary to readable text file
    with open("./results/bm25_baseline_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BM25 BASELINE RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for r in all_results:
            f.write(f"Dataset: {r['dataset'].upper()}\n")
            f.write(f"  NDCG@10:    {r['ndcg@10']:.4f}\n")
            f.write(f"  NDCG@100:   {r['ndcg@100']:.4f}\n")
            f.write(f"  Recall@100: {r['recall@100']:.4f}\n")
            f.write(f"  P@3:        {r['p@3']:.4f}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Dataset':<12} | {'NDCG@10':>8} | {'P@3':>8} | {'Recall@100':>10}\n")
        f.write("-" * 50 + "\n")
        for r in all_results:
            f.write(f"{r['dataset'].upper():<12} | {r['ndcg@10']:>8.4f} | {r['p@3']:>8.4f} | {r['recall@100']:>10.4f}\n")
    
    print("\n✅ Results saved to:")
    print("   - ./results/bm25_baseline_summary.json")
    print("   - ./results/bm25_baseline_summary.txt")