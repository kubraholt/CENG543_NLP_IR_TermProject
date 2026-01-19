# 02_bm25_baseline_improved.py
import json
import os
import subprocess
import numpy as np
from datetime import datetime
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search.lucene import LuceneSearcher

def create_pyserini_corpus(corpus, output_dir):
    """Convert BEIR corpus to Pyserini JSONL format with consistent ordering"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "corpus.jsonl")
    
    # Reproducibility için sort et
    sorted_corpus = sorted(corpus.items())
    
    print(f"   Creating corpus file with {len(corpus):,} documents...")
    with open(output_file, "w") as f:
        for doc_id, doc in sorted_corpus:
            record = {
                "id": doc_id,
                "contents": f"{doc.get('title', '')} {doc.get('text', '')}"
            }
            f.write(json.dumps(record) + "\n")
    
    return output_file

def build_index(corpus_dir, index_dir):
    """Build Lucene index using Pyserini with error handling"""
    try:
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", corpus_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "4",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   Index built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Index build failed!")
        print(f"   Error: {e.stderr}")
        raise

def analyze_dataset(corpus, queries, qrels):
    """Calculate dataset statistics"""
    stats = {
        "num_queries": len(queries),
        "num_docs": len(corpus),
        "num_qrels": sum(len(rels) for rels in qrels.values()),
        "avg_query_length_words": np.mean([len(q.split()) for q in queries.values()]),
        "avg_doc_length_words": np.mean([len(d.get('text', '').split()) for d in corpus.values()]),
        "avg_doc_length_chars": np.mean([len(d.get('title', '') + ' ' + d.get('text', '')) for d in corpus.values()]),
        "avg_relevant_per_query": np.mean([len(rels) for rels in qrels.values()])
    }
    return stats

def run_bm25_retrieval(dataset_name, k=100, bm25_params=(0.9, 0.4)):
    """Run BM25 retrieval for a dataset with comprehensive evaluation"""
    k1, b = bm25_params
    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name.upper()}")
    print(f"   BM25 params: k1={k1}, b={b}")
    print(f"   Top-K: {k}")
    print(f"{'='*60}")

    # Load dataset
    data_path = f"./datasets/{dataset_name}"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"Loaded -> Corpus: {len(corpus):,} | Queries: {len(queries):,}")

    # Analyze dataset
    print("Analyzing dataset statistics...")
    dataset_stats = analyze_dataset(corpus, queries, qrels)
    print(f"   Avg query length: {dataset_stats['avg_query_length_words']:.1f} words")
    print(f"   Avg doc length: {dataset_stats['avg_doc_length_words']:.1f} words ({dataset_stats['avg_doc_length_chars']:.0f} chars)")
    print(f"   Avg relevant docs per query: {dataset_stats['avg_relevant_per_query']:.1f}")
    
    # Prepare directories
    corpus_dir = f"./indices/{dataset_name}/corpus"
    index_dir = f"./indices/{dataset_name}/index"
    os.makedirs("./results", exist_ok=True)
    
    # Create corpus and build index (if not exists)
    if not os.path.exists(index_dir):
        print("Creating corpus file...")
        create_pyserini_corpus(corpus, corpus_dir)

        print("Building index (this may take a while for NQ)...")
        build_index(corpus_dir, index_dir)
    else:
        print("Index already exists, skipping build...")

    # Search
    print(f"Running BM25 search (Top-{k})...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=k1, b=b)
    
    results = {}
    query_stats = {}  # Per-query statistics
    
    for i, (qid, query_text) in enumerate(queries.items(), 1):
        if i % 100 == 0:
            print(f"   Processed {i}/{len(queries)} queries...")
        
        hits = searcher.search(query_text, k=k)
        results[qid] = {hit.docid: float(hit.score) for hit in hits}
        
        # Track per-query stats
        query_stats[qid] = {
            "num_retrieved": len(hits),
            "top_score": float(hits[0].score) if hits else 0.0,
            "query_length": len(query_text.split())
        }
    
    # Save raw results
    output_file = f"./results/{dataset_name}_bm25_top{k}_k1{k1}_b{b}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_file}")

    # Evaluate with extended metrics
    print(f"\nBM25 Performance on {dataset_name.upper()}:")
    evaluator = EvaluateRetrieval()

    # EXTENDED METRICS: Now includes Recall@20 and @50
    k_values = [1, 3, 5, 10, 20, 50, 100]
    ndcg, map_score, recall, precision = evaluator.evaluate(qrels, results, k_values)
    
    # Print key metrics
    print(f"   NDCG@10:    {ndcg['NDCG@10']:.4f}")
    print(f"   NDCG@20:    {ndcg['NDCG@20']:.4f}")
    print(f"   Recall@20:  {recall['Recall@20']:.4f}")
    print(f"   Recall@100: {recall['Recall@100']:.4f}")
    print(f"   P@3:        {precision['P@3']:.4f}")
    print(f"   P@10:       {precision['P@10']:.4f}")
    print(f"   MAP:        {map_score['MAP@100']:.4f}")
    
    # Per-query performance analysis
    query_performance = analyze_query_performance(qrels, results, queries)
    
    return {
        "dataset": dataset_name,
        "bm25_params": {"k1": k1, "b": b},
        "dataset_stats": dataset_stats,
        "metrics": {
            "ndcg@10": ndcg['NDCG@10'],
            "ndcg@20": ndcg['NDCG@20'],
            "ndcg@100": ndcg['NDCG@100'],
            "recall@20": recall['Recall@20'],
            "recall@50": recall['Recall@50'],
            "recall@100": recall['Recall@100'],
            "p@3": precision['P@3'],
            "p@10": precision['P@10'],
            "map@100": map_score['MAP@100']
        },
        "query_performance": query_performance
    }

def analyze_query_performance(qrels, results, queries):
    """Analyze per-query performance to identify hard/easy queries"""
    from beir.retrieval.evaluation import EvaluateRetrieval
    
    evaluator = EvaluateRetrieval()
    query_ndcg = {}
    
    for qid in qrels.keys():
        if qid in results:
            single_qrel = {qid: qrels[qid]}
            single_result = {qid: results[qid]}
            ndcg, _, _, _ = evaluator.evaluate(single_qrel, single_result, [10])
            query_ndcg[qid] = ndcg['NDCG@10']
        else:
            query_ndcg[qid] = 0.0
    
    # Sort by performance
    sorted_queries = sorted(query_ndcg.items(), key=lambda x: x[1])
    
    return {
        "best_queries": [
            {"qid": qid, "ndcg@10": score, "query": queries.get(qid, "")[:100]} 
            for qid, score in sorted_queries[-10:]
        ],
        "worst_queries": [
            {"qid": qid, "ndcg@10": score, "query": queries.get(qid, "")[:100]} 
            for qid, score in sorted_queries[:10]
        ],
        "avg_ndcg@10": np.mean(list(query_ndcg.values())),
        "std_ndcg@10": np.std(list(query_ndcg.values()))
    }

def save_comprehensive_report(all_results, bm25_configs):
    """Save detailed analysis report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON format (for programmatic access)
    summary = {
        "experiment": "BM25 Baseline - Comprehensive Evaluation",
        "date": datetime.now().isoformat(),
        "configurations_tested": bm25_configs,
        "results": all_results
    }
    
    json_file = f"./results/bm25_comprehensive_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Readable text report
    txt_file = f"./results/bm25_comprehensive_{timestamp}.txt"
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BM25 BASELINE - COMPREHENSIVE EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset statistics
        f.write("DATASET STATISTICS\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            stats = r['dataset_stats']
            f.write(f"\n{r['dataset'].upper()}:\n")
            f.write(f"  Documents: {stats['num_docs']:,}\n")
            f.write(f"  Queries: {stats['num_queries']:,}\n")
            f.write(f"  Avg query length: {stats['avg_query_length_words']:.1f} words\n")
            f.write(f"  Avg doc length: {stats['avg_doc_length_words']:.1f} words ({stats['avg_doc_length_chars']:.0f} chars)\n")
            f.write(f"  Avg relevant docs per query: {stats['avg_relevant_per_query']:.1f}\n")

        # Performance metrics
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        
        # Table header
        f.write(f"\n{'Dataset':<12} | {'NDCG@10':>8} | {'NDCG@20':>8} | {'Recall@20':>10} | {'Recall@100':>11} | {'MAP':>8}\n")
        f.write("-" * 80 + "\n")
        
        for r in all_results:
            m = r['metrics']
            f.write(f"{r['dataset'].upper():<12} | {m['ndcg@10']:>8.4f} | {m['ndcg@20']:>8.4f} | "
                   f"{m['recall@20']:>10.4f} | {m['recall@100']:>11.4f} | {m['map@100']:>8.4f}\n")
        
        # Query difficulty analysis
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("QUERY DIFFICULTY ANALYSIS\n")
        f.write("-" * 80 + "\n")

        for r in all_results:
            qperf = r['query_performance']
            f.write(f"\n{r['dataset'].upper()}:\n")
            f.write(f"  Average NDCG@10: {qperf['avg_ndcg@10']:.4f} ± {qperf['std_ndcg@10']:.4f}\n")

            f.write(f"\n  Top 3 Best Queries:\n")
            for i, q in enumerate(qperf['best_queries'][-3:], 1):
                f.write(f"    {i}. NDCG@10={q['ndcg@10']:.4f} | {q['query']}\n")

            f.write(f"\n  Top 3 Worst Queries:\n")
            for i, q in enumerate(qperf['worst_queries'][:3], 1):
                f.write(f"    {i}. NDCG@10={q['ndcg@10']:.4f} | {q['query']}\n")
        
        f.write("\n" + "=" * 80 + "\n")

    print(f"\nComprehensive reports saved:")
    print(f"   - {json_file}")
    print(f"   - {txt_file}")
    
    return json_file, txt_file

if __name__ == "__main__":
    import os
    
    # Set Java environment
    os.environ["JAVA_HOME"] = os.popen("/usr/libexec/java_home").read().strip()

    # BM25 Parameter Configurations to Test
    bm25_configs = [
        (0.9, 0.4),   # Your current config
        (0.82, 0.68), # MS MARCO tuned
        (1.2, 0.75),  # Robertson's defaults
    ]

    # Run experiments
    print("\n" + "="*60)
    print("STARTING BM25 COMPREHENSIVE EVALUATION")
    print(f"   Testing {len(bm25_configs)} BM25 configurations")
    print(f"   Datasets: scifact, arguana, nq")
    print("="*60)
    
    all_results = []
    
    for k1, b in bm25_configs:
        print(f"\n\n{'#'*60}")
        print(f"BM25 CONFIG: k1={k1}, b={b}")
        print(f"{'#'*60}")

        for dataset in ["scifact", "arguana", "nq"]:
            try:
                result = run_bm25_retrieval(dataset, k=100, bm25_params=(k1, b))
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {dataset} with k1={k1}, b={b}: {e}")
                import traceback
                traceback.print_exc()

    # Save comprehensive report
    save_comprehensive_report(all_results, bm25_configs)

    print("\n" + "="*60)
    print("BM25 BASELINE EVALUATION COMPLETE!")
    print("="*60)