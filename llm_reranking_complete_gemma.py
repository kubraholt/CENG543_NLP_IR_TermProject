# llm_rerank_gemma2.py
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from collections import defaultdict
import numpy as np
from scipy import stats

# Dataset configs
DATASET_CONFIG = {
    "scifact": {"best_bm25_config": "k10.82_b0.68"},
    "arguana": {"best_bm25_config": "k11.2_b0.75"},
    "nq": {"best_bm25_config": "k10.9_b0.4"},
}

# Model config
HF_MODEL_ID = "google/gemma-2-9b-it"

# Global model cache
_MODEL = None
_TOKENIZER = None

def load_model_once():
    """Load model once and cache"""
    global _MODEL, _TOKENIZER
    
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER
    
    print(f"📦 Loading HF model: {HF_MODEL_ID}")
    _TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    
    _MODEL = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    _MODEL.eval()
    print(f"✅ Model loaded on device: {next(_MODEL.parameters()).device}")
    
    return _MODEL, _TOKENIZER


def create_prompt(query: str, doc_text: str, dataset_name: str) -> str:
    """Dataset-specific zero-shot prompts"""
    
    if dataset_name == "scifact":
        return f"""You are a scientific evidence assessor.

Claim:
{query}

Document:
{doc_text}

Task: Rate how well the document provides scientific evidence for the claim.

Return EXACTLY one integer 0-10 (only the number). Use these anchors:
0: Completely unrelated
1-3: Mentions related concepts but no direct evidence
4-6: Partially relevant; provides some supporting context
7-9: Strongly relevant; provides supporting evidence
10: Direct, conclusive evidence for the claim

Output only one integer (0-10), nothing else."""

    elif dataset_name == "nq":
        return f"""You are a question-answering relevance assessor.

Question:
{query}

Document:
{doc_text}

Task: Rate how well the document answers the question.

Return EXACTLY one integer 0-10 (only the number). Use these anchors:
0: Completely unrelated
1-3: Mentions related topics but does not answer
4-6: Partially answers; incomplete
7-9: Mostly answers clearly
10: Fully and directly answers with complete information

Output only one integer (0-10), nothing else."""

    elif dataset_name == "arguana":
        return f"""You are an argument analyst.

Argument:
{query}

Document:
{doc_text}

Task: Rate how strongly the document ARGUES AGAINST or REFUTES the given argument.

Important explicit rule: If the document AGREES with the argument (i.e., it supports the same position) you MUST return 0.

Return EXACTLY one integer 0-10 (only the number). Use these anchors:
0: Supports the argument OR is unrelated
1-3: Mentions the topic but does not clearly oppose
4-6: Presents some opposing points / partial refutation
7-9: Clearly argues against with reasoning or evidence
10: Strong, direct refutation with counter-evidence or decisive reasoning

Output only one integer (0-10), nothing else."""

def parse_score(response: str) -> float:
    """Parse numeric score from LLM response"""
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', response.strip())
    if match:
        score = float(match.group(1))
        return max(0.0, min(10.0, score))
    return 5.0

def call_llm(prompt: str) -> float:
    """Call HF Gemma-2 and parse score"""
    try:
        model, tokenizer = load_model_once()
        
        # Gemma-2 chat template format
        messages = [
            {"role": "user", "content": f"Return only one integer from 0 to 10. Do not output any other text.\n\n{prompt}"},
        ]
        
        # Encode with chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=8192,  # Gemma-2 has 8k context
        ).to(model.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,  # temperature=0.0
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            )
        
        # Decode only new tokens
        prompt_len = input_ids.shape[1]
        new_tokens = outputs[0, prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # DEBUG: Print first 3 calls
        if not hasattr(call_llm, 'call_count'):
            call_llm.call_count = 0
        call_llm.call_count += 1
        if call_llm.call_count <= 3:
            print(f"\n[LLM DEBUG {call_llm.call_count}]")
            print(f"Response: '{response}'")
            print(f"Parsed score: {parse_score(response)}")
        
        return parse_score(response)
        
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return 5.0

def load_bm25(dataset_name: str) -> dict:
    """Load best BM25 results"""
    config = DATASET_CONFIG[dataset_name]["best_bm25_config"]
    path = f"./results/bm25_baseline_improved_results/{dataset_name}_bm25_top100_{config}.json"
    with open(path) as f:
        return json.load(f)

def normalize_scores(scores: dict, min_val: float = 0.0, max_val: float = 1.0) -> dict:
    """Normalize scores to [min_val, max_val] range"""
    if not scores:
        return scores
    
    vals = list(scores.values())
    min_score = min(vals)
    max_score = max(vals)
    
    if max_score == min_score:
        return {k: max_val for k in scores}
    
    normalized = {}
    for doc_id, score in scores.items():
        norm_score = (score - min_score) / (max_score - min_score)
        normalized[doc_id] = min_val + norm_score * (max_val - min_val)
    
    return normalized

def calculate_improvement_metrics(bm25_metrics: dict, llm_metrics: dict) -> dict:
    """Calculate improvement metrics (Δ values)"""
    improvements = {}
    
    for metric_name in bm25_metrics:
        if metric_name in llm_metrics:
            bm25_val = bm25_metrics[metric_name]
            llm_val = llm_metrics[metric_name]
            
            delta = llm_val - bm25_val
            if bm25_val != 0:
                delta_pct = (delta / bm25_val) * 100
            else:
                delta_pct = 0.0
            
            improvements[f"Δ{metric_name}"] = delta
            improvements[f"Δ{metric_name}_pct"] = delta_pct
    
    return improvements

def compute_query_level_metrics(qrels: dict, results: dict, k_values: list) -> dict:
    """Compute per-query NDCG, Recall, Precision"""
    evaluator = EvaluateRetrieval()
    query_metrics = defaultdict(dict)
    
    for qid in qrels:
        if qid not in results:
            continue
        
        single_qrel = {qid: qrels[qid]}
        single_result = {qid: results[qid]}
        
        ndcg, map_score, recall, precision = evaluator.evaluate(
            single_qrel, single_result, k_values
        )
        
        query_metrics[qid] = {
            **{f"NDCG@{k}": ndcg[f"NDCG@{k}"] for k in k_values},
            **{f"Recall@{k}": recall[f"Recall@{k}"] for k in k_values},
            **{f"P@{k}": precision[f"P@{k}"] for k in k_values},
            **{f"MAP@{k}": map_score[f"MAP@{k}"] for k in k_values},
        }
    
    return dict(query_metrics)

def rerank(dataset_name: str, num_queries: int = 300):
    """Re-rank with LLM and generate comprehensive reports"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()} | Model: Gemma-2-9B-IT | Queries: {num_queries}")
    print(f"{'='*80}")
    
    # Load model first
    load_model_once()
    
    # Load data
    corpus, queries, qrels = GenericDataLoader(f"./datasets/{dataset_name}").load(split="test")
    bm25_results = load_bm25(dataset_name)
    query_ids = list(bm25_results.keys())[:num_queries]
    
    # Prepare storage
    llm_results = {}
    detailed_records = {}
    
    # Re-rank
    for qid in tqdm(query_ids, desc="Re-ranking"):
        query_text = queries[qid]
        bm25_ranking = bm25_results[qid]
        top_docs = list(bm25_ranking.keys())[:20]
        
        # Get LLM scores for top 20
        llm_scores = {}
        for doc_id in top_docs:
            if doc_id in corpus:
                doc = corpus[doc_id]
                doc_text = f"{doc.get('title', '')} {doc.get('text', '')}"[:800]
                prompt = create_prompt(query_text, doc_text, dataset_name)
                llm_scores[doc_id] = call_llm(prompt)
        
        # Normalize LLM scores to match BM25 scale
        bm25_top20_scores = {doc_id: bm25_ranking[doc_id] for doc_id in top_docs if doc_id in bm25_ranking}
        if bm25_top20_scores:
            bm25_min = min(bm25_top20_scores.values())
            bm25_max = max(bm25_top20_scores.values())
            llm_scores_normalized = normalize_scores(llm_scores, bm25_min, bm25_max)
        else:
            llm_scores_normalized = llm_scores
        
        # Merge: top 20 from LLM + remaining from BM25
        remaining_docs = {doc_id: score for doc_id, score in bm25_ranking.items() if doc_id not in llm_scores_normalized}
        final_ranking = {**llm_scores_normalized, **remaining_docs}
        final_ranking = dict(sorted(final_ranking.items(), key=lambda x: x[1], reverse=True))
        
        llm_results[qid] = final_ranking
        
        # DEBUG PRINTS (only for first query in test mode)
        if num_queries <= 5 and qid == query_ids[0]:
            print(f"\n{'='*80}")
            print(f"[DEBUG] First Query Analysis:")
            print(f"{'='*80}")
            print(f"Query ID: {qid}")
            print(f"Query Text: {query_text[:100]}...")
            print(f"\nBM25 Top-3:")
            for i, (doc_id, score) in enumerate(list(bm25_ranking.items())[:3], 1):
                print(f"  {i}. {doc_id}: {score:.4f}")
            
            print(f"\nLLM Raw Scores (Top-3):")
            sorted_llm_raw = sorted(llm_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (doc_id, score) in enumerate(sorted_llm_raw, 1):
                print(f"  {i}. {doc_id}: {score:.2f}/10")
            
            print(f"\nLLM Normalized Scores (Top-3):")
            sorted_llm_norm = sorted(llm_scores_normalized.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (doc_id, score) in enumerate(sorted_llm_norm, 1):
                print(f"  {i}. {doc_id}: {score:.4f}")
            
            print(f"\nFinal Ranking (Top-3):")
            for i, (doc_id, score) in enumerate(list(final_ranking.items())[:3], 1):
                print(f"  {i}. {doc_id}: {score:.4f}")
            
            print(f"\nBM25 score range: [{bm25_min:.4f}, {bm25_max:.4f}]")
            print(f"LLM raw score range: [{min(llm_scores.values()):.2f}, {max(llm_scores.values()):.2f}]")
            
            if qid in qrels:
                relevant_docs = set(qrels[qid].keys())
                print(f"\nRelevant docs: {relevant_docs}")
                print(f"Relevant in BM25 top-3: {set(list(bm25_ranking.keys())[:3]) & relevant_docs}")
                print(f"Relevant in LLM top-3: {set(list(final_ranking.keys())[:3]) & relevant_docs}")
            print(f"{'='*80}\n")
        
        # Store detailed info
        detailed_records[qid] = {
            "query": query_text,
            "llm_raw_scores": llm_scores,
            "llm_normalized_scores": llm_scores_normalized,
            "bm25_top20_positions": {doc_id: i+1 for i, doc_id in enumerate(top_docs)},
            "llm_top20_positions": {doc_id: i+1 for i, doc_id in enumerate(sorted(llm_scores_normalized, key=llm_scores_normalized.get, reverse=True))},
            "position_changes": {
                doc_id: (list(bm25_ranking.keys()).index(doc_id) + 1) - (list(final_ranking.keys()).index(doc_id) + 1)
                for doc_id in top_docs if doc_id in final_ranking
            }
        }
    
    # Evaluate LLM results
    evaluator = EvaluateRetrieval()
    filtered_qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    
    k_values = [1, 3, 5, 10, 20, 100]
    ndcg_llm, map_llm, recall_llm, precision_llm = evaluator.evaluate(
        filtered_qrels, llm_results, k_values
    )
    
    # Evaluate BM25 results (for comparison)
    bm25_results_filtered = {qid: bm25_results[qid] for qid in query_ids}
    ndcg_bm25, map_bm25, recall_bm25, precision_bm25 = evaluator.evaluate(
        filtered_qrels, bm25_results_filtered, k_values
    )
    
    # Compute query-level metrics
    query_metrics_llm = compute_query_level_metrics(filtered_qrels, llm_results, k_values)
    query_metrics_bm25 = compute_query_level_metrics(filtered_qrels, bm25_results_filtered, k_values)
    
    # Calculate MRR
    def compute_mrr(qrels, results):
        reciprocal_ranks = []
        for qid in qrels:
            if qid not in results:
                continue
            relevant_docs = set(qrels[qid].keys())
            ranked_docs = list(results[qid].keys())
            for i, doc_id in enumerate(ranked_docs):
                if doc_id in relevant_docs:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    mrr_llm = compute_mrr(filtered_qrels, llm_results)
    mrr_bm25 = compute_mrr(filtered_qrels, bm25_results_filtered)
    
    # Aggregate metrics
    llm_metrics = {
        **{f"NDCG@{k}": ndcg_llm[f"NDCG@{k}"] for k in k_values},
        **{f"MAP@{k}": map_llm[f"MAP@{k}"] for k in k_values},
        **{f"Recall@{k}": recall_llm[f"Recall@{k}"] for k in k_values},
        **{f"P@{k}": precision_llm[f"P@{k}"] for k in k_values},
        "MRR": mrr_llm
    }
    
    bm25_metrics = {
        **{f"NDCG@{k}": ndcg_bm25[f"NDCG@{k}"] for k in k_values},
        **{f"MAP@{k}": map_bm25[f"MAP@{k}"] for k in k_values},
        **{f"Recall@{k}": recall_bm25[f"Recall@{k}"] for k in k_values},
        **{f"P@{k}": precision_bm25[f"P@{k}"] for k in k_values},
        "MRR": mrr_bm25
    }
    
    # Calculate improvements
    improvements = calculate_improvement_metrics(bm25_metrics, llm_metrics)
    
    # Query-level improvement analysis
    query_improvements = {}
    for qid in query_metrics_llm:
        if qid in query_metrics_bm25:
            query_improvements[qid] = {
                "ΔNDCG@10": query_metrics_llm[qid]["NDCG@10"] - query_metrics_bm25[qid]["NDCG@10"],
                "ΔNDCG@20": query_metrics_llm[qid]["NDCG@20"] - query_metrics_bm25[qid]["NDCG@20"],
                "ΔRecall@20": query_metrics_llm[qid]["Recall@20"] - query_metrics_bm25[qid]["Recall@20"],
            }
    
    # Win/Loss/Tie analysis
    wins = sum(1 for v in query_improvements.values() if v["ΔNDCG@10"] > 0)
    losses = sum(1 for v in query_improvements.values() if v["ΔNDCG@10"] < 0)
    ties = sum(1 for v in query_improvements.values() if v["ΔNDCG@10"] == 0)
    
    # Statistical significance
    bm25_ndcg10_values = [query_metrics_bm25[qid]["NDCG@10"] for qid in query_metrics_bm25]
    llm_ndcg10_values = [query_metrics_llm[qid]["NDCG@10"] for qid in query_metrics_llm if qid in query_metrics_bm25]
    
    if len(bm25_ndcg10_values) > 1 and len(llm_ndcg10_values) > 1:
        t_stat, p_value = stats.ttest_rel(llm_ndcg10_values, bm25_ndcg10_values)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(llm_ndcg10_values, bm25_ndcg10_values)
    else:
        t_stat, p_value, wilcoxon_stat, wilcoxon_p = None, None, None, None
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/llm_reranking/gemma2_9b/top20"
    
    # 1. Final rankings JSON
    with open(f"{output_dir}/{dataset_name}_gemma2_9b_{timestamp}.json", "w") as f:
        json.dump(llm_results, f, indent=2)
    
    # 2. Detailed records JSON
    detailed_output = {
        "metadata": {
            "dataset": dataset_name,
            "model": "gemma2_9b",
            "num_queries": num_queries,
            "bm25_config": DATASET_CONFIG[dataset_name]["best_bm25_config"],
            "timestamp": timestamp
        },
        "query_details": detailed_records,
        "query_metrics_llm": query_metrics_llm,
        "query_metrics_bm25": query_metrics_bm25,
        "query_improvements": query_improvements
    }
    
    with open(f"{output_dir}/{dataset_name}_gemma2_9b_detailed_{timestamp}.json", "w") as f:
        json.dump(detailed_output, f, indent=2)
    
    # 3. Human-readable report TXT
    report_lines = [
        "="*80,
        f"LLM RE-RANKING EVALUATION REPORT",
        f"Dataset: {dataset_name.upper()}",
        f"Model: Gemma-2-9B-IT (Google)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        "",
        "📊 CONFIGURATION",
        "-"*80,
        f"Number of queries: {num_queries}",
        f"BM25 baseline config: {DATASET_CONFIG[dataset_name]['best_bm25_config']}",
        f"Re-ranking strategy: Top-20 documents",
        f"LLM temperature: 0.0",
        "",
        "="*80,
        "📈 OVERALL METRICS COMPARISON",
        "-"*80,
        f"{'Metric':<20} | {'BM25':<12} | {'LLM':<12} | {'Δ (abs)':<12} | {'Δ (%)':<12}",
        "-"*80,
    ]
    
    for metric in ["NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20", "Recall@10", "Recall@20", "Recall@100", "P@3", "P@10", "MRR"]:
        bm25_val = bm25_metrics.get(metric, 0.0)
        llm_val = llm_metrics.get(metric, 0.0)
        delta = improvements.get(f"Δ{metric}", 0.0)
        delta_pct = improvements.get(f"Δ{metric}_pct", 0.0)
        
        report_lines.append(
            f"{metric:<20} | {bm25_val:<12.4f} | {llm_val:<12.4f} | {delta:>+12.4f} | {delta_pct:>+11.2f}%"
        )
    
    report_lines.extend([
        "",
        "="*80,
        "🔍 QUERY-LEVEL ANALYSIS",
        "-"*80,
        f"Queries improved (NDCG@10): {wins} ({wins/len(query_improvements)*100:.1f}%)",
        f"Queries degraded (NDCG@10): {losses} ({losses/len(query_improvements)*100:.1f}%)",
        f"Queries unchanged (NDCG@10): {ties} ({ties/len(query_improvements)*100:.1f}%)",
        "",
        f"Average ΔNDCG@10: {np.mean([v['ΔNDCG@10'] for v in query_improvements.values()]):.4f} ± {np.std([v['ΔNDCG@10'] for v in query_improvements.values()]):.4f}",
        "",
    ])
    
    # Top 5 best improvements
    sorted_improvements = sorted(query_improvements.items(), key=lambda x: x[1]["ΔNDCG@10"], reverse=True)
    report_lines.extend([
        "🏆 TOP 5 BEST IMPROVEMENTS (ΔNDCG@10):",
        "-"*80,
    ])
    for i, (qid, metrics) in enumerate(sorted_improvements[:5], 1):
        query_text = queries[qid][:100] + "..." if len(queries[qid]) > 100 else queries[qid]
        report_lines.append(f"{i}. ΔNDCG@10={metrics['ΔNDCG@10']:+.4f} | {query_text}")
    
    report_lines.append("")
    
    # Top 5 worst degradations
    report_lines.extend([
        "⚠️  TOP 5 WORST DEGRADATIONS (ΔNDCG@10):",
        "-"*80,
    ])
    for i, (qid, metrics) in enumerate(sorted_improvements[-5:], 1):
        query_text = queries[qid][:100] + "..." if len(queries[qid]) > 100 else queries[qid]
        report_lines.append(f"{i}. ΔNDCG@10={metrics['ΔNDCG@10']:+.4f} | {query_text}")
    
    if t_stat is not None and p_value is not None:
        report_lines.extend([
            "",
            "="*80,
            "📊 STATISTICAL SIGNIFICANCE",
            "-"*80,
            f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}",
            f"Wilcoxon signed-rank test: statistic={wilcoxon_stat:.4f}, p={wilcoxon_p:.4f}",
            f"Significance (α=0.05): {'YES ✓' if p_value < 0.05 else 'NO ✗'}",
        ])
    
    report_lines.extend([
        "",
        "="*80,
        "📊 LLM SCORE DISTRIBUTION (Top-20 documents)",
        "-"*80,
    ])
    
    all_llm_scores = []
    for qid in detailed_records:
        all_llm_scores.extend(detailed_records[qid]["llm_raw_scores"].values())
    
    if all_llm_scores:
        report_lines.extend([
            f"Mean: {np.mean(all_llm_scores):.2f}",
            f"Median: {np.median(all_llm_scores):.2f}",
            f"Std Dev: {np.std(all_llm_scores):.2f}",
            f"Min: {np.min(all_llm_scores):.2f}",
            f"Max: {np.max(all_llm_scores):.2f}",
        ])
    
    report_lines.append("="*80)
    
    with open(f"{output_dir}/{dataset_name}_gemma2_9b_report_{timestamp}.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    # Return summary for comprehensive report
    return {
        "dataset": dataset_name,
        "model": "gemma2_9b",
        "num_queries": num_queries,
        "bm25_config": DATASET_CONFIG[dataset_name]["best_bm25_config"],
        "metrics": llm_metrics,
        "bm25_metrics": bm25_metrics,
        "improvements": improvements,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "statistical_tests": {
            "t_test": {"statistic": t_stat, "p_value": p_value} if t_stat else None,
            "wilcoxon": {"statistic": wilcoxon_stat, "p_value": wilcoxon_p} if wilcoxon_stat else None,
        },
        "timestamp": timestamp
    }

if __name__ == "__main__":
    import os
    os.makedirs("./results/llm_reranking/gemma2_9b/top20", exist_ok=True)
    
    # TEST MODE (5 queries)
    # print("\n🧪 TEST MODE (5 queries from SciFact)")
    # test_result = rerank("scifact", num_queries=5)
    # print(f"\n✅ Test Complete!")
    # print(f"NDCG@10: {test_result['metrics']['NDCG@10']:.4f}")
    # print(f"ΔNDCG@10: {test_result['improvements']['ΔNDCG@10']:+.4f} ({test_result['improvements']['ΔNDCG@10_pct']:+.2f}%)")
    # print(f"Check reports in: ./results/llm_reranking/gemma2_9b/top20/")
    # print("\n✅ If test looks good, uncomment FULL MODE below.\n")
    
    # FULL MODE (300 queries each) - UNCOMMENT TO RUN
    print("\n🚀 FULL MODE (300 queries per dataset)")
    all_results = []
    
    for dataset in ["scifact", "arguana", "nq"]:
        result = rerank(dataset, num_queries=300)
        all_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"{dataset.upper()} Summary:")
        print(f"  NDCG@10: {result['metrics']['NDCG@10']:.4f} (BM25: {result['bm25_metrics']['NDCG@10']:.4f})")
        print(f"  ΔNDCG@10: {result['improvements']['ΔNDCG@10']:+.4f} ({result['improvements']['ΔNDCG@10_pct']:+.2f}%)")
        print(f"  Wins/Losses/Ties: {result['wins']}/{result['losses']}/{result['ties']}")
        print(f"{'='*80}")
    
    # Generate comprehensive summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_lines = [
        "="*80,
        "COMPREHENSIVE SUMMARY - LLM RE-RANKING (Gemma-2-9B-IT)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        "",
        "📊 OVERALL RESULTS ACROSS DATASETS",
        "-"*80,
        f"{'Dataset':<12} | {'NDCG@10 (BM25)':<18} | {'NDCG@10 (LLM)':<18} | {'Δ':<12} | {'Wins/Losses/Ties'}",
        "-"*80,
    ]
    
    for result in all_results:
        dataset = result['dataset'].upper()
        bm25_ndcg = result['bm25_metrics']['NDCG@10']
        llm_ndcg = result['metrics']['NDCG@10']
        delta = result['improvements']['ΔNDCG@10']
        wlt = f"{result['wins']}/{result['losses']}/{result['ties']}"
        
        summary_lines.append(
            f"{dataset:<12} | {bm25_ndcg:<18.4f} | {llm_ndcg:<18.4f} | {delta:>+12.4f} | {wlt}"
        )
    
    summary_lines.extend([
        "",
        "="*80,
        "📁 DETAILED REPORTS",
        "-"*80,
    ])
    
    for result in all_results:
        summary_lines.append(f"  {result['dataset']}_gemma2_9b_report_{result['timestamp']}.txt")
    
    summary_lines.append("="*80)
    
    with open(f"./results/llm_reranking/gemma2_9b/top20/comprehensive_summary_{timestamp}.txt", "w") as f:
        f.write("\n".join(summary_lines))
    
    print("\n✅ COMPLETE! All reports saved.")