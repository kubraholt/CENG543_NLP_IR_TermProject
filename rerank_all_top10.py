# llm_rerank_all_models_top10.py
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

# Model configs
MODELS = {
    "phi3": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        "use_cache_flag": False,
    },
    "mistral7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "use_cache_flag": None,
    },
    "llama3_8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "trust_remote_code": False,
        "use_cache_flag": None,
    },
    "gemma2_9b": {
        "hf_id": "google/gemma-2-9b-it",
        "trust_remote_code": False,
        "use_cache_flag": None,
    },
    "qwen2.5_7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": False,
        "use_cache_flag": None,
    }
}

# Dataset configs
DATASET_CONFIG = {
    "scifact": {"best_bm25_config": "k10.82_b0.68"},
    "arguana": {"best_bm25_config": "k11.2_b0.75"},
    "nq": {"best_bm25_config": "k10.9_b0.4"},
}

# TOP-K value
TOP_K = 10


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


def call_llm(prompt: str, model, tokenizer, model_config: dict) -> float:
    """Call model and parse score"""
    try:
        # Check if model supports system role
        model_name = model_config['hf_id'].lower()
        
        if 'gemma' in model_name:
            # Gemma doesn't support system role
            messages = [
                {"role": "user", "content": f"Return only one integer from 0 to 10. Do not output any other text.\n\n{prompt}"},
            ]
        else:
            # Other models support system role
            messages = [
                {"role": "system", "content": "Return only one integer from 0 to 10. Do not output any other text."},
                {"role": "user", "content": prompt},
            ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(model.device)
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(model.device)
        
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 8,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
        }
        
        if model_config.get("use_cache_flag") is not None:
            gen_kwargs["use_cache"] = model_config["use_cache_flag"]
        
        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)
        
        prompt_len = input_ids.shape[1]
        new_tokens = outputs[0, prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
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
    """Calculate improvement metrics"""
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
    """Compute per-query metrics"""
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


def rerank_single_model(model_key: str, dataset_name: str, num_queries: int = 300):
    """Re-rank with single model"""
    
    model_config = MODELS[model_key]
    
    print(f"\n{'='*80}")
    print(f"Model: {model_key.upper()} | Dataset: {dataset_name.upper()} | Top-K: {TOP_K}")
    print(f"{'='*80}")
    
    # Load model
    print(f"📦 Loading: {model_config['hf_id']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['hf_id'],
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['hf_id'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    model.eval()
    print(f"✅ Model loaded")
    
    # Load data
    corpus, queries, qrels = GenericDataLoader(f"./datasets/{dataset_name}").load(split="test")
    bm25_results = load_bm25(dataset_name)
    query_ids = list(bm25_results.keys())[:num_queries]
    
    # Re-rank
    llm_results = {}
    detailed_records = {}
    
    for qid in tqdm(query_ids, desc="Re-ranking"):
        query_text = queries[qid]
        bm25_ranking = bm25_results[qid]
        top_docs = list(bm25_ranking.keys())[:TOP_K]
        
        # Get LLM scores
        llm_scores = {}
        for doc_id in top_docs:
            if doc_id in corpus:
                doc = corpus[doc_id]
                doc_text = f"{doc.get('title', '')} {doc.get('text', '')}"[:800]
                prompt = create_prompt(query_text, doc_text, dataset_name)
                llm_scores[doc_id] = call_llm(prompt, model, tokenizer, model_config)
        
        # Normalize LLM scores
        bm25_topk_scores = {doc_id: bm25_ranking[doc_id] for doc_id in top_docs if doc_id in bm25_ranking}
        if bm25_topk_scores:
            bm25_min = min(bm25_topk_scores.values())
            bm25_max = max(bm25_topk_scores.values())
            llm_scores_normalized = normalize_scores(llm_scores, bm25_min, bm25_max)
        else:
            llm_scores_normalized = llm_scores
        
        # Merge
        remaining_docs = {doc_id: score for doc_id, score in bm25_ranking.items() if doc_id not in llm_scores_normalized}
        final_ranking = {**llm_scores_normalized, **remaining_docs}
        final_ranking = dict(sorted(final_ranking.items(), key=lambda x: x[1], reverse=True))
        
        llm_results[qid] = final_ranking
        
        # Store detailed info
        detailed_records[qid] = {
            "query": query_text,
            "llm_raw_scores": llm_scores,
            "llm_normalized_scores": llm_scores_normalized,
        }
    
    # Evaluate
    evaluator = EvaluateRetrieval()
    filtered_qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    
    k_values = [1, 3, 5, 10, 20, 100]
    ndcg_llm, map_llm, recall_llm, precision_llm = evaluator.evaluate(
        filtered_qrels, llm_results, k_values
    )
    
    # Evaluate BM25
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
            }
    
    # Win/Loss/Tie
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
    output_dir = f"./results/llm_reranking/{model_key}/top10"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Final rankings JSON
    with open(f"{output_dir}/{dataset_name}_{model_key}_top10_{timestamp}.json", "w") as f:
        json.dump(llm_results, f, indent=2)
    
    # 2. Generate TXT report
    report_lines = [
        "="*80,
        f"LLM RE-RANKING EVALUATION REPORT (TOP-10)",
        f"Dataset: {dataset_name.upper()}",
        f"Model: {model_key.upper()}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        "",
        "📊 CONFIGURATION",
        "-"*80,
        f"Number of queries: {num_queries}",
        f"BM25 baseline config: {DATASET_CONFIG[dataset_name]['best_bm25_config']}",
        f"Re-ranking strategy: Top-{TOP_K} documents",
        f"LLM temperature: 0.0 (greedy decoding)",
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
        f"📊 LLM SCORE DISTRIBUTION (Top-{TOP_K} documents)",
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
    
    with open(f"{output_dir}/{dataset_name}_{model_key}_top10_report_{timestamp}.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    # Result summary
    result = {
        "model": model_key,
        "dataset": dataset_name,
        "top_k": TOP_K,
        "num_queries": num_queries,
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
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print(f"✅ {model_key.upper()} - {dataset_name.upper()}: NDCG@10={llm_metrics['NDCG@10']:.4f} (Δ={improvements['ΔNDCG@10']:+.4f}, {improvements['ΔNDCG@10_pct']:+.2f}%)")
    
    return result


def run_all_models():
    """Run all models on all datasets with Top-10"""
    
    print("\n" + "="*80)
    print("🚀 LLM RE-RANKING - ALL MODELS - TOP-10 COMPARISON")
    print("="*80)
    
    datasets = ["scifact", "arguana", "nq"]
    all_results = []
    
    for model_key in MODELS.keys():
        for dataset in datasets:
            try:
                result = rerank_single_model(model_key, dataset, num_queries=300)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error: {model_key} - {dataset}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save comprehensive JSON summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        "experiment": "LLM Re-ranking - All Models - Top-10",
        "date": datetime.now().isoformat(),
        "top_k": TOP_K,
        "results": all_results
    }
    
    with open(f"./results/llm_reranking/all_models_top10_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create comprehensive TXT summary
    summary_lines = [
        "="*80,
        "COMPREHENSIVE SUMMARY - LLM RE-RANKING (ALL MODELS - TOP-10)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        "",
        "📊 OVERALL RESULTS ACROSS ALL MODELS",
        "-"*80,
        f"{'Model':<15} | {'Dataset':<10} | {'NDCG@10 (BM25)':<15} | {'NDCG@10 (LLM)':<15} | {'Δ':<12} | {'W/L/T'}",
        "-"*80,
    ]
    
    for result in all_results:
        model = result['model']
        dataset = result['dataset'].upper()
        bm25_ndcg = result['bm25_metrics']['NDCG@10']
        llm_ndcg = result['metrics']['NDCG@10']
        delta = result['improvements']['ΔNDCG@10']
        wlt = f"{result['wins']}/{result['losses']}/{result['ties']}"
        
        summary_lines.append(
            f"{model:<15} | {dataset:<10} | {bm25_ndcg:<15.4f} | {llm_ndcg:<15.4f} | {delta:>+12.4f} | {wlt}"
        )
    
    summary_lines.extend([
        "",
        "="*80,
        "📁 DETAILED REPORTS (per model-dataset)",
        "-"*80,
    ])
    
    for result in all_results:
        summary_lines.append(f"  {result['dataset']}_{result['model']}_top10_report_{result['timestamp']}.txt")
    
    summary_lines.append("="*80)
    
    with open(f"./results/llm_reranking/all_models_top10_summary_{timestamp}.txt", "w") as f:
        f.write("\n".join(summary_lines))
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS - TOP-10 RE-RANKING")
    print("="*80)
    print(f"{'Model':<15} | {'Dataset':<10} | {'NDCG@10 (BM25)':<15} | {'NDCG@10 (LLM)':<15} | {'Δ%':<10}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['model']:<15} | {result['dataset']:<10} | {result['bm25_metrics']['NDCG@10']:<15.4f} | {result['metrics']['NDCG@10']:<15.4f} | {result['improvements']['ΔNDCG@10_pct']:>+9.2f}%")
    
    print("="*80)
    print(f"✅ JSON summary: all_models_top10_summary_{timestamp}.json")
    print(f"✅ TXT summary: all_models_top10_summary_{timestamp}.txt")
    print("="*80)


if __name__ == "__main__":
    run_all_models()