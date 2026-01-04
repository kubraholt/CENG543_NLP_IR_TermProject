# 04_llm_reranker.py
import json
import ollama
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from datetime import datetime

def create_rerank_prompt(query: str, doc_id: str, doc_text: str, dataset_name: str = "scifact") -> str:
    """
    Create task-specific prompts for different datasets.
    - Arguana: Looks for COUNTER-ARGUMENTS (opposing views)
    - Others: Looks for RELEVANCE (standard IR)
    """
    
    # ARGUANA: Counter-Argument Detection
    if dataset_name == "arguana":
        return f"""You are an expert debater and argument analyst.
                Your task is to determine if the document contains a COUNTER-ARGUMENT (opposing view) to the query argument.

                Query Argument: {query}

                Document: {doc_text[:1500]}

                Rate strictly based on whether the document DISAGREES with or OPPOSES the query's argument:
                - 0: Agrees with the query or is completely unrelated
                - 1-3: Discusses the topic but does not clearly oppose
                - 4-6: Presents some opposing points or mild disagreement
                - 7-9: Clearly argues against the query's position
                - 10: Directly and strongly refutes the query's argument

                Respond with ONLY a single number (0-10), nothing else."""

    # SCIFACT / NQ / DEFAULT: Standard Relevance
    else:
        return f"""You are an expert relevance assessor. Given a query and a document, rate how relevant the document is to the query.

                Query: {query}

                Document: {doc_text[:1500]}

                Rate the relevance on a scale of 0-10:
                - 0: Completely irrelevant
                - 1-3: Slightly relevant, mentions related topics
                - 4-6: Moderately relevant, partially answers the query
                - 7-9: Highly relevant, directly addresses the query
                - 10: Perfect match, completely answers the query

                Respond with ONLY a single number (0-10), nothing else."""

def get_llm_score(prompt: str, model: str = "llama3:8b") -> float:
    """Get relevance score from LLM"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        score_text = response["message"]["content"].strip()
        # Extract first number found
        for word in score_text.split():
            try:
                score = float(word.replace(",", "").replace(".", ""))
                return min(max(score, 0), 10)  # Clamp to 0-10
            except:
                continue
        return 5.0  # Default if parsing fails
    except Exception as e:
        print(f"Error: {e}")
        return 5.0

def rerank_with_llm(dataset_name: str, top_k_rerank: int = 20, num_queries: int = 50):
    """
    Re-rank BM25 results using LLM
    
    Args:
        dataset_name: Name of dataset (scifact, arguana, nq)
        top_k_rerank: Number of top documents to re-rank per query
        num_queries: Number of queries to process (for testing)
    """
    print(f"\n{'='*60}")
    print(f"🤖 LLM RE-RANKING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    corpus, queries, qrels = GenericDataLoader(f"./datasets/{dataset_name}").load(split="test")
    print(f"✅ Loaded {len(queries)} queries, {len(corpus)} documents")
    
    # Load BM25 results
    with open(f"./results/{dataset_name}_bm25_top100.json") as f:
        bm25_results = json.load(f)
    
    # Limit queries for testing
    query_ids = list(bm25_results.keys())[:num_queries]
    print(f"📝 Processing {len(query_ids)} queries, re-ranking top-{top_k_rerank} docs each")
    
    # Re-rank
    llm_results = {}
    
    for qid in tqdm(query_ids, desc="Re-ranking"):
        query_text = queries[qid]
        bm25_docs = bm25_results[qid]
        
        # Get top-k docs to re-rank
        top_doc_ids = list(bm25_docs.keys())[:top_k_rerank]
        
        doc_scores = {}
        for doc_id in top_doc_ids:
            if doc_id in corpus:
                doc = corpus[doc_id]
                doc_text = f"{doc.get('title', '')} {doc.get('text', '')}"
                
                prompt = create_rerank_prompt(query_text, doc_id, doc_text, dataset_name)
                score = get_llm_score(prompt)
                doc_scores[doc_id] = score
        
        # Sort by LLM score (descending)
        sorted_docs = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))
        llm_results[qid] = sorted_docs
    
    # Save results
    output_file = f"./results/{dataset_name}_llm_reranked_top{top_k_rerank}.json"
    with open(output_file, "w") as f:
        json.dump(llm_results, f, indent=2)
    print(f"💾 Saved: {output_file}")
    
    # Evaluate
    print(f"\n📊 LLM Re-ranking Performance:")
    evaluator = EvaluateRetrieval()
    
    # Filter qrels to only include processed queries
    filtered_qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    
    ndcg, map_score, recall, precision = evaluator.evaluate(
        filtered_qrels, llm_results, [1, 3, 5, 10]
    )
    
    results = {
        "dataset": dataset_name,
        "model": "llama3:8b",
        "num_queries": len(query_ids),
        "top_k_rerank": top_k_rerank,
        "ndcg@10": ndcg.get("NDCG@10", 0),
        "ndcg@5": ndcg.get("NDCG@5", 0),
        "p@3": precision.get("P@3", 0),
        "p@1": precision.get("P@1", 0),
    }
    
    print(f"   NDCG@10: {results['ndcg@10']:.4f}")
    print(f"   NDCG@5:  {results['ndcg@5']:.4f}")
    print(f"   P@3:     {results['p@3']:.4f}")
    print(f"   P@1:     {results['p@1']:.4f}")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    
    all_results = []
    
    # Start with SciFact (smallest, fastest)
    # Adjust num_queries and top_k_rerank based on time constraints
    for dataset in ["arguana"]: # Add "arguana", "nq" later
        result = rerank_with_llm(
            dataset_name=dataset,
            top_k_rerank=20,    # Re-rank top 20 docs per query
            num_queries=50      # Process 50 queries for testing
        )
        all_results.append(result)
    
    # Save summary
    summary = {
        "experiment": "LLM Re-ranking (Llama 3 8B)",
        "date": datetime.now().isoformat(),
        "results": all_results
    }
    
    with open("./results/llm_reranking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ LLM RE-RANKING COMPLETE")
    print("="*60)