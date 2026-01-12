import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_generator import ContentGenerator
from llm_client import build_client_from_env
from schema import ContentGenConfig, ContentGenRequest
from retrieval.simple_embedding_retriever import SimpleEmbeddingRetriever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", default=str(ROOT / "data/fast_index.pt"))
    ap.add_argument("--meta_path", default=None)
    ap.add_argument("--query", default="Capitalism destroys the environment")
    ap.add_argument("--text", default="The free market is the most efficient distributor of resources and ensures environmental protection through innovation.")
    ap.add_argument("--style", default="Kritik (Eco-Marxism)")
    ap.add_argument("--intensity", type=int, default=8)
    ap.add_argument("--audience", default="Policy Judges")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # 1) retrieve evidence
    print(f"Loading Dense Retriever from {args.index_path}")
    retriever = SimpleEmbeddingRetriever(index_path=args.index_path, metadata_path=args.meta_path)
    
    print(f"Searching for query: {args.query} ...")
    results = retriever.search(args.query, k=args.k)
    print(f"Found {len(results)} evidences.")
    
    evidences_text = []
    for i, e in enumerate(results):
        print(f"[{i+1}] (Score: {e['score']:.4f}) {e['text'][:100]}...")
        evidences_text.append(f"Evidence {i+1} (Score {e['score']:.2f}): {e['text']}")

    if not os.environ.get("LLM_API_KEY"):
        print("\n[!] LLM_API_KEY not found or empty. Skipping content generation step.")
        return

    # 2) build a plan placeholder with evidence
    plan = {
        "title": "Dense Retrieval Rebuttal",
        "main_arguments": [
            {"claim": "Capitalism drives innovation but at the cost of ecological collapse.", "strategies": ["Kritik", "Turn"]},
            {"claim": "Market mechanisms cannot solve externalities they create.", "strategies": ["Structural Analysis"]}
        ],
        "retrieved_evidence": evidences_text
    }

    # 3) Generate Content
    cfg = ContentGenConfig(
        model_name="gpt-4o", 
        temperature=0.7,
        max_tokens=1000
    )
    # Initialize LLM Client from environment variables
    try:
        client = build_client_from_env()
    except Exception as e:
        print(f"Failed to create LLM client: {e}")
        return

    generator = ContentGenerator(cfg, client=client)
    
    req = ContentGenRequest(
        text=args.text,
        fallacy_type="N/A",
        confidence=1.0,
        strategy_plan=plan,
        style=args.style,
        intensity=args.intensity,
        audience=args.audience,
        context=f"Topic: {args.query}"
    )

    print("\nGenerating Rebuttal...")
    try:
        response = generator.generate(req)
        print("\n=== GENERATED CONTENT ===\n")
        print(f"Title: {response.rebuttal.title}")
        print(f"\nFull Rebuttal:\n{response.rebuttal.full}")
        print("\n=========================")
    except Exception as e:
        print(f"Generation failed (likely missing API Key): {e}")

if __name__ == "__main__":
    main()
