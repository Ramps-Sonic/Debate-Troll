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
from retrieval.colbertv2_retriever import ColBERTv2Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_root", default=str(ROOT / "data/colbert_indexes"))
    # Adjusted to match ColBERT default output structure (root/indexes/name)
    ap.add_argument("--index_name", default="indexes/opencaselist")
    ap.add_argument("--collection", default=str(ROOT / "data/opencaselist_colbert/collection.tsv"))
    ap.add_argument("--meta", default=str(ROOT / "data/opencaselist_colbert/meta.jsonl"))
    ap.add_argument("--query", default="Coloniality Modernity Capitalism")
    ap.add_argument("--text", default="Western liberal democracy and capitalism are the only paths to true freedom and progress.")
    ap.add_argument("--style", default="Critical Theory (Kritik)")
    ap.add_argument("--intensity", type=int, default=7)
    ap.add_argument("--audience", default="Philosophy Judges")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # 1) retrieve evidence
    print(f"Searching for query: {args.query} ...")
    retriever = ColBERTv2Retriever(
        index_root=args.index_root,
        index_name=args.index_name,
        collection_tsv=args.collection,
        meta_jsonl=args.meta,
    )
    evidences = retriever.search(args.query, k=args.k)
    print(f"Found {len(evidences)} evidences.")
    for i, e in enumerate(evidences):
        print(f"[{i+1}] {e.text[:100]}...")

    if not os.environ.get("LLM_API_KEY"):
        print("\n[!] LLM_API_KEY not found in env. Jumping content generation step.")
        print("Set LLM_API_KEY and LLM_API_BASE to generate rebuttal.")
        return

    # 2) build a very small plan placeholder (you can plug StrategyPlanner output here)
    plan = {
        "primary_strategy": "Kritik: Expose the colonial matrix of power",
        "attack_steps": [
            {"step": 1, "goal": "Link to Coloniality", "move": "Reveal the hidden coloniality", "example": "Modernity is inseparable from coloniality.", "evidence_needed": "Quijano/Mignolo on colonial matrix"},
            {"step": 2, "goal": "Reject Linear Progress", "move": "Deconstruct the myth of progress", "example": "Progress is a eurocentric construct.", "evidence_needed": "Critique of developmentalism"},
        ],
        "risk_assessment": {"risk_level": "medium", "risks": ["Jargon overload"], "mitigation": ["Explain terms like 'coloniality'"]},
        "success_probability": 0.82,
        "timing_advice": ["Start with the link to colonial history"],
        "evidence_needs": [args.query],
    }

    # 3) call content generator (optional: requires LLM env vars)
    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE", "")
    can_llm = bool(api_key and api_base)

    gen = None
    if can_llm:
        client = build_client_from_env(default_model=os.environ.get("LLM_MODEL", "qwen-max"))
        gen = ContentGenerator(ContentGenConfig(model_name=os.environ.get("LLM_MODEL", "qwen-max")), client)

    # Inject evidences into plan (ContentGenerator prompt will read it)
    plan["retrieved_evidence"] = [e.format_for_prompt() for e in evidences]

    # Force English output via robust prompting
    constraints = {"no_personal_attack": True, "no_new_facts": True, "must_cite_evidence": True, "language": "English"}

    req = ContentGenRequest(
        text=args.text,
        fallacy_type="hasty_generalization",
        confidence=0.6,
        strategy_plan=plan,
        style=args.style,
        intensity=args.intensity,
        audience=args.audience,
        constraints=constraints,
    )

    # HACK: Patch the prompt building temporarily to enforce English if needed
    # Ideally prompts.py should handle this, but passing it in constraints helps if the model is smart.
    # Otherwise, we rely on the input text being English.
    
    out = gen.generate(req, enable_satire=True) if gen is not None else None

    print("\n=== Retrieved Evidence ===")
    for e in evidences:
        print(e.format_for_prompt())
        print("-")

    if out is None:
        print("\n[SKIP] LLM env vars not set; only retrieval shown.")
        print("Set LLM_API_KEY and LLM_API_BASE to enable generation.")
        return

    print("\n=== Rebuttal (short) ===\n", out.rebuttal.short)
    print("\n=== Rebuttal (full) ===\n", out.rebuttal.full)
    if out.satire:
        print("\n=== Satire (short) ===\n", out.satire.short)


if __name__ == "__main__":
    main()
