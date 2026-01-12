import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.colbertv2_retriever import build_colbert_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=str(ROOT / "data/opencaselist_colbert/collection.tsv"))
    ap.add_argument("--index_root", default=str(ROOT / "data/colbert_indexes"))
    ap.add_argument("--index_name", default="opencaselist")
    ap.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    ap.add_argument("--doc_maxlen", type=int, default=180)
    ap.add_argument("--nbits", type=int, default=2)
    args = ap.parse_args()

    Path(args.index_root).mkdir(parents=True, exist_ok=True)

    build_colbert_index(
        index_root=args.index_root,
        index_name=args.index_name,
        collection_tsv=args.collection,
        checkpoint=args.checkpoint,
        doc_maxlen=args.doc_maxlen,
        nbits=args.nbits,
    )

    print("Index build done")


if __name__ == "__main__":
    main()
