import argparse
import sys
import json
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.opencaselist import load_opencaselist_from_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Yusuf5/OpenCaselist")
    ap.add_argument("--out", default=str(ROOT / "data/opencaselist.jsonl"))
    ap.add_argument(
        "--split",
        default=None,
        help="Optional HF split to download, e.g. 'train' or 'train[:1%]'. If omitted, downloads DatasetDict.",
    )
    ap.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap after loading a split (useful for quick RAG smoke tests).",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming dataset from {args.repo}...")
    ds = load_opencaselist_from_hf(args.repo, split=args.split, max_records=args.max_records)
    
    # Check if it's already a list/Dataset or IterableDataset
    # load_opencaselist_from_hf returns Dataset if max_records is set, or IterableDataset if not.
    
    print(f"Saving to jsonl at: {out} ...")
    count = 0
    with open(out, 'w', encoding='utf-8') as f:
        # ds might be a Dataset or IterableDataset. Both align w/ 'for item in ds'
        for item in tqdm(ds):
            # Clean items to be JSON serializable (handle bad floats etc if any)
            # HF datasets handle basic types well, but let's be safe
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + "\n")
            count += 1
            
    print(f"Successfully saved {count} records to {out}")


if __name__ == "__main__":
    main()
