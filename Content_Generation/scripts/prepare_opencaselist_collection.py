import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# from retrieval.opencaselist import load_opencaselist_from_disk, iter_passages 
# Changing import to avoid load_opencaselist_from_disk which expects Arrow directory
from retrieval.opencaselist import iter_passages

def load_jsonl(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data/opencaselist.jsonl")) # Changed default
    ap.add_argument("--out_dir", default=str(ROOT / "data/opencaselist_colbert"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk_chars", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    # ds = load_opencaselist_from_disk(args.dataset)
    # Switch to generator from JSONL
    print(f"Reading from {args.dataset}...")
    ds_iter = load_jsonl(args.dataset)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collection_tsv = out_dir / "collection.tsv"
    meta_jsonl = out_dir / "meta.jsonl"

    n = 0
    # iter_passages in opencaselist.py expects a Dataset or list generally, but if we check logic...
    # It likely expects dict access. yielded items from load_jsonl are dicts.
    # We need to make sure iter_passages supports an iterator as input.
    
    with collection_tsv.open("w", encoding="utf-8") as ft, meta_jsonl.open("w", encoding="utf-8") as fm:
        for original_pid, passage, meta in tqdm(iter_passages(ds_iter, limit=args.limit)):
            # ColBERT (standard loader) expects PID to be the 0-based line number
            pid_int = n
            
            # Store original structural PID in meta
            meta["original_pid"] = original_pid
            
            # ColBERT expects: pid<TAB>passage
            # Must remove newlines and tabs from passage to keep TSV format valid (one record per line)
            clean_passage = passage.replace(chr(9), ' ').replace('\n', ' ').replace('\r', ' ').strip()
            if not clean_passage:
                continue

            ft.write(f"{pid_int}\t{clean_passage}\n")
            
            # Update meta file with the integer PID
            fm.write(json.dumps({"pid": pid_int, "meta": meta}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} passages")
    print(f"collection.tsv: {collection_tsv}")
    print(f"meta.jsonl: {meta_jsonl}")


if __name__ == "__main__":
    main()
