import json
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

def clean_text(text):
    if not text:
        return ""
    # Remove tabs and newlines to keep TSV clean
    return text.replace("\t", " ").replace("\n", " ").strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="ContentGenerator/data/debatesum", help="Directory to save output files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records (0 for all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    collection_path = out_dir / "collection.tsv"
    meta_path = out_dir / "meta.jsonl"
    
    print("Loading DebateSum dataset...")
    try:
        ds = load_dataset("Hellisotherpeople/DebateSum", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Processing and writing files...")
    
    count = 0
    with open(collection_path, "w", encoding="utf-8", newline="") as f_tsv, \
         open(meta_path, "w", encoding="utf-8") as f_meta:
        
        writer = csv.writer(f_tsv, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        
        for item in tqdm(ds):
            # Extract fields
            tag = clean_text(item.get("Tag", ""))
            citation = clean_text(item.get("Citation", ""))
            extract = clean_text(item.get("Extract", ""))
            full_doc = clean_text(item.get("Full-Document", ""))
            abstract = clean_text(item.get("Abstract", ""))
            
            # content selection logic
            content = extract
            if not content or len(content) < 50:
                 content = full_doc
            if not content or len(content) < 50:
                 content = abstract
            
            # If still nothing, skip
            if not content:
                continue

            # Truncate very long content for retrieval efficiency (ColBERT handles shorter chunks better)
            # But let's keep it reasonable, say 2000 chars
            if len(content) > 2000:
                content = content[:2000]

            # Construct display text for retrieval
            # title: tag, source: citation
            # We put them in the text so ColBERT can see them? 
            # Or just put content. ColBERT works best on "passages".
            # Standard practice: Title [SEP] Content.
            text_for_index = f"Topic: {tag} | Source: {citation} | {content}"
            
            # write TSV: id, text
            pid = str(count)
            writer.writerow([pid, text_for_index])
            
            # write Meta
            meta_obj = {
                "pid": pid,
                "meta": {
                    "tag": tag,
                    "citation": citation,
                    "id": item.get("Unnamed: 0", pid), # original ID if available
                    "camp": item.get("DebateCamp", "")
                }
            }
            f_meta.write(json.dumps(meta_obj) + "\n")
            
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
                
    print(f"Finished. Processed {count} records.")
    print(f"Collection: {collection_path}")
    print(f"Meta: {meta_path}")

if __name__ == "__main__":
    main()
