import argparse
import sys
import torch
import csv
from pathlib import Path
from tqdm import tqdm
import os
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.simple_embedding_retriever import SimpleEmbeddingRetriever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, help="Path to full collection.tsv")
    ap.add_argument("--index_out", required=True, help="Path to save embeddings.pt")
    ap.add_argument("--meta_out", required=True, help="Path to save metadata.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Max records to index (0 for all)")
    ap.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding")
    args = ap.parse_args()

    print("Initializing embedding model...")
    retriever = SimpleEmbeddingRetriever(
        index_path=args.index_out, 
        metadata_path=args.meta_out
    )
    
    records = []
    print(f"Reading {args.collection}...")
    
    count = 0
    csv.field_size_limit(sys.maxsize)
    
    try:
        with open(args.collection, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in tqdm(reader):
                if len(row) < 2:
                    continue
                
                pid, text = row[0], row[1]
                if len(text) < 20: continue 
                
                records.append({
                    "id": pid,
                    "text": text
                })
                count += 1
                if args.limit > 0 and count >= args.limit:
                    break
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return
                
    print(f"Loaded {count} records. Starting build...")
    retriever.build_index(records)
    print("Build complete.")

if __name__ == "__main__":
    main()
