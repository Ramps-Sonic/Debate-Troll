import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
from pathlib import Path
import sys

class SimpleEmbeddingRetriever:
    def __init__(self, index_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", metadata_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        self.embeddings = None
        self.metadata = []
        
        # Load index if exists
        if self.index_path.exists():
            self._load_index()

    def _load_index(self):
        print(f"Loading Simple Embedding Index from {self.index_path}...")
        try:
            # Check if metadata path is separate
            if self.metadata_path and self.metadata_path.exists():
                # Load embeddings
                self.embeddings = torch.load(self.index_path, map_location=self.device)
                
                # Load metadata line by line (memory efficient) or json load
                print(f"Loading metadata from {self.metadata_path}...")
                self.metadata = []
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                     # Check extension
                     if self.metadata_path.suffix == '.jsonl':
                         for line in f:
                             try:
                                self.metadata.append(json.loads(line))
                             except:
                                pass
                     else:
                         # Fallback 
                         pass
            else:
                # Legacy: everything in pt
                data = torch.load(self.index_path, map_location=self.device)
                if isinstance(data, dict):
                    self.embeddings = data['embeddings'].to(self.device)
                    self.metadata = data.get('metadata', [])
                else:
                    self.embeddings = data.to(self.device)
                    self.metadata = []
                
            print(f"Loaded {len(self.metadata)} documents.")
        except Exception as e:
            print(f"Error loading index: {e}")
            self.embeddings = None
            self.metadata = []

    def save_index(self):
        if self.metadata_path:
            # Save separate
            print(f"Saving embeddings to {self.index_path}...")
            torch.save(self.embeddings, self.index_path)
            
            print(f"Saving metadata to {self.metadata_path}...")
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                for item in self.metadata:
                    f.write(json.dumps(item) + "\n")
        else:
            # Legacy save
            print(f"Saving index to {self.index_path}")
            torch.save({
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, self.index_path)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 256):
        all_embs = []
        self.model.eval()
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i : i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embs.append(sentence_embeddings.cpu())
            
        return torch.cat(all_embs, dim=0)

    def build_index(self, records: List[Dict[str, Any]]):
        """
        records: List of dicts with 'text' and 'id' keys.
        """
        print(f"Encoding {len(records)} documents...")
        texts = [r['text'] for r in records]
        self.metadata = records
        self.embeddings = self.encode(texts).to(self.device)
        self.save_index()

    def search(self, query: str, k: int = 5):
        """
        Search for query in the index.
        Returns list of objects with 'text', 'score', 'id'.
        """
        if self.embeddings is None:
            raise ValueError("Index not loaded. Call load_index or build_index first.")

        query_emb = self.encode([query]).to(self.device) # (1, D)
        
        # Cosine similarity
        # embeddings is (N, D), normalized
        # query_emb is (1, D), normalized
        scores = torch.mm(query_emb, self.embeddings.transpose(0, 1)).squeeze(0) # (N,)
        
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(self.embeddings)))
        
        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            idx = idx.item()
            score = score.item()
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append({
                "id": meta.get("id"),
                "text": meta.get("text"),
                "score": score,
                "metadata": meta
            })
            
        return results
