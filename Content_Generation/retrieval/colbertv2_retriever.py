from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from retrieval.types import EvidenceSnippet


@dataclass
class ColBERTv2Config:
    checkpoint: str = "colbert-ir/colbertv2.0"
    doc_maxlen: int = 180
    nbits: int = 2
    k: int = 5


class ColBERTv2Retriever:
    """ColBERTv2 wrapper.

    This uses `colbert-ai` (a.k.a. `colbert`) to build and search an index.

    Notes on Windows:
      - You may need to install `torch` matching your CUDA/CPU setup.
      - `faiss` wheels availability can vary; CPU-only is usually easiest.
    """

    def __init__(
        self,
        index_root: str,
        index_name: str,
        collection_tsv: str,
        meta_jsonl: str,
        cfg: Optional[ColBERTv2Config] = None,
    ):
        self.index_root = str(index_root)
        self.index_name = index_name
        self.collection_tsv = str(collection_tsv)
        self.meta_jsonl = str(meta_jsonl)
        self.cfg = cfg or ColBERTv2Config()

        self._meta_by_pid: Dict[str, Dict] = {}
        self._load_meta()

        # lazy init searcher
        self._searcher = None

    def _load_meta(self) -> None:
        meta_path = Path(self.meta_jsonl)
        if not meta_path.exists():
            return
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = str(obj.get("pid"))
                if pid:
                    self._meta_by_pid[pid] = obj.get("meta", {})

    def _ensure_searcher(self):
        if self._searcher is not None:
            return

        from colbert import Searcher  # type: ignore
        from colbert.infra import ColBERTConfig  # type: ignore

        self._searcher = Searcher(
            index=self.index_name,
            collection=self.collection_tsv,
            config=ColBERTConfig(root=self.index_root),
            index_root=self.index_root,  # Explicitly pass index_root
        )

    def search(self, query: str, k: Optional[int] = None) -> List[EvidenceSnippet]:
        self._ensure_searcher()
        assert self._searcher is not None

        topk = int(k or self.cfg.k)
        pids, ranks, scores = self._searcher.search(query, k=topk)

        out: List[EvidenceSnippet] = []
        for idx in range(len(pids)):
            pid_val = pids[idx]
            score = float(scores[idx])
            
            # pid_val comes from searcher, might be int or tensor
            try:
                pid_int = int(pid_val)
            except:
                pid_int = -1

            # meta key is string version of int PID? 
            # In prepare_opencaselist, we saved {"pid": int_pid, "meta": ...} to meta.jsonl
            # So self._meta_by_pid keys are stringified ints (e.g. "0", "1")
            
            meta = self._meta_by_pid.get(str(pid_int), {})
            
            # Collection access: if it's a list, use int index
            try:
                text_content = self._searcher.collection[pid_int]
            except (IndexError, TypeError):
                # Fallback if collection assumes string keys or something else
                text_content = self._searcher.collection[str(pid_int)]

            out.append(EvidenceSnippet(eid=f"E{idx+1}", text=text_content, score=score, meta=meta))

        return out


def build_colbert_index(
    index_root: str,
    index_name: str,
    collection_tsv: str,
    checkpoint: str = "colbert-ir/colbertv2.0",
    doc_maxlen: int = 180,
    nbits: int = 2,
) -> None:
    """Build a ColBERTv2 index.

    This must be run once offline after generating `collection.tsv`.
    """

    from colbert import Indexer  # type: ignore
    from colbert.infra import Run, RunConfig, ColBERTConfig  # type: ignore

    os.makedirs(index_root, exist_ok=True)

    with Run().context(RunConfig(nranks=1, experiment=index_root)):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, root=index_root)
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection_tsv)
