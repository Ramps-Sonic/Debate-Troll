from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def load_opencaselist_from_hf(repo: str = "Yusuf5/OpenCaselist", split: str | None = None, max_records: int | None = None):
    """Download/load dataset from HF (requires internet on first run).

    Args:
        repo: HF dataset repo.
        split: Optional split spec, e.g. 'train' or 'train[:1%]'.
        max_records: Optional cap after loading a split.
    """
    from datasets import load_dataset, Dataset, IterableDataset  # lazy import

    # Use streaming=True to avoid full dataset type-checking errors (e.g. "CA" in float column)
    # and to allow partial download.
    ds = load_dataset(repo, split=split or 'train', streaming=True)
    
    if max_records is not None:
        ds = ds.take(int(max_records))
    
    # Return the iterable directly. Do not materialize into RAM.
    return ds


def load_opencaselist_from_disk(path: str):
    from datasets import load_from_disk  # lazy import

    return load_from_disk(path)


def _flatten_strings(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append((prefix or "text", s))
        return out
    if isinstance(obj, (int, float, bool)):
        return out
    if isinstance(obj, list):
        for i, it in enumerate(obj[:50]):
            out.extend(_flatten_strings(it, f"{prefix}[{i}]" if prefix else f"[{i}]"))
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_strings(v, kp))
        return out
    return out


_PREFERRED_KEYS = [
    "case_name",
    "title",
    "name",
    "citation",
    "court",
    "year",
    "date",
    "summary",
    "facts",
    "issue",
    "holding",
    "decision",
    "opinion",
    "content",
    "text",
    "body",
]


def record_to_document(record: Dict[str, Any], max_chars: int = 6000) -> Dict[str, Any]:
    """Convert a dataset record into a document dict with `title` and `text`.

    We don't assume schema; we heuristically pick useful string fields.
    """
    strings = _flatten_strings(record)

    # pick title
    title = ""
    for k in ["case_name", "title", "name"]:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            title = v.strip()
            break

    picked: List[str] = []
    used_keys = set()

    # prefer known keys
    for key in _PREFERRED_KEYS:
        for k, s in strings:
            if k == key or k.endswith("." + key):
                if k in used_keys:
                    continue
                used_keys.add(k)
                picked.append(s)
                break

    # fallback: add other strings until length
    if not picked:
        for k, s in strings:
            if k in used_keys:
                continue
            used_keys.add(k)
            picked.append(s)
            if sum(len(x) for x in picked) > max_chars:
                break

    text = "\n\n".join(picked)
    text = text[:max_chars]

    meta = {}
    for k in ["case_name", "title", "citation", "court", "year", "date", "url", "source"]:
        v = record.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            meta[k] = v

    if title and "title" not in meta:
        meta["title"] = title

    return {"title": title, "text": text, "meta": meta}


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_chars <= overlap:
        overlap = max(0, chunk_chars // 5)

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def iter_passages(ds, split_preference: List[str] | None = None, limit: int | None = None):
    """Yield passages as (pid, passage_text, meta)."""

    split_preference = split_preference or ["train", "validation", "test"]

    # ds may be DatasetDict or Dataset
    if hasattr(ds, "keys"):
        split = None
        for s in split_preference:
            if s in ds:
                split = s
                break
        if split is None:
            split = list(ds.keys())[0]
        dataset = ds[split]
    else:
        dataset = ds

    count = 0
    for i, rec in enumerate(dataset):
        doc = record_to_document(rec)
        passages = chunk_text(doc["text"])
        base_meta = dict(doc.get("meta", {}))
        if doc.get("title"):
            base_meta.setdefault("title", doc["title"])

        for j, p in enumerate(passages):
            pid = f"{i}-{j}"
            meta = dict(base_meta)
            meta["doc_index"] = i
            meta["chunk_index"] = j
            yield pid, p, meta
            count += 1
            if limit is not None and count >= limit:
                return
