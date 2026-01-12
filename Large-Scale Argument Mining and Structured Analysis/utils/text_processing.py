import pandas as pd
from config import config
from transformers import AutoTokenizer
import re
import numpy as np

# 加载tokenizer（全局单例，避免重复加载）
tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL, local_files_only=True)

def clean_text(text: str) -> str:
    """清洗文本：去除HTML标签、多余空格、特殊字符"""
    text = re.sub(r"<.*?>", "", text)  # 去除HTML markup
    text = re.sub(r"[^\w\s\.\,\!\?]", "", text)  # 保留常见标点
    text = re.sub(r"\s+", " ", text).strip()  # 合并多余空格
    return text

def split_long_text(text: str) -> list:
    """长文档分段（强制截断+滑动窗口，确保不超过模型最大长度）"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_length = len(tokens)
    chunks = []
    max_seq_len = config.MAX_SEQ_LENGTH - 2  # 预留[CLS]和[SEP]位置
    
    # 修复：兼容Config无STEP_SIZE属性的情况（保留原逻辑，仅修复属性访问方式）
    step_size = config.STEP_SIZE if hasattr(config, 'STEP_SIZE') else max_seq_len // 2
    
    # 强制截断超长文本
    if total_length > max_seq_len * 10:  # 超过10倍最大长度则直接截断前10段
        tokens = tokens[:max_seq_len * 10]
        total_length = len(tokens)
    
    # 滑动窗口分段（使用修复后的step_size）
    for i in range(0, total_length, step_size):
        end = i + max_seq_len
        chunk_tokens = tokens[i:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if len(chunk_text) > 10:  # 过滤过短分段
            chunks.append(chunk_text)
    
    return chunks if chunks else [text[:500]]  # 极端情况返回前500字符

def encode_stance_labels(labels: list) -> list:
    """立场标签编码（兼容全称/简写：Affirmative/A=0，Negative/N=1）"""
    # 扩展标签映射，支持全称和简写
    label2id = {
        "Affirmative": 0, "A": 0,
        "Negative": 1, "N": 1
    }
    # 处理异常标签 + 显式转为int64类型（核心修改）
    encoded = [int(label2id.get(label.strip(), 0)) for label in labels]
    # 转为numpy int64，避免后续DataLoader自动转float
    return np.array(encoded, dtype=np.int64).tolist()

def decode_stance_labels(label_ids: list) -> list:
    """立场标签解码"""
    id2label = {0: "Affirmative", 1: "Negative"}
    return [id2label[label_id] for label_id in label_ids]

def build_arg_component_labels(text_chunks: list, signal_data: dict) -> list:
    """基于tag/summary构建论证单元伪标签（用于无直接标注场景）"""
    # 逻辑：tag=claim（核心论点），summary=premise（证据支撑），spoken=warrant（逻辑连接）
    label2id = {"claim": 0, "premise": 1, "warrant": 2}
    pseudo_labels = []
    
    for chunk in text_chunks:
        tag = signal_data.get("tag", "").lower()
        summary = signal_data.get("summary", "").lower()
        
        if tag in chunk.lower() and len(tag) > 5:  # tag是论点摘要，含tag则为claim
            pseudo_labels.append(label2id["claim"])
        elif summary in chunk.lower() and len(summary) > 10:  # summary是证据摘要，含则为premise
            pseudo_labels.append(label2id["premise"])
        else:  # 其余为warrant（逻辑连接）
            pseudo_labels.append(label2id["warrant"])
    
    return pseudo_labels

def build_strength_scores(text_chunks: list, signal_data: dict) -> list:
    """基于report和文本长度构建立场强度分数（0-1）"""
    strength_scores = []
    text_length = signal_data.get("textlength", 0)
    report = signal_data.get("report", "")
    
    # 长度越长、report含肯定性词汇，强度越高
    base_score = min(text_length / 5000, 1.0)  # 文本长度归一化（最大5000字符）
    positive_words = ["convincing", "strong", "effective", "compelling"]
    report_score = sum(1 for word in positive_words if word in report.lower()) / len(positive_words)
    
    for _ in text_chunks:
        strength_scores.append((base_score + report_score) / 2)
    
    return strength_scores