import os
from config import config
from utils.text_processing import (
    clean_text, split_long_text, encode_stance_labels,
    build_arg_component_labels, build_strength_scores
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorForTokenClassification, DataCollatorWithPadding
from datasets import Dataset

# 全局加载tokenizer（仅加载一次，避免重复）
tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL, local_files_only=True)

# 生成模拟数据（匹配config所需字段）
def generate_sample_dataset():
    if not os.path.exists(config.DATASET_PATH):
        os.makedirs("./data", exist_ok=True)
        sample_data = {
            "fulltext": [f"Debate argument {i}: This is a sample claim with evidence." for i in range(100)],
            "side": np.random.choice(["Affirmative", "Negative", "A", "N"], 100),
            "tag": [f"Sample claim {i}" for i in range(100)],
            "summary": [f"Sample summary for argument {i}" for i in range(100)],
            "spoken": [f"Spoken summary {i}" for i in range(100)],
            "textlength": np.random.randint(500, 5000, 100),
            "report": [f"Report for round {i}: Strong argument." for i in range(100)],
            "cite": [f"Source {i}" for i in range(100)],
            "duplicateCount": np.random.randint(0, 3, 100)
        }
        pd.DataFrame(sample_data).to_csv(config.DATASET_PATH, index=False)
        print("生成模拟数据集完成，路径：", config.DATASET_PATH)

def load_raw_dataset() -> pd.DataFrame:
    """加载原始数据集，仅采样1%用于本地验证"""
    generate_sample_dataset()
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"数据集文件未找到：{config.DATASET_PATH}，请检查路径")
    
    df = pd.read_csv(config.DATASET_PATH)
    
    # 筛选实际存在的列
    required_cols = [
        config.TEXT_COL, config.STANCE_LABEL_COL,
        *config.ARG_COMPONENT_SIGNAL_COLS, *config.STRENGTH_SIGNAL_COLS
    ]
    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols].dropna(subset=[config.TEXT_COL, config.STANCE_LABEL_COL])
    
    # 采样逻辑
    sample_ratio = config.DATA_SAMPLE_RATIO if len(df) >= 100 else 1.0
    df_sampled = df.sample(frac=sample_ratio, random_state=42)
    print(f"加载数据集完成，采样后样本数：{len(df_sampled)}")
    
    return df_sampled

def preprocess_dataset(df: pd.DataFrame) -> Dataset:
    """预处理数据集：清洗、分段、构建伪标签"""
    processed_data = []
    
    for idx, row in df.iterrows():
        # 文本清洗
        raw_text = row[config.TEXT_COL]
        clean_txt = clean_text(raw_text)
        
        # 长文档分段
        text_chunks = split_long_text(clean_txt)
        if not text_chunks:
            continue
        
        # 构建信号数据字典（仅读取存在的列）
        signal_data = {}
        for col in config.ARG_COMPONENT_SIGNAL_COLS + config.STRENGTH_SIGNAL_COLS:
            if col in df.columns:
                signal_data[col] = row[col] if pd.notna(row[col]) else ""
        
        # 构建伪标签和强度分数
        arg_component_labels = build_arg_component_labels(text_chunks, signal_data)
        stance_labels = encode_stance_labels([row[config.STANCE_LABEL_COL]] * len(text_chunks))
        strength_scores = build_strength_scores(text_chunks, signal_data)
        
        # 循环内定义chunk，正确拼接数据（移除多余字段读取）
        for i, chunk in enumerate(text_chunks):
            base_info = {
                "text": chunk,
                "arg_component_id": arg_component_labels[i],
                "stance_id": stance_labels[i],
                "strength_score": strength_scores[i],
                "original_idx": idx
            }
            processed_data.append(base_info)
    
    # 转换为Dataset并过滤异常值
    processed_df = pd.DataFrame(processed_data)
    processed_df = processed_df[processed_df["text"].str.len() > 10]
    return Dataset.from_pandas(processed_df)

# 核心Tokenize函数（仅新增labels列，其余逻辑不变）
def tokenize_wrapper(examples, task_type="classification", label_col=None):
    if task_type == "token_classification":
        # 序列标注逻辑不变
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_overflowing_tokens=False
        )
        if label_col and label_col in examples:
            labels = []
            for label in examples[label_col]:
                label_padded = [label] + [-100] * (config.MAX_SEQ_LENGTH - 1)
                labels.append(label_padded[:config.MAX_SEQ_LENGTH])
            tokenized["labels"] = labels
        return tokenized
    else:
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        # 1. 立场分类：标签重命名为 cls_labels（避免和回归 labels 冲突）
        if "stance_id" in examples:
            tokenized["cls_labels"] = examples["stance_id"]  # 不再用 labels 命名
        # 2. 强度回归：设置 labels 列为浮点型（核心修复！）
        if "strength_score" in examples:
            tokenized["labels"] = [float(score) for score in examples["strength_score"]]
        return tokenized

def prepare_dataloaders() -> tuple:
    """准备训练/验证/测试数据集（8:1:1划分，适配datasets.Dataset类型）"""
    # 加载并预处理数据
    raw_df = load_raw_dataset()
    dataset = preprocess_dataset(raw_df)
    
    # 转换为DataFrame后划分
    dataset_df = dataset.to_pandas()
    total_size = len(dataset_df)
    
    # 划分数据集（适配小数据量）
    if total_size >= 10:
        train_df, temp_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        # 转换回datasets.Dataset
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    else:
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset
    
    # 1. 序列标注任务（论证单元识别）- 保留labels列
    def tokenize_tc(examples):
        return tokenize_wrapper(examples, task_type="token_classification", label_col="arg_component_id")
    
    tokenized_train_tc = train_dataset.map(
        tokenize_tc,
        batched=True,
        # 只删除原始文本列，保留labels和关键标签列
        remove_columns=[col for col in train_dataset.column_names if col not in ["arg_component_id"]]
    )
    tokenized_val_tc = val_dataset.map(
        tokenize_tc,
        batched=True,
        remove_columns=[col for col in val_dataset.column_names if col not in ["arg_component_id"]]
    )
    tokenized_test_tc = test_dataset.map(
        tokenize_tc,
        batched=True,
        remove_columns=[col for col in test_dataset.column_names if col not in ["arg_component_id"]]
    )
    
    # 2. 分类/回归任务（立场检测/强度估计）- 保留标签列
    def tokenize_cls(examples):
        return tokenize_wrapper(examples, task_type="classification")
    
    tokenized_train_cls = train_dataset.map(
        tokenize_cls,
        batched=True,
        # 只删除原始文本列，保留核心标签列（移除多余字段）
        remove_columns=[col for col in train_dataset.column_names if col not in ["stance_id", "strength_score"]]
    )
    tokenized_val_cls = val_dataset.map(
        tokenize_cls,
        batched=True,
        remove_columns=[col for col in val_dataset.column_names if col not in ["stance_id", "strength_score"]]
    )
    tokenized_test_cls = test_dataset.map(
        tokenize_cls,
        batched=True,
        remove_columns=[col for col in test_dataset.column_names if col not in ["stance_id", "strength_score"]]
    )
    
    # 格式化标签列（匹配模型输入要求，移除多余字段）
    tc_cols = ["input_ids", "attention_mask", "labels"]  # 序列标注用labels
    cls_cols = ["input_ids", "attention_mask", "cls_labels", "labels"]    # 仅保留核心列
    
    # 设置Torch格式
    for ds in [tokenized_train_tc, tokenized_val_tc, tokenized_test_tc]:
        ds.set_format(type="torch", columns=tc_cols)
    for ds in [tokenized_train_cls, tokenized_val_cls, tokenized_test_cls]:
        ds.set_format(type="torch", columns=cls_cols)
    
    # 数据整理器
    data_collator_tc = DataCollatorForTokenClassification(tokenizer=tokenizer)
    data_collator_cls = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return {
        "token_classification": (tokenized_train_tc, tokenized_val_tc, tokenized_test_tc, data_collator_tc),
        "classification": (tokenized_train_cls, tokenized_val_cls, tokenized_test_cls, data_collator_cls)
    }