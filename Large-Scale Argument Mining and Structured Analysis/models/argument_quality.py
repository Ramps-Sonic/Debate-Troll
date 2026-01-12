from config import config
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils.metrics import compute_quality_metrics
from utils.text_processing import tokenizer, clean_text
import os
import numpy as np
from datasets import Dataset
import torch  # 新增：导入torch，确保张量类型正确

# 新增：适配多维度回归标签的DataCollator（如果使用默认collator，需确保labels不被错误padding）
class RegressionDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = self.tokenizer.data_collator
    
    def __call__(self, features):
        # 分离labels和token特征
        labels = [torch.tensor(f.pop("labels"), dtype=torch.float32) for f in features]
        # 处理token特征（input_ids/attention_mask）
        batch = self.default_collator(features)
        # 添加labels，保持4维回归标签形状
        batch["labels"] = torch.stack(labels)
        return batch

class ArgumentQualityModel:
    def __init__(self, data_collator=None):
        # 多维度质量评估模型（输出4个分数：逻辑/相关/充分/可信）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.BACKBONE_MODEL,
            num_labels=4,  # 4个质量维度
            problem_type="regression",  # 回归任务（输出连续分数）
            local_files_only=True
        ).to(config.DEVICE)
        
        # 训练参数
        self.training_args = TrainingArguments(
            output_dir=os.path.join(config.CHECKPOINT_DIR, "argument_quality"),
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            num_train_epochs=config.EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            logging_dir=os.path.join(config.LOG_DIR, "argument_quality"),
            logging_steps=config.EVALUATE_EVERY_STEP,
            evaluation_strategy="epoch",
            eval_steps=None,
            save_strategy="epoch",
            load_best_model_at_end=True,
            # 关键修改：将 metric_for_best_model 改为 avg_r2（对应返回的 eval_avg_r2）
            metric_for_best_model="avg_r2",
            greater_is_better=True,  # R²越大越好，显式声明避免歧义
            fp16=config.DEVICE == "cuda",
            push_to_hub=False,
            report_to="none"
        )
        
        # 训练器：使用适配多维度回归的collator（优先用传入的，否则用自定义的）
        self.collator = data_collator or RegressionDataCollator(tokenizer)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_quality_metrics,
            data_collator=self.collator,  # 关键：确保labels被正确处理
            tokenizer=tokenizer
        )
    
    def build_quality_labels(self, dataset: Dataset) -> Dataset:
        """基于信号列构建质量评估伪标签（4个维度，0-1）"""
        def map_labels(examples):
            # 清洗文本后计算长度（复用text_processing的清洗逻辑）
            text = examples.get("text", "")
            cleaned_text = clean_text(text)
            textlength = len(cleaned_text)
            
            # 缺失字段兜底
            duplicateCount = examples.get("duplicateCount", 0)
            tag = examples.get("tag", "")
            cite = examples.get("cite", "")
            
            # 逻辑维度：文本长度+重复次数（重复越多逻辑越稳定）
            logic = min(textlength / 5000, 1.0) * (1 - duplicateCount / 10)
            
            # 相关维度：与tag的相似度
            relevance = self._compute_tag_similarity(cleaned_text, tag)
            
            # 充分维度：引用源数量
            sufficiency = 1.0 if cite and len(cite) > 3 else 0.5
            
            # 可信维度：重复次数
            credibility = min(1.0, duplicateCount / 5)
            
            # 关键修改：标签列名改为labels（模型默认识别）
            return {
                "labels": [logic, relevance, sufficiency, credibility]
            }
        
        return dataset.map(map_labels)
    
    def _compute_tag_similarity(self, text: str, tag: str) -> float:
        """计算文本与tag的相似度（简单字符串匹配）"""
        if not tag or len(tag) < 3:
            return 0.5
        return 1.0 if tag.lower() in text.lower() else 0.3
    
    def train(self, train_dataset, val_dataset):
        """训练质量评估模型"""
        # 构建伪标签（标签列名已改为labels）
        train_dataset = self.build_quality_labels(train_dataset)
        val_dataset = self.build_quality_labels(val_dataset)
        
        # 格式化标签列：确保labels是torch.float32类型（回归任务要求）
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],  # 关键：替换为labels
            output_all_columns=False
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],  # 关键：替换为labels
            output_all_columns=False
        )
        
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = val_dataset
        resume_path = os.path.join(config.CHECKPOINT_DIR, "argument_quality") if config.INCREMENTAL_TRAINING else None
        self.trainer.train(resume_from_checkpoint=resume_path)
    
    def evaluate(self, test_dataset):
        """评估模型"""
        test_dataset = self.build_quality_labels(test_dataset)
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=False
        )
        if len(test_dataset) == 0:
            print("警告：测试数据集为空，跳过评估")
            return {}
        metrics = self.trainer.evaluate(eval_dataset=test_dataset)
        print(f"质量评估模型评估结果：逻辑R2={metrics['eval_logic_r2']:.4f}，可信R2={metrics['eval_credibility_r2']:.4f}")
        return metrics
    
    def save_model(self, path=None):
        """保存模型"""
        save_path = path or os.path.join(config.OUTPUT_DIR, "argument_quality_model")
        self.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"质量评估模型保存至：{save_path}")