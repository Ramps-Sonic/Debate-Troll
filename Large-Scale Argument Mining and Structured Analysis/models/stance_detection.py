from config import config
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback  
)
from utils.metrics import compute_stance_metrics, compute_strength_metrics
from utils.text_processing import tokenizer
import os
import torch  

class StanceDetectionModel:
    def __init__(self):
        # ========== 1. 分类/回归Collator（无需修改，动态适配标签类型） ==========
        # 分类任务collator（Long标签）
        def cls_data_collator(features):
            labels = []
            for f in features:
                if "labels" in f:
                    label_tensor = f["labels"]
                    # 修复张量复制警告
                    if isinstance(label_tensor, torch.Tensor):
                        label = label_tensor.clone().detach().long()
                    else:
                        label = torch.tensor(label_tensor, dtype=torch.long)
                    labels.append(label)
                    del f["labels"]
            
            collator = DataCollatorWithPadding(tokenizer=tokenizer)
            batch = collator(features)
            if labels:
                batch["labels"] = torch.stack(labels)
            return batch
        
        # 回归任务collator（Float标签）
        def reg_data_collator(features):
            labels = []
            for f in features:
                if "labels" in f:
                    label_tensor = f["labels"]
                    if isinstance(label_tensor, torch.Tensor):
                        label = label_tensor.clone().detach().float()
                    else:
                        label = torch.tensor(label_tensor, dtype=torch.float)
                    labels.append(label)
                    del f["labels"]
            
            collator = DataCollatorWithPadding(tokenizer=tokenizer)
            batch = collator(features)
            if labels:
                batch["labels"] = torch.stack(labels)
            return batch
        
        # ========== 2. 模型初始化（核心修改：扩展为3类立场） ==========
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(
            config.BACKBONE_MODEL,
            num_labels=config.NUM_LABELS_STANCE,  # 需确保config里NUM_LABELS_STANCE=3
            # 核心修改：添加Neutral（中立）类别，id=2
            id2label={0: "Affirmative", 1: "Negative", 2: "Neutral"},
            label2id={"Affirmative": 0, "Negative": 1, "Neutral": 2},
            local_files_only=True
        ).to(config.DEVICE)
        
        self.regression_model = AutoModelForSequenceClassification.from_pretrained(
            config.BACKBONE_MODEL,
            num_labels=1,  
            problem_type="regression",  
            local_files_only=True
        ).to(config.DEVICE)
        
        # ========== 3. 训练参数（无需修改，仅需确保config.EPOCHS=1） ==========
        self.cls_training_args = TrainingArguments(
            output_dir=os.path.join(config.CHECKPOINT_DIR, "stance_classification"),
            learning_rate=2e-5,  
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            num_train_epochs=config.EPOCHS,  
            weight_decay=0.1,  
            warmup_ratio=0.1,
            logging_dir=os.path.join(config.LOG_DIR, "stance_classification"),
            logging_steps=config.EVALUATE_EVERY_STEP,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # 多分类下f1仍适用（weighted f1）
            fp16=config.DEVICE == "cuda",
            push_to_hub=False,
            report_to="none"
        )
        
        self.reg_training_args = TrainingArguments(
            output_dir=os.path.join(config.CHECKPOINT_DIR, "stance_regression"),
            learning_rate=5e-6,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            num_train_epochs=config.EPOCHS,
            weight_decay=0.05,
            warmup_ratio=0.1,
            logging_dir=os.path.join(config.LOG_DIR, "stance_regression"),
            logging_steps=config.EVALUATE_EVERY_STEP,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="r2",
            fp16=config.DEVICE == "cuda",
            push_to_hub=False,
            report_to="none"
        )
        
        # ========== 4. 训练器（无需修改） ==========
        self.cls_trainer = Trainer(
            model=self.classification_model,
            args=self.cls_training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_stance_metrics,
            data_collator=cls_data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
        )
        
        self.reg_trainer = Trainer(
            model=self.regression_model,
            args=self.reg_training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_strength_metrics,
            data_collator=reg_data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)]
        )
    
    # ========== 5. Token截断（无需修改） ==========
    def _fix_token_length(self, dataset):
        """修复Token长度超过512的问题（核心：truncation=True）"""
        def tokenize_truncate(examples):
            text_col = "fulltext" if "fulltext" in examples else "text"
            return tokenizer(
                examples[text_col],
                truncation=True,  
                padding="max_length",
                max_length=512,
                return_overflowing_tokens=False
            )
        if "input_ids" in dataset.features and len(dataset[0]["input_ids"]) > 512:
            dataset = dataset.map(tokenize_truncate, batched=True)
        return dataset
    
    # ========== 6. 训练（无需修改，数据集需包含中立标签id=2） ==========
    def train(self, train_dataset, val_dataset):
        train_dataset = self._fix_token_length(train_dataset)
        val_dataset = self._fix_token_length(val_dataset)
        
        self.cls_trainer.train_dataset = train_dataset
        self.cls_trainer.eval_dataset = val_dataset
        resume_cls_path = os.path.join(config.CHECKPOINT_DIR, "stance_classification") if config.INCREMENTAL_TRAINING else None
        self.cls_trainer.train(resume_from_checkpoint=resume_cls_path)
        
        def replace_to_strength(examples):
            examples["labels"] = [float(s) if s is not None else 0.0 for s in examples["strength_score"]]
            return examples
        
        train_dataset_reg = train_dataset.map(replace_to_strength, batched=True)
        val_dataset_reg = val_dataset.map(replace_to_strength, batched=True)
        
        self.reg_trainer.train_dataset = train_dataset_reg
        self.reg_trainer.eval_dataset = val_dataset_reg
        resume_reg_path = os.path.join(config.CHECKPOINT_DIR, "stance_regression") if config.INCREMENTAL_TRAINING else None
        self.reg_trainer.train(resume_from_checkpoint=resume_reg_path)
    
    # ========== 7. 评估+保存（无需修改） ==========
    def evaluate(self, test_dataset):
        test_dataset = self._fix_token_length(test_dataset)
        test_dataset_reg = test_dataset.map(lambda x: {"labels": [float(s) if s is not None else 0.0 for s in x["strength_score"]]}, batched=True)
        
        cls_metrics = self.cls_trainer.evaluate(eval_dataset=test_dataset)
        reg_metrics = self.reg_trainer.evaluate(eval_dataset=test_dataset_reg)
        
        print(f"立场检测评估结果：准确率={cls_metrics['eval_accuracy']:.4f}，F1={cls_metrics['eval_f1']:.4f}")
        print(f"立场强度估计评估结果：MSE={reg_metrics['eval_mse']:.4f}，R2={reg_metrics['eval_r2']:.4f}")
        return {"classification": cls_metrics, "regression": reg_metrics}
    
    def save_model(self, path=None):
        cls_save_path = path or os.path.join(config.OUTPUT_DIR, "stance_classification_model")
        reg_save_path = path or os.path.join(config.OUTPUT_DIR, "stance_regression_model")
        
        self.classification_model.save_pretrained(cls_save_path)
        self.regression_model.save_pretrained(reg_save_path)
        tokenizer.save_pretrained(cls_save_path)
        tokenizer.save_pretrained(reg_save_path)
        
        print(f"立场检测模型保存至：{cls_save_path}")
        print(f"立场强度模型保存至：{reg_save_path}")