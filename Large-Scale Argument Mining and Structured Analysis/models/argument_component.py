from config import config
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from utils.metrics import compute_arg_component_metrics
from utils.text_processing import tokenizer
import os

class ArgumentComponentModel:
    def __init__(self, data_collator):
        # 加载预训练模型+序列标注头
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.BACKBONE_MODEL,
            num_labels=config.NUM_LABELS_ARG_COMPONENT,
            id2label={0: "claim", 1: "premise", 2: "warrant"},
            label2id={"claim": 0, "premise": 1, "warrant": 2},
            local_files_only=True
        ).to(config.DEVICE)
        
        # 训练参数（轻量化）
        self.training_args = TrainingArguments(
            output_dir=os.path.join(config.CHECKPOINT_DIR, "arg_component"),
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            num_train_epochs=config.EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            logging_dir=os.path.join(config.LOG_DIR, "arg_component"),
            logging_steps=config.EVALUATE_EVERY_STEP,
            evaluation_strategy="epoch",  # 改为与保存策略一致（epoch）
            eval_steps=None,  # 评估策略为epoch时，无需指定steps
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=config.DEVICE == "cuda",
            push_to_hub=False,
            report_to="none"
        )
        
        # 训练器
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_arg_component_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    
    def train(self, train_dataset, val_dataset):
        """训练模型，支持增量续训"""
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = val_dataset
        # 增量训练：从最近checkpoint续训
        resume_path = os.path.join(config.CHECKPOINT_DIR, "argument_component") if config.INCREMENTAL_TRAINING else None
        self.trainer.train(resume_from_checkpoint=resume_path)
    
    def evaluate(self, test_dataset):
        """评估模型"""
        metrics = self.trainer.evaluate(eval_dataset=test_dataset)
        print(f"论证单元识别评估结果：F1={metrics['eval_f1']:.4f}")
        return metrics
    
    def save_model(self, path=None):
        """保存模型"""
        save_path = path or os.path.join(config.OUTPUT_DIR, "argument_component_model")
        self.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"模型保存至：{save_path}")