from config import config
from seqeval.metrics import f1_score, classification_report
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score as skf1

# 标签映射
ARG_COMPONENT_ID2LABEL = {0: "claim", 1: "premise", 2: "warrant"}
STANCE_ID2LABEL = {0: "Affirmative", 1: "Negative"}

def compute_arg_component_metrics(eval_pred):
    """计算论证单元识别指标（F1分数）"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # 去除padding标签（-100）
    true_labels = [[ARG_COMPONENT_ID2LABEL[label] for label in seq if label != -100] for seq in labels]
    true_predictions = [[ARG_COMPONENT_ID2LABEL[pred] for pred, label in zip(seq, label_seq) if label != -100] 
                        for seq, label_seq in zip(predictions, labels)]
    
    f1 = f1_score(true_labels, true_predictions, average=config.METRIC_TYPE)
    return {"f1": f1, "report": classification_report(true_labels, true_predictions)}

def compute_stance_metrics(eval_pred):
    """计算立场检测指标（准确率+F1）"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = skf1(labels, predictions, average=config.METRIC_TYPE)
    return {"accuracy": accuracy, "f1": f1}

def compute_strength_metrics(eval_pred):
    """计算立场强度估计指标（MSE+R2）"""
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"mse": mse, "r2": r2}

def compute_quality_metrics(eval_pred):
    """
    适配Trainer的评估函数（仅接收EvalPrediction参数）
    计算4维质量评估的回归指标：每个维度的MSE和R2
    """
    # 从EvalPrediction中提取预测值和真实标签
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    # 确保数据类型正确（避免float32/float64不匹配）
    predictions = np.array(predictions, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    
    # 4个质量维度：逻辑、相关、充分、可信
    dimensions = ["logic", "relevance", "sufficiency", "credibility"]
    metrics = {}
    
    # 逐个维度计算MSE和R2
    for i, dim in enumerate(dimensions):
        # 提取该维度的预测值和标签
        pred_dim = predictions[:, i]
        label_dim = labels[:, i]
        
        # 计算指标（过滤掉全NaN的情况，避免报错）
        if not np.isnan(pred_dim).all() and not np.isnan(label_dim).all():
            metrics[f"{dim}_mse"] = mean_squared_error(label_dim, pred_dim)
            metrics[f"{dim}_r2"] = r2_score(label_dim, pred_dim)
        else:
            metrics[f"{dim}_mse"] = np.nan
            metrics[f"{dim}_r2"] = np.nan
    
    # 计算平均指标（可选，便于整体评估）
    metrics["avg_mse"] = np.mean([metrics[f"{dim}_mse"] for dim in dimensions])
    metrics["avg_r2"] = np.mean([metrics[f"{dim}_r2"] for dim in dimensions])
    
    return metrics