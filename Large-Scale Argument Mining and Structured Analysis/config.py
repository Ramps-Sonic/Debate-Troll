import os
from dataclasses import dataclass
import torch

@dataclass
class Config:
    # 基础路径（请你后续修改为实际数据集路径）
    DATASET_PATH = "./data/evidence-2014.csv"  # 数据集路径
    ARGKP_DATASET_PATH = "./data/ArgKP"  # ArgKP数据集路径（后续可配置）
    OUTPUT_DIR = "./output"  # 模型/日志输出路径
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # 增量续训 checkpoint
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

    # 数据集配置（基于readme列名，间接构建标注信号）
    TEXT_COL = "fulltext"  # 核心输入文本列
    ARG_COMPONENT_SIGNAL_COLS = ["tag", "summary", "spoken"]  # 论证单元识别信号列
    STANCE_LABEL_COL = "side"  # 立场标签列（Affirmative/Negative）
    STRENGTH_SIGNAL_COLS = ["report", "textlength"]  # 立场强度信号列
    QUALITY_SIGNAL_COLS = ["cite", "report", "duplicateCount"]  # 质量评估信号列
    RELATION_SIGNAL_COLS = ["roundId", "caselistId"]  # 论证关系信号列

    # 模型配置（轻量化，适配本地CPU/GPU）
    #BACKBONE_MODEL = "roberta-base"  # 轻量骨干模型（比DeBERTa-v3更省资源）
    MAX_SEQ_LENGTH = 512  # 序列最大长度，是roberta的预训练长度
    WINDOW_SIZE = 512  # 长文档滑动窗口大小
    STEP_SIZE = 256  # 滑动窗口步长（重叠50%，保证上下文连贯）
    NUM_LABELS_ARG_COMPONENT = 3  # claim/premise/warrant
    NUM_LABELS_STANCE = 3  # 仅支持/反对/中立

    # 训练配置（轻量化，本地验证用）
    DATA_SAMPLE_RATIO = 0.01  # 仅用1%数据训练
    BATCH_SIZE = 8  # 小批量，适配本地显存/内存
    LEARNING_RATE = 2e-5
    EPOCHS = 3  # 总轮次，即针对整个训练集训练的轮次，在小批次训练时可以调整
    WEIGHT_DECAY = 1e-4
    WARMUP_RATIO = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备
    CUDA_DEVICE_ID = 0  # 单GPU默认编号
    INCREMENTAL_TRAINING = False  # 支持基于checkpoint续训
    SAVE_CHECKPOINT_EVERY_EPOCH = True
    EVALUATE_EVERY_STEP = 500  # 每50步评估一次（快速反馈）

    # 评估配置
    REQUIRED_F1_SCORE = 0.75
    METRIC_TYPE = "micro"

    LOCAL_MODEL_PATH = "./models/roberta-base"
    BACKBONE_MODEL = LOCAL_MODEL_PATH

    NUM_KEY_POINT_CLUSTERS = 5

# 初始化配置
config = Config()

# 创建输出目录（不存在则自动创建）
for dir_path in [config.OUTPUT_DIR, config.CHECKPOINT_DIR, config.LOG_DIR, config.VISUALIZATION_DIR]:
    os.makedirs(dir_path, exist_ok=True)