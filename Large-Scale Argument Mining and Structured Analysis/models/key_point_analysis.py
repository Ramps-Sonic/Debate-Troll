# key_point_analysis.py 完整代码
from config import config
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import torch  # 新增：处理Tensor类型input_ids
from utils.text_processing import tokenizer  # 导入tokenizer，用于解码input_ids

class KeyPointAnalysis:
    def __init__(self):
        # 加载SentenceTransformer模型（本地路径/远程下载）
        self.sentence_model = SentenceTransformer("./models/all-MiniLM-L6-v2")
        self.num_clusters = getattr(config, "NUM_KEY_POINT_CLUSTERS", 5)  # 兜底默认5个聚类
        self.cluster_model = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.key_point_embeddings = None  # 存储核心论点嵌入
        self.key_points = None  # 存储核心论点文本
        self.original_arguments = None  # 存储原始论点文本（新增）
        self.cluster_labels = None  # 存储原始论点的聚类标签（新增）

    def encode_texts(self, texts: list) -> np.ndarray:
        """文本编码为向量（基于SentenceTransformer）"""
        # 过滤空文本，避免编码报错
        texts = [text.strip() for text in texts if isinstance(text, str) and len(text.strip()) > 0]
        if not texts:
            raise ValueError("输入文本列表为空或全是空白字符")
        return self.sentence_model.encode(texts, convert_to_numpy=True)

    def decode_from_input_ids(self, input_ids_list: list) -> list:
        """从input_ids解码为原始文本（适配你的数据集）"""
        texts = []
        for input_ids in input_ids_list:
            try:
                # 确保input_ids是列表类型（兼容Tensor转来的列表）
                if isinstance(input_ids, (torch.Tensor, np.ndarray)):
                    input_ids = input_ids.tolist()
                # decode参数说明：
                # skip_special_tokens=True：跳过[CLS]/[SEP]/[PAD]等特殊token
                # clean_up_tokenization_spaces=True：清理多余空格
                text = tokenizer.decode(
                    input_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                texts.append(text)
            except Exception as e:
                print(f"解码input_ids失败：{e}，跳过该样本")
                texts.append("")
        # 过滤空文本
        texts = [t for t in texts if len(t.strip()) > 0]
        return texts

    def cluster_key_points(self, arguments: list) -> tuple:
        """论点聚类，提取核心维度（修复逻辑错误）"""
        # 保存原始论点，用于后续排序
        self.original_arguments = arguments
        
        # 编码所有论点
        embeddings = self.encode_texts(arguments)
        
        # 处理聚类数量大于文本数量的情况
        if len(arguments) < self.num_clusters:
            self.num_clusters = len(arguments)
            self.cluster_model = KMeans(n_clusters=self.num_clusters, random_state=42)
        
        # 聚类
        self.cluster_labels = self.cluster_model.fit_predict(embeddings)
        
        # 提取每个聚类的核心论点（与聚类中心最相似的文本）
        key_points = []
        key_point_embeddings = []
        for cluster_id in range(self.num_clusters):
            # 找到该聚类的所有嵌入和文本（修复索引逻辑）
            cluster_mask = self.cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_texts = [arguments[i] for i in range(len(arguments)) if cluster_mask[i]]
            
            if not cluster_texts:
                continue
            
            # 计算聚类中心
            center = cluster_embeddings.mean(axis=0)
            
            # 找到与中心最相似的文本作为核心论点
            similarities = cosine_similarity([center], cluster_embeddings)[0]
            best_idx = np.argmax(similarities)
            key_points.append(cluster_texts[best_idx])
            key_point_embeddings.append(cluster_embeddings[best_idx])
        
        self.key_points = key_points
        self.key_point_embeddings = np.array(key_point_embeddings) if key_point_embeddings else np.array([])
        return key_points, self.cluster_labels

    def calculate_coverage(self, argument: str) -> float:
        """计算论点对核心维度的覆盖率"""
        if self.key_point_embeddings is None or len(self.key_point_embeddings) == 0:
            raise ValueError("请先调用 cluster_key_points 提取核心论点")
        
        arg_embedding = self.encode_texts([argument])[0]
        similarities = cosine_similarity([arg_embedding], self.key_point_embeddings)[0]
        return np.mean(similarities)  # 平均相似度作为覆盖率

    def rank_key_points(self) -> list:
        """核心论点重要性排序（修复内聚度计算逻辑）"""
        if self.key_points is None or len(self.key_points) == 0:
            raise ValueError("请先调用 cluster_key_points 提取核心论点")
        if self.original_arguments is None:
            raise ValueError("缺少原始论点数据，无法计算排序得分")
        
        # 计算每个核心论点的重要性得分（聚类大小 + 内聚度）
        rankings = []
        embeddings = self.encode_texts(self.original_arguments)
        
        for i, key_point in enumerate(self.key_points):
            # 1. 聚类大小权重（该聚类的样本数 / 总样本数）
            cluster_size = np.sum(self.cluster_labels == i)
            size_score = cluster_size / len(self.original_arguments) if len(self.original_arguments) > 0 else 0
            
            # 2. 内聚度（聚类内所有样本与核心论点的平均相似度）
            cluster_mask = self.cluster_labels == i
            cluster_embeddings = embeddings[cluster_mask]
            if len(cluster_embeddings) == 0:
                cohesion_score = 0
            else:
                cohesion_score = np.mean(cosine_similarity([self.key_point_embeddings[i]], cluster_embeddings)[0])
            
            # 加权得分（大小60% + 内聚度40%）
            total_score = (size_score * 0.6) + (cohesion_score * 0.4)
            rankings.append((key_point, total_score))
        
        # 按得分降序排序
        rankings.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in rankings]

    def train(self, train_data, val_data=None):
        """补全train方法（适配trainer.py的调用，兼容Tensor类型input_ids）
        Args:
            train_data: 训练数据（可以是Tensor/列表/ndarray类型的input_ids，或原始文本列表）
            val_data: 验证数据（可选，KPA任务暂不做验证）
        """
        print("开始训练KPA关键点分析模型...")
        
        # 步骤1：处理输入数据（兼容Tensor/列表/ndarray类型的input_ids，或原始文本）
        if isinstance(train_data, torch.Tensor):
            # 如果是Tensor，先转成numpy数组，再转成Python列表（CPU避免CUDA错误）
            print("检测到Tensor类型的input_ids，转换为列表...")
            train_data = train_data.cpu().numpy().tolist()
        elif isinstance(train_data, np.ndarray):
            # 如果是numpy数组，直接转列表
            train_data = train_data.tolist()
        
        if isinstance(train_data[0], (list, np.ndarray, torch.Tensor)):
            # 如果是input_ids列表（嵌套列表/Tensor），先解码为文本
            print("检测到input_ids格式，正在解码为文本...")
            train_texts = self.decode_from_input_ids(train_data)
        else:
            # 如果是原始文本，直接过滤空文本
            train_texts = [text.strip() for text in train_data if isinstance(text, str) and len(text.strip()) > 0]
        
        if not train_texts:
            raise ValueError("训练文本为空，请检查数据格式")
        print(f"有效训练文本数量：{len(train_texts)}")
        
        # 步骤2：聚类提取核心论点（KPA的核心训练逻辑）
        print(f"开始聚类核心论点（聚类数量：{self.num_clusters}）...")
        key_points, cluster_labels = self.cluster_key_points(train_texts)
        print(f"提取到核心论点数量：{len(key_points)}")
        
        # 步骤3：核心论点排序
        print("开始排序核心论点...")
        ranked_key_points = self.rank_key_points()
        print("核心论点排序完成，前3个核心论点：")
        for i, point in enumerate(ranked_key_points[:3]):
            print(f"{i+1}. {point[:100]}...")
        
        # 步骤4：保存模型（可选，根据需要开启）
        # self.save_model()
        
        return {
            "key_points": key_points,
            "ranked_key_points": ranked_key_points,
            "cluster_labels": cluster_labels
        }

    def save_model(self, path=None):
        """保存核心论点模型和聚类结果"""
        save_path = path or os.path.join(config.OUTPUT_DIR, "key_point_analysis_model")
        os.makedirs(save_path, exist_ok=True)
        
        # 保存SentenceTransformer模型
        self.sentence_model.save(os.path.join(save_path, "sentence_transformer"))
        
        # 保存聚类模型和核心论点
        if self.key_point_embeddings is not None:
            np.save(os.path.join(save_path, "key_point_embeddings.npy"), self.key_point_embeddings)
        with open(os.path.join(save_path, "key_points.txt"), "w", encoding="utf-8") as f:
            for point in self.key_points:
                f.write(point + "\n")
        
        print(f"核心论点分析模型保存至：{save_path}")