from config import config
from data_loader import prepare_dataloaders
from models.argument_component import ArgumentComponentModel
from models.stance_detection import StanceDetectionModel
from models.key_point_analysis import KeyPointAnalysis
from models.argument_quality import ArgumentQualityModel

class JointTrainer:
    def __init__(self):
        # 加载数据
        self.dataloaders = prepare_dataloaders()
        self.tc_data = self.dataloaders["token_classification"]  # 序列标注数据（论证单元识别）
        self.cls_data = self.dataloaders["classification"]  # 分类/回归数据（立场/质量）
        
        # 初始化所有模型
        self.arg_component_model = ArgumentComponentModel(data_collator=self.tc_data[3])
        self.stance_model = StanceDetectionModel()
        self.kpa_model = KeyPointAnalysis()
        self.quality_model = ArgumentQualityModel(data_collator=self.cls_data[3])
    
    def train_all(self):
        """训练所有模块"""
        print("="*50)
        print("开始训练论证单元识别模型...")
        self.arg_component_model.train(self.tc_data[0], self.tc_data[1])
        
        print("\n" + "="*50)
        print("开始训练立场检测与强度估计模型...")
        self.stance_model.train(self.cls_data[0], self.cls_data[1])
        
        print("\n" + "="*50)
        print("开始训练论证质量评估模型...")
        self.quality_model.train(self.cls_data[0], self.cls_data[1])
        
        print("\n" + "="*50)
        #print("开始训练KPA关键点分析模型...")
        # KPA训练：基于训练集文本聚类
        self.kpa_model.train(self.cls_data[0]["input_ids"])
        
        print("\n所有模块训练完成！")
    
    def evaluate_all(self):
        """评估所有模块"""
        print("="*50)
        print("开始评估论证单元识别模型...")
        self.arg_component_model.evaluate(self.tc_data[2])
        
        print("\n" + "="*50)
        print("开始评估立场检测与强度估计模型...")
        self.stance_model.evaluate(self.cls_data[2])
        
        print("\n" + "="*50)
        print("开始评估论证质量评估模型...")
        self.quality_model.evaluate(self.cls_data[2])
        
        print("\n所有模块评估完成！")
    
    def save_all_models(self):
        """保存所有模型"""
        self.arg_component_model.save_model()
        self.stance_model.save_model()
        self.kpa_model.save_model()
        self.quality_model.save_model()
        print("\n所有模型保存完成！")