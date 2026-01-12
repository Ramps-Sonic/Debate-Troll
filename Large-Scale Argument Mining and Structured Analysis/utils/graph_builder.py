from config import config
from pyvis.network import Network
import os

class ArgumentDependencyGraph:
    def __init__(self):
        self.net = Network(directed=True, height="800px", width="100%", bgcolor="#222222", font_color="white")
        self.node_id = 0  # 节点唯一ID
        self.node_map = {}  # 文本->节点ID映射
    
    def add_node(self, text: str, node_type: str) -> int:
        """添加节点（node_type: claim/premise/warrant）"""
        if text in self.node_map:
            return self.node_map[text]
        
        # 节点颜色：claim=红色，premise=蓝色，warrant=绿色
        color_map = {"claim": "#ff4444", "premise": "#4444ff", "warrant": "#44ff44"}
        color = color_map.get(node_type, "#ffffff")
        
        self.net.add_node(self.node_id, label=text[:50] + "..." if len(text) > 50 else text, color=color)
        self.node_map[text] = self.node_id
        self.node_id += 1
        return self.node_id - 1
    
    def add_edge(self, source_text: str, target_text: str, relation_type: str = "supports"):
        """添加边（关系：支持/反对/依赖）"""
        source_id = self.add_node(source_text, self._get_node_type(source_text))
        target_id = self.add_node(target_text, self._get_node_type(target_text))
        
        # 边颜色：支持=绿色，反对=红色，依赖=灰色
        color_map = {"supports": "#00ff00", "opposes": "#ff0000", "depends": "#888888"}
        color = color_map.get(relation_type, "#888888")
        
        self.net.add_edge(source_id, target_id, label=relation_type, color=color)
    
    def _get_node_type(self, text: str) -> str:
        """根据文本内容推断节点类型（简化版）"""
        # 实际应使用论证单元识别模型预测，此处为简化版
        if any(keyword in text.lower() for keyword in ["argue", "claim", "assert"]):
            return "claim"
        elif any(keyword in text.lower() for keyword in ["evidence", "data", "study", "source"]):
            return "premise"
        else:
            return "warrant"
    
    def visualize(self, output_filename: str = "argument_graph.html"):
        """保存可视化图表（HTML文件）"""
        output_path = os.path.join(config.VISUALIZATION_DIR, output_filename)
        self.net.show(output_path, notebook=False)
        print(f"论证依赖图已保存至：{output_path}")