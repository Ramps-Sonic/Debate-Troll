import logging
import json
import re
from autogen import ConversableAgent
# 从 agentchat 子模块导入
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.agentchat.groupchat import GroupChatManager

# 导入自定义模型客户端所需的库
from types import SimpleNamespace
from typing import List, Dict, Union
from llama_cpp import Llama 

# -----------------------------
# 导入 StrategyPlanner 相关依赖
# -----------------------------
from config import StrategyPlannerConfig
from schema import StrategyRequest, FallacySignal
from strategy_planner import StrategyPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 核心修复 1: 基于栈的鲁棒 JSON 提取器
# ==============================================================================
def extract_json_with_stack(text: str) -> dict:
    """
    使用堆栈逻辑从文本中提取最外层的 JSON 对象。
    这比正则表达式更可靠，能处理嵌套的大括号。
    """
    text = text.strip()
    
    # 尝试寻找第一个 {
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No '{' found")
    
    stack = []
    json_str = ""
    
    # 从第一个 { 开始遍历
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == '{':
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
        
        # 记录当前字符
        json_str += char
        
        # 如果栈空了，说明找到了闭合的最外层对象
        if not stack:
            try:
                # 尝试解析提取到的片段
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 如果解析失败（比如内部有语法错误），继续尝试找下一个闭合
                raise ValueError("Found matching braces but content is invalid JSON")
                
    raise ValueError("Unbalanced braces or invalid JSON")

# ---------------------------------------------
# 1. 定义 Llama.cpp Custom Model Client
# ---------------------------------------------
class LlamaCppClient:
    """遵循 Autogen ModelClient 协议的自定义客户端"""
    RESPONSE_USAGE_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(self, config: Dict, **kwargs):
        model_path = config.get("model_path")
        self.model_name = config.get("model", "llama-3-8b-local")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        
        self.llama = Llama(
            model_path=model_path, 
            n_ctx=self.max_tokens * 4, # 加大上下文窗口防止截断
            n_gpu_layers=-1,
            verbose=False
        )
        print(f"✅ LlamaCppClient initialized with model: {model_path}")

    def create(self, params: Dict) -> SimpleNamespace:
        messages = params.get("messages", [])
        
        # 构建 Prompt
        prompt = self._messages_to_prompt(messages)
        
        try:
            # 统一使用 completion 接口以获得更稳定的控制
            response_data = self.llama.create_completion(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=params.get("max_tokens", self.max_tokens),
                stop=params.get("stop", ["<|eot_id|>"]), # 确保及时停止
            )
        except Exception as e:
            print(f"Llama inference failed: {e}")
            return SimpleNamespace(choices=[], model=self.model_name, usage={})

        # 封装响应
        response = SimpleNamespace()
        response.choices = []
        response.model = self.model_name
        
        content = response_data['choices'][0]['text']
        choice = SimpleNamespace(message=SimpleNamespace(content=content, role='assistant'))
        response.choices.append(choice)
        response.usage = response_data.get('usage', {})
        return response

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Llama 3 标准 Prompt 格式"""
        prompt = ""
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def message_retrieval(self, response: SimpleNamespace) -> Union[List[str], List[SimpleNamespace]]:
        return [choice.message for choice in response.choices]

    def cost(self, response: SimpleNamespace) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: SimpleNamespace) -> Dict:
        return {}

# ---------------------------------------------
# 1.1 核心修复 2: 强壮的适配器
# ---------------------------------------------
class LocalLLMAdapterForPlanner:
    def __init__(self, autogen_client: LlamaCppClient):
        self.client = autogen_client

    def create_completion(self, messages, **kwargs):
        params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 800), # 限制生成长度，降低错误率
            "temperature": kwargs.get("temperature", 0.6)
        }
        response = self.client.create(params)
        raw_content = response.choices[0].message.content
        
        # --- 这里的逻辑保证了 StrategyPlanner 绝对不会因为 JSON 格式而崩溃 ---
        try:
            # 1. 尝试使用栈提取器清洗数据
            json_obj = extract_json_with_stack(raw_content)
            # 成功清洗，重新打包成标准字符串
            return json.dumps(json_obj, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON 提取失败 ({e})，正在使用兜底策略。原始输出片段: {raw_content[:50]}...")
            
            # 2. 构造兜底 JSON (必须包含 StrategyPlanner 需要的所有字段)
            # StrategyPlanner 通常需要 'plan', 'analysis', 'scores', 'rationale'
            fallback_json = {
                "plan": f"（系统自动生成）由于模型输出格式异常，无法生成针对性策略。建议直接指出对方的逻辑漏洞：{messages[-1].get('content', '')[-50:]}...",
                "analysis": "模型输出无法解析为标准 JSON。",
                "scores": {"feasibility": 5, "effectiveness": 5},
                "rationale": "JSON Parsing Error Fallback"
            }
            return json.dumps(fallback_json, ensure_ascii=False)

    def chat(self, messages, **kwargs):
        return self.create_completion(messages, **kwargs)

# ---------------------------------------------
# 2. Autogen 配置
# ---------------------------------------------
llm_config = {
    "temperature": 0.7,
    "config_list": [
        {
            "model": "llama-3-8b-local", 
            "model_client_cls": "LlamaCppClient",
            "model_path": "C:/Users/xing5/Downloads/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
            "max_tokens": 1024,
        }
    ],
}

# 实例化 Client 和 Adapter
llama_client_instance = LlamaCppClient(config=llm_config["config_list"][0])
planner_adapter = LocalLLMAdapterForPlanner(llama_client_instance)

# -----------------------------
# 3. 定义 Agent
# -----------------------------

analyzer = ConversableAgent(
    name="argument_analyzer",
    system_message="你是一个专业的论证分析师。请提取用户论点、前提和推理结构。**请务必全程使用中文回答**。",
    llm_config=llm_config,
)

critic_system_prompt = """
你是一个逻辑批判专家。你负责指出逻辑谬误。
**必须**且**只能**输出 JSON 格式。不要输出任何开头或结尾的废话。
格式：
{
    "fallacy_type": "谬误名称",
    "confidence": 0.9,
    "reasoning": "简短理由"
}
"""

critic = ConversableAgent(
    name="logic_critic",
    system_message=critic_system_prompt,
    llm_config=llm_config,
)

# 3.3 Strategy Planner Agent
planner_agent = ConversableAgent(
    name="strategy_planner",
    system_message="负责生成反驳策略。",
    llm_config=False, 
)

# 初始化 Planner
planner_config = StrategyPlannerConfig(
    backend="local", 
    api_key="dummy", 
    model_name="llama-3",
    verbose=True 
)
strategy_logic = StrategyPlanner(planner_config, planner_adapter)

def planner_reply_func(recipient, messages, sender, config):
    last_message = messages[-1].get("content", "")
    print(f"\n[Debug] Critic Output: {last_message[:100]}...\n")

    # 1. 解析 Critic JSON
    try:
        critic_data = extract_json_with_stack(last_message)
        fallacy_type = critic_data.get("fallacy_type", "通用谬误")
        confidence = float(critic_data.get("confidence", 0.5))
    except Exception:
        fallacy_type = "通用逻辑漏洞"
        confidence = 0.5

    # 2. 构造请求
    # 尝试获取 analyzer 的内容作为原文
    opponent_text = "对方使用了错误的逻辑。" 
    for msg in reversed(messages):
        if msg.get("name") == "argument_analyzer":
            opponent_text = msg.get("content")
            break
            
    req = StrategyRequest(
        text=opponent_text,
        fallacy=FallacySignal(fallacy_type=fallacy_type, confidence=confidence),
        user_goal="win_debate",
        context=str([m['content'] for m in messages[-3:]]) 
    )

    # 3. 调用 Planner (现在它是安全的)
    print(f"🔄 StrategyPlanner 正在生成策略... (类型: {fallacy_type})")
    try:
        plan_result = strategy_logic.plan(req)
        
        # 安全获取属性
        best_plan = getattr(plan_result, "plan", "无法生成策略")
        rationale = getattr(plan_result, "rationale", "")
        
        reply_text = (
            f"【检测谬误】{fallacy_type} (置信度 {confidence})\n"
            f"【策略建议】\n{best_plan}\n"
            f"【理由】\n{rationale}"
        )
        return True, reply_text
    except Exception as e:
        # 这几乎不可能发生了，因为 adapter 已经兜底了
        return True, f"策略生成异常: {str(e)}"

planner_agent.register_reply([ConversableAgent, None], planner_reply_func)

# 3.4 Generator
generator = ConversableAgent(
    name="counter_generator",
    system_message="你负责根据前一位 StrategyPlanner 的策略建议，生成一段犀利的中文反驳。输出后请单独一行写 'TERMINATE'。",
    is_termination_msg=lambda x:"TERMINATE" in (x.get("content","") or "").upper(),
    llm_config=llm_config,
)



claim = """
人类不需要减少碳排放，因为几百年来地球一直会自己调节气候。
即使我们什么都不做，气候变化最终也会自己稳定下来。
所以所有应对气候变化的政策都是浪费钱。
"""


# -----------------------------
# 4. 注册与运行
# -----------------------------
def register_custom_client(agent):
    agent.register_model_client(model_client_cls=LlamaCppClient)

register_custom_client(analyzer)
register_custom_client(critic)
register_custom_client(generator)

# 1. 先创建 GroupChat (包含所有 agent)
# 注意：你需要手动导入 GroupChat
from autogen.agentchat.groupchat import GroupChat 

group_chat = GroupChat(
    agents=[analyzer, critic, planner_agent, generator],
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin" # 显式指定轮询，替代 AutoPattern
)

# 2. 再创建 GroupChatManager (直接传入 group_chat)
debate_manager = GroupChatManager(
    name="debate_manager",
    groupchat=group_chat, # 在这里直接传入！不要用 = None
    llm_config=False,     # 禁用 Manager 的 LLM
)

# 3. 开始运行
print("🚀 开始运行多 Agent 辩论 (集成 StrategyPlanner)...")

debate_manager.initiate_chat(
    recipient=generator, # 对于 GroupChatManager，recipient 通常填群里的任意一个 agent 即可
    message=claim,
)