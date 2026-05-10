"""Dialogue Agent — 多轮对话优化整合"""
from models.schemas import DialogueIn, DialogueOut


async def chat(dialogue: DialogueIn, current_integration_state: dict) -> DialogueOut:
    """
    Input:  { message, history, current_integration_state }
    Output: DialogueOut { reply, updated_decisions, affected_decision_ids }

    支持场景:
      - "为什么合并 A 和 B？" → 解释理由
      - "请保留 C" → 改 remove 为 keep
      - "把 X 和 Y 分开" → 拆分 merge
    """
    raise NotImplementedError
