# -*- coding: utf-8 -*-
"""
conversation_knowledge_extractor.py
-----------------------------------

A module for extracting, consolidating, and accumulating structured knowledge
from user-AI conversations.

Main Stages:
1. Extract knowledge from raw dialogues using LLM
2. Batch process multiple conversations
3. Filter & merge similar knowledge items via LLM
4. Generate a clean, structured knowledge base for indexing (BM25 / Vector DB)

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

# ============================================================
# 🔧 Import Dependencies
# ============================================================
import os
import json
from collections import Counter
from datetime import datetime

# Import shared utility functions
from utils.llm_utils import get_completion, preprocess_json_response


# ============================================================
# 🧠 Class: ConversationKnowledgeExtractor
# ============================================================
class ConversationKnowledgeExtractor:
    """
    A high-level class for extracting structured knowledge from user-AI conversations,
    merging similar facts, and maintaining a persistent knowledge base.
    """

    # --------------------------------------------------------
    # 🧩 Initialization
    # --------------------------------------------------------
    def __init__(self, model: str = "qwen-turbo-latest"):
        """
        Initialize the knowledge extractor.

        Args:
            model (str): The LLM model name for extraction and merging.
        """
        self.model = model
        self.extracted_knowledge = []  # All raw knowledge points
        self.knowledge_frequency = (
            Counter()
        )  # Frequency counter for occurrence statistics

    # --------------------------------------------------------
    # 📘 Step 1: Extract Knowledge from a Single Conversation
    # --------------------------------------------------------
    def extract_knowledge_from_conversation(self, conversation: str) -> dict:
        """
        Extract structured knowledge from a single conversation using the LLM.

        Args:
            conversation (str): The full text of the user-AI conversation.

        Returns:
            dict: A structured JSON object containing extracted knowledge items,
                  a conversation summary, and user intent.
        """

        # 1️⃣ Instruction prompt for the model
        instruction = """
        你是一个专业的知识提取专家。请从给定的对话中提取有价值的知识点，包括：
        1. 事实性信息（地点、时间、价格、规则等）
        2. 用户需求和偏好
        3. 常见问题和解答
        4. 操作流程和步骤
        5. 注意事项和提醒

        请返回JSON格式：
        {
            "extracted_knowledge": [
                {
                    "knowledge_type": "知识类型（事实/需求/问题/流程/注意）",
                    "content": "知识内容",
                    "confidence": "置信度(0-1)",
                    "source": "来源（用户/AI/对话）",
                    "keywords": ["关键词1", "关键词2"],
                    "category": "分类"
                }
            ],
            "conversation_summary": "对话摘要",
            "user_intent": "用户意图"
        }
        """

        # 2️⃣ Construct final prompt
        prompt = f"""
        ### 指令 ###
        {instruction}

        ### 对话内容 ###
        {conversation}

        ### 提取结果 ###
        """

        # 3️⃣ Call LLM
        response = get_completion(prompt, self.model)

        # 4️⃣ Clean and parse JSON
        response = preprocess_json_response(response)
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"[⚠️ JSON解析失败] {e}")
            print(f"AI返回内容: {response[:200]}...")
            return {
                "extracted_knowledge": [],
                "conversation_summary": "无法解析对话",
                "user_intent": "未知",
            }

    # --------------------------------------------------------
    # 📚 Step 2: Batch Process Multiple Conversations
    # --------------------------------------------------------
    def batch_extract_knowledge(self, conversations: list) -> list:
        """
        Process multiple conversations and aggregate all extracted knowledge.

        Args:
            conversations (list): A list of conversation strings.

        Returns:
            list: Aggregated list of extracted knowledge items.
        """
        all_knowledge = []

        for i, conversation in enumerate(conversations):
            print(f"\n[对话 {i+1}/{len(conversations)}] 开始提取知识...")
            result = self.extract_knowledge_from_conversation(conversation)
            extracted = result.get("extracted_knowledge", [])

            # Collect and record frequency
            all_knowledge.extend(extracted)
            for item in extracted:
                key = f"{item.get('knowledge_type')}:{item.get('content')[:50]}"
                self.knowledge_frequency[key] += 1

        print(f"\n✅ 批量提取完成，共获得 {len(all_knowledge)} 条知识。")
        return all_knowledge

    # --------------------------------------------------------
    # 🧩 Step 3: Merge Similar Knowledge Items
    # --------------------------------------------------------
    def merge_similar_knowledge(self, knowledge_list: list) -> list:
        """
        Merge similar or duplicate knowledge items by calling LLM for consolidation.

        Args:
            knowledge_list (list): List of extracted knowledge entries.

        Returns:
            list: Consolidated list of high-quality, deduplicated knowledge entries.
        """

        # 1️⃣ Filter out non-persistent types (需求 / 问题)
        filtered_knowledge = [
            k for k in knowledge_list if k.get("knowledge_type") not in ["需求", "问题"]
        ]
        print("\n--- 知识过滤阶段 ---")
        print(f"原始知识数: {len(knowledge_list)}")
        print(f"过滤后知识数: {len(filtered_knowledge)}")

        # 2️⃣ Group by knowledge type
        knowledge_by_type = {}
        for k in filtered_knowledge:
            knowledge_type = k.get("knowledge_type", "其他")
            knowledge_by_type.setdefault(knowledge_type, []).append(k)

        # 3️⃣ Merge each group
        merged_knowledge = []
        print("\n--- 合并阶段 ---")
        for knowledge_type, group in knowledge_by_type.items():
            if len(group) == 1:
                merged_knowledge.append(group[0])
            else:
                merged = self.merge_knowledge_with_llm(group, knowledge_type)
                merged_knowledge.append(merged)

        print(f"✅ 合并完成，共输出 {len(merged_knowledge)} 条最终知识。")
        return merged_knowledge

    # --------------------------------------------------------
    # 🧠 Step 4: LLM-based Merging Logic
    # --------------------------------------------------------
    def merge_knowledge_with_llm(
        self, knowledge_group: list, knowledge_type: str
    ) -> dict:
        """
        Use the LLM to intelligently merge a group of similar knowledge items.

        Args:
            knowledge_group (list): List of similar knowledge items.
            knowledge_type (str): The shared knowledge type of the group.

        Returns:
            dict: Merged knowledge entry.
        """
        # Prepare textual summary for the LLM
        knowledge_contents = []
        all_keywords = set()
        all_sources = []

        for i, k in enumerate(knowledge_group, 1):
            content = k.get("content", "")
            confidence = k.get("confidence", 0.5)
            category = k.get("category", "")
            source = k.get("source", "")
            keywords = k.get("keywords", [])

            knowledge_contents.append(f"{i}. 内容: {content}")
            knowledge_contents.append(f"   置信度: {confidence}")
            knowledge_contents.append(f"   分类: {category}")
            knowledge_contents.append(f"   来源: {source}")
            knowledge_contents.append(f"   关键词: {', '.join(keywords)}\n")

            all_keywords.update(keywords)
            if source and source not in all_sources:
                all_sources.append(source)

        # Build merge prompt
        prompt = f"""
        你是一个专业的知识整理专家。请将以下 {knowledge_type} 类型的知识点进行智能合并，生成一个更完整、准确的知识点。

        ### 合并要求：
        1. 保留所有重要信息，避免信息丢失
        2. 消除重复内容，整合相似表述
        3. 提高内容的准确性和完整性
        4. 保持逻辑清晰，结构合理
        5. 合并后的置信度取所有知识点中的最高值

        ### 待合并的知识点：
        {chr(10).join(knowledge_contents)}

        ### 请返回JSON格式：
        {{
            "knowledge_type": "{knowledge_type}",
            "content": "合并后的知识内容",
            "confidence": 最高置信度值,
            "keywords": ["合并后的关键词列表"],
            "category": "合并后的分类",
            "sources": ["所有来源"],
            "frequency": {len(knowledge_group)}
        }}
        """

        # Call LLM
        response = get_completion(prompt, self.model)
        response = preprocess_json_response(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[⚠️ 合并JSON解析失败] {e}")
            print(f"AI返回内容: {response[:200]}...")
            # Fallback: choose the highest-confidence item
            best = max(knowledge_group, key=lambda x: x.get("confidence", 0))
            return {
                "knowledge_type": knowledge_type,
                "content": best.get("content", ""),
                "confidence": best.get("confidence", 0.5),
                "keywords": list(all_keywords),
                "category": best.get("category", "未分类"),
                "sources": all_sources,
                "frequency": len(knowledge_group),
            }


# ============================================================
# 🚀 Demonstration (Executed when run directly)
# ============================================================
def main():
    """Demonstration of conversation knowledge extraction and merging."""
    extractor = ConversationKnowledgeExtractor()

    print("=== 🧩 对话知识提取与沉淀示例（上海迪士尼） ===\n")

    # ----------------------------------------
    # 🗨️ 示例对话数据
    # ----------------------------------------
    sample_conversations = [
        """
        用户: 我想去上海迪士尼乐园玩，门票多少钱？
        AI: 上海迪士尼乐园平日成人票399元，周末499元。
        用户: 从浦东机场怎么去？
        AI: 可乘坐地铁2号线到广兰路站换乘11号线到迪士尼站，约1小时。
        """,
        """
        用户: 迪士尼乐园今天开放吗？
        AI: 每天开放8:00至20:00，但建议查看官网确认是否有调整。
        """,
        """
        用户: 带小孩去迪士尼需要注意什么？
        AI: 注意防晒、身高限制、携带零食、下载APP查看排队时间。
        """,
    ]

    # ----------------------------------------
    # 🔹 示例1：单次提取
    # ----------------------------------------
    print("\n--- 示例1：从单次对话中提取知识 ---")
    result = extractor.extract_knowledge_from_conversation(sample_conversations[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ----------------------------------------
    # 🔹 示例2：批量提取
    # ----------------------------------------
    print("\n--- 示例2：批量提取知识 ---")
    all_knowledge = extractor.batch_extract_knowledge(sample_conversations)
    print(f"共提取 {len(all_knowledge)} 条知识。")

    # ----------------------------------------
    # 🔹 示例3：知识合并
    # ----------------------------------------
    print("\n--- 示例3：合并相似知识 ---")
    merged = extractor.merge_similar_knowledge(all_knowledge)
    print(json.dumps(merged, ensure_ascii=False, indent=2))


# ============================================================
# 🏁 Entry Point
# ============================================================
if __name__ == "__main__":
    main()
