# -*- coding: utf-8 -*-
"""
knowledge_base_health_checker.py
-------------------------------------

A module for evaluating the overall health of a knowledge base.

Responsibilities:
1. Detect missing knowledge coverage (content gaps)
2. Detect outdated or stale information
3. Detect conflicting or inconsistent knowledge
4. Compute overall "health score" with weighted metrics
5. Provide actionable recommendations for improvement

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

# ============================================================
# 🔧 Import Dependencies
# ============================================================
import json
from datetime import datetime
from utils.llm_utils import get_completion, preprocess_json_response


# ============================================================
# 🧠 Class: KnowledgeBaseHealthChecker
# ============================================================
class KnowledgeBaseHealthChecker:
    """
    A comprehensive health checker for evaluating knowledge base quality.
    Performs integrity, freshness, and consistency checks using LLM reasoning.
    """

    # --------------------------------------------------------
    # 🧩 Initialization
    # --------------------------------------------------------
    def __init__(self, model: str = "qwen-turbo-latest"):
        self.model = model

    # --------------------------------------------------------
    # 📘 Step 1: Check for Missing Knowledge
    # --------------------------------------------------------
    def check_missing_knowledge(self, knowledge_base, test_queries):
        """
        Check for missing or uncovered knowledge by comparing test queries
        against the existing knowledge base.
        """
        instruction = """
        你是一个知识库完整性检查专家。请分析给定的测试查询和知识库内容，
        判断知识库中是否缺少相关的知识。

        检查标准：
        1. 查询是否能在知识库中找到相关答案
        2. 知识是否完整、准确
        3. 是否覆盖了用户的主要需求
        4. 是否存在知识空白

        请返回JSON格式：
        {
            "missing_knowledge": [
                {
                    "query": "测试查询",
                    "missing_aspect": "缺少的知识方面",
                    "importance": "重要性（高/中/低）",
                    "suggested_content": "建议的知识内容",
                    "category": "知识分类"
                }
            ],
            "coverage_score": "覆盖率评分(0-1)",
            "completeness_analysis": "完整性分析"
        }
        """

        # Build summarized text from knowledge base
        knowledge_text = "\n".join(
            [
                f"ID: {chunk.get('id', 'unknown')} - {chunk.get('content', '')}"
                for chunk in knowledge_base
            ]
        )

        queries_text = "\n".join(
            [
                f"查询: {item['query']} | 期望答案: {item.get('expected_answer', '')}"
                for item in test_queries
            ]
        )

        prompt = f"""
        ### 指令 ###
        {instruction}

        ### 知识库内容 ###
        {knowledge_text}

        ### 测试查询 ###
        {queries_text}

        ### 分析结果 ###
        """

        response = get_completion(prompt, self.model)
        response = preprocess_json_response(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ Missing knowledge check failed.")
            return {
                "missing_knowledge": [],
                "coverage_score": 0,
                "completeness_analysis": "解析失败",
            }

    # --------------------------------------------------------
    # 📗 Step 2: Check for Outdated Knowledge
    # --------------------------------------------------------
    def check_outdated_knowledge(self, knowledge_base):
        """
        Identify potentially outdated or stale information.
        """
        instruction = """
        你是一个知识时效性检查专家。请分析给定的知识内容，判断是否存在过期或需要更新的信息。

        检查标准：
        1. 时间相关信息是否过期
        2. 价格信息是否最新
        3. 政策规则是否更新
        4. 活动信息是否仍有效
        5. 联系方式是否准确
        6. 技术信息是否过时

        请返回JSON格式：
        {
            "outdated_knowledge": [
                {
                    "chunk_id": "知识切片ID",
                    "content": "知识内容",
                    "outdated_aspect": "过期方面",
                    "severity": "严重程度（高/中/低）",
                    "suggested_update": "建议更新内容",
                    "last_verified": "最后验证时间"
                }
            ],
            "freshness_score": "新鲜度评分(0-1)",
            "update_recommendations": "更新建议"
        }
        """

        # Build summarized text
        knowledge_text = "\n".join(
            [
                f"ID: {chunk.get('id', 'unknown')} | 更新时间: {chunk.get('last_updated', 'unknown')} | 内容: {chunk.get('content', '')}"
                for chunk in knowledge_base
            ]
        )

        prompt = f"""
        ### 指令 ###
        {instruction}

        ### 知识库内容 ###
        {knowledge_text}

        ### 当前时间 ###
        {datetime.now().strftime('%Y年%m月%d日')}

        ### 分析结果 ###
        """

        response = get_completion(prompt, self.model)
        response = preprocess_json_response(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ Outdated knowledge check failed.")
            return {
                "outdated_knowledge": [],
                "freshness_score": 0,
                "update_recommendations": "解析失败",
            }

    # --------------------------------------------------------
    # 📙 Step 3: Check for Conflicting Knowledge
    # --------------------------------------------------------
    def check_conflicting_knowledge(self, knowledge_base):
        """
        Detect conflicting or inconsistent statements across the knowledge base.
        """
        instruction = """
        你是一个知识一致性检查专家。请分析给定的知识库，找出可能存在冲突或矛盾的信息。

        检查标准：
        1. 同一主题的不同说法
        2. 价格信息的差异
        3. 时间信息的不一致
        4. 政策或规则冲突
        5. 流程或操作差异
        6. 联系方式矛盾

        请返回JSON格式：
        {
            "conflicting_knowledge": [
                {
                    "conflict_type": "冲突类型",
                    "chunk_ids": ["相关切片ID"],
                    "conflicting_content": ["冲突内容"],
                    "severity": "严重程度（高/中/低）",
                    "resolution_suggestion": "解决建议"
                }
            ],
            "consistency_score": "一致性评分(0-1)",
            "conflict_analysis": "冲突分析"
        }
        """

        knowledge_text = "\n".join(
            [
                f"ID: {chunk.get('id', 'unknown')} | 内容: {chunk.get('content', '')}"
                for chunk in knowledge_base
            ]
        )

        prompt = f"""
        ### 指令 ###
        {instruction}

        ### 知识库内容 ###
        {knowledge_text}

        ### 分析结果 ###
        """

        response = get_completion(prompt, self.model)
        response = preprocess_json_response(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ Conflicting knowledge check failed.")
            return {
                "conflicting_knowledge": [],
                "consistency_score": 0,
                "conflict_analysis": "解析失败",
            }

    # --------------------------------------------------------
    # 🧮 Step 4: Compute Overall Health Score
    # --------------------------------------------------------
    def calculate_overall_health_score(
        self, missing_result, outdated_result, conflicting_result
    ):
        """
        Compute weighted overall score combining coverage, freshness, and consistency.
        """
        coverage = float(missing_result.get("coverage_score", 0))
        freshness = float(outdated_result.get("freshness_score", 0))
        consistency = float(conflicting_result.get("consistency_score", 0))

        overall = 0.4 * coverage + 0.3 * freshness + 0.3 * consistency
        return round(overall, 3)

    # --------------------------------------------------------
    # 📊 Step 5: Generate Complete Health Report
    # --------------------------------------------------------
    def generate_health_report(self, knowledge_base, test_queries):
        """
        Generate a detailed health report summarizing all LLM-based checks.
        """
        print("\n=== 🧠 正在执行知识库健康检查 ===")

        print("1️⃣ 检查缺少的知识...")
        missing = self.check_missing_knowledge(knowledge_base, test_queries)

        print("2️⃣ 检查过期的知识...")
        outdated = self.check_outdated_knowledge(knowledge_base)

        print("3️⃣ 检查冲突的知识...")
        conflicting = self.check_conflicting_knowledge(knowledge_base)

        print("4️⃣ 计算整体健康评分...")
        score = self.calculate_overall_health_score(missing, outdated, conflicting)

        report = {
            "overall_health_score": score,
            "health_level": self.get_health_level(score),
            "missing_knowledge": missing,
            "outdated_knowledge": outdated,
            "conflicting_knowledge": conflicting,
            "recommendations": self.generate_recommendations(
                missing, outdated, conflicting
            ),
            "check_date": datetime.now().isoformat(),
        }

        return report

    # --------------------------------------------------------
    # 🧾 Helper: Determine Health Level
    # --------------------------------------------------------
    def get_health_level(self, score: float) -> str:
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "需要改进"

    # --------------------------------------------------------
    # 💡 Helper: Generate Improvement Recommendations
    # --------------------------------------------------------
    def generate_recommendations(self, missing, outdated, conflicting):
        recommendations = []
        if len(missing.get("missing_knowledge", [])) > 0:
            recommendations.append("补充缺少的知识点以提高覆盖率。")
        if len(outdated.get("outdated_knowledge", [])) > 0:
            recommendations.append("更新过期信息以确保时效性。")
        if len(conflicting.get("conflicting_knowledge", [])) > 0:
            recommendations.append("解决知识冲突以保持一致性。")
        if not recommendations:
            recommendations.append("知识库状态良好，无需立即调整。")
        return recommendations


# ============================================================
# 🚀 Demonstration
# ============================================================
def main():
    print("=== 知识库健康检查示例 ===")

    # -----------------------------
    # 示例知识库（包含冲突/过期信息）
    # -----------------------------
    knowledge_base = [
        {
            "id": "kb_001",
            "content": "上海迪士尼乐园于2016年开园。",
            "last_updated": "2024-01-15",
        },
        {
            "id": "kb_002",
            "content": "门票平日399元，周末499元。",
            "last_updated": "2023-12-01",
        },
        {
            "id": "kb_003",
            "content": "成人票平日350元，周末450元。",
            "last_updated": "2024-02-01",
        },
        {
            "id": "kb_004",
            "content": "乐园营业时间为8:00至20:00。",
            "last_updated": "2024-01-20",
        },
    ]

    # -----------------------------
    # 示例测试查询
    # -----------------------------
    test_queries = [
        {"query": "上海迪士尼在哪里", "expected_answer": "浦东新区"},
        {"query": "门票多少钱", "expected_answer": "价格信息"},
        {"query": "几点开门", "expected_answer": "8:00-20:00"},
        {"query": "停车费多少", "expected_answer": "停车信息"},
    ]

    # -----------------------------
    # 生成报告
    # -----------------------------
    checker = KnowledgeBaseHealthChecker()
    report = checker.generate_health_report(knowledge_base, test_queries)

    print("\n=== 🧾 知识库健康报告 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


# ============================================================
# 🏁 Entry Point
# ============================================================
if __name__ == "__main__":
    main()
