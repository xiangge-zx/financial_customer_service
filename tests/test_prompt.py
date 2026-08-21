from app.prompts import (
    FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    WORKFLOW_DESCRIPTIONS,
)


def test_prompt_defines_identity_and_rag_boundary() -> None:
    prompt = FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT

    assert "财经客服" in prompt
    assert "知识库检索证据" in prompt
    assert "不编造" in prompt
    assert "资产查询结构化结果" in prompt


def test_intent_prompt_includes_workflow_descriptions() -> None:
    assert "asset_query" in WORKFLOW_DESCRIPTIONS
    assert "asset_dossier" in WORKFLOW_DESCRIPTIONS
    assert "faq_rag" in WORKFLOW_DESCRIPTIONS
    assert "asset_query" in INTENT_CLASSIFICATION_SYSTEM_PROMPT
    assert "结构化分类结果" in INTENT_CLASSIFICATION_SYSTEM_PROMPT
