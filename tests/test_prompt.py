from app.prompts import FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT


def test_prompt_defines_identity_and_rag_boundary() -> None:
    prompt = FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT

    assert "财经客服" in prompt
    assert "search_project_knowledge" in prompt
    assert "没有联网" in prompt
    assert "不编造" in prompt
