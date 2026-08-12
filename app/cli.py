"""本地多轮对话命令行。"""

from typing import Any

from app.agent import create_financial_agent


EXIT_COMMANDS = {"exit", "quit", "q", "退出", "结束"}


def _message_text(message: Any) -> str:
    """兼容字符串和内容块两种 LangChain 消息格式。"""
    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts)

    return str(content)


def run_cli() -> None:
    """启动一个在当前进程内保留上下文的财经客服会话。"""
    try:
        agent = create_financial_agent()
    except (RuntimeError, ValueError) as exc:
        print(f"启动失败：{exc}")
        return

    messages: list[Any] = []
    print("财经客服 Agent 已启动。输入“退出”结束会话。")
    print("提醒：第一版没有联网、行情、账户或业务系统查询能力。")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话已结束。")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("会话已结束。")
            return

        try:
            result = agent.invoke(
                {"messages": [*messages, {"role": "user", "content": question}]}
            )
        except Exception as exc:  # API/网络错误需要在命令行友好呈现
            print(f"\n调用 DeepSeek 失败：{exc}")
            continue

        messages = result["messages"]
        print(f"\n财经客服：{_message_text(messages[-1])}")
