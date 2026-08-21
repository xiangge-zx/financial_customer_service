"""本地多轮对话命令行。"""

from dotenv import load_dotenv

# huggingface_hub 在 import 时固化 HF_HUB_OFFLINE；必须先于 agent/rag 导入加载 .env
load_dotenv()

from typing import Any

from app.agent import create_financial_agent


EXIT_COMMANDS = {"exit", "quit", "q", "退出", "结束"}


def _message_text(message: Any) -> str:
    """兼容字符串和内容块两种 LangChain 消息格式。"""
    content = getattr(message, "content", message)
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


def _stream_reply(agent: Any, messages: list[Any], question: str) -> list[Any]:
    """流式打印 respond 节点回复，并用最终 state 更新多轮上下文。"""
    print("\n财经客服：", end="", flush=True)
    updated_messages = messages
    streamed_any = False

    for mode, chunk in agent.stream(
        {"messages": [*messages, {"role": "user", "content": question}]},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            token, metadata = chunk
            if metadata.get("langgraph_node") != "respond":
                continue
            text = _message_text(token)
            if text:
                print(text, end="", flush=True)
                streamed_any = True
        elif mode == "values":
            updated_messages = chunk.get("messages", updated_messages)

    if not streamed_any and updated_messages:
        assistant = updated_messages[-1]
        content = (
            assistant.get("content", "")
            if isinstance(assistant, dict)
            else _message_text(assistant)
        )
        print(content, end="", flush=True)

    print()
    return updated_messages


def run_cli() -> None:
    """启动一个在当前进程内保留上下文的财经客服会话。"""
    try:
        agent = create_financial_agent()
    except (RuntimeError, ValueError) as exc:
        print(f"启动失败：{exc}")
        return

    messages: list[Any] = []
    print("财经客服 Agent 已启动。输入“退出”结束会话。")
    print(
        "DeepSeek 负责意图识别；匹配资产查询则查 MySQL，"
        "否则检索 knowledge/ 并由 DeepSeek 基于证据回复。"
    )

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
            messages = _stream_reply(agent, messages, question)
        except Exception as exc:  # 意图识别 / 回复生成 / 网络错误
            print(f"\n处理失败：{exc}")
            continue
