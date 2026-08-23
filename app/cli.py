"""本地多轮对话命令行。

对 Java 同学：这里不是 Servlet/Controller，而是控制台 REPL（类似用 Scanner
在 while(true) 里读 stdin）。本文件只负责 IO 与多轮上下文；意图识别、MySQL、
RAG、生成回复都在 LangGraph 工作流内部（见 app/workflow.py），对应 README 里的
understand → route → tools/clarify → respond。
"""

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 必须先于 agent / rag 相关 import 加载 .env
#
# 类比 Java：某个第三方 SDK 在 <clinit>（静态初始化）里读 System.getenv，
# 若此时还没把 .env 灌进环境变量，Spring @Value 来不及，值就被固化错了。
# huggingface_hub 在 import 时会固化 HF_HUB_OFFLINE，所以 load_dotenv() 必须
# 写在 `from app.agent import ...` 之前。
#
# 本文件自己再调一次 load_dotenv：直接 `python -m app.cli` 时不会走 main.py
# 里那一次；重复调用是幂等的（已存在的环境变量默认不覆盖）。
# ---------------------------------------------------------------------------
load_dotenv()

from typing import Any

from app.agent import create_financial_agent


# 退出口令集合，类比 Scanner 循环里判断 "quit"/"exit"
EXIT_COMMANDS = {"exit", "quit", "q", "退出", "结束"}


def _message_text(message: Any) -> str:
    """把 LangChain 消息的 content 收成给人看的纯文本。

    Java 类比：content 可能是 String，也可能是 List<Map>（多模态 content block）。
    这里做 duck typing：有 content 属性就取，否则把对象本身当 content。
    """
    # getattr(obj, "content", obj) ≈ 有 getContent() 就用，否则当 obj 本身就是内容
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content

    # 多模态块：只拼 type=="text" 的 text，忽略 image / tool 等块
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts)

    return str(content)


def _stream_reply(agent: Any, messages: list[Any], question: str) -> list[Any]:
    """一次用户提问：流式打印最终回复，并返回更新后的多轮 messages。

    类比 WebFlux Flux / SSE：一次 stream 订阅两个通道：
    - messages：每个 LLM token（带节点元数据）
    - values：每个图节点跑完后的完整 state 快照

    CLI 不实现业务路由；图内部才是 understand → tools/clarify → respond。
    """
    # end="" 不换行；flush=True 立刻刷出（≈ System.out.print + flush）
    print("\n财经客服：", end="", flush=True)
    updated_messages = messages
    streamed_any = False

    # [*messages, 本轮 user]：复制历史再 add，不原地 mutate
    # ≈ new ArrayList<>(sessionMessages); list.add(userMsg);
    # stream_mode 一次订阅 token 流 + 完整 state 流
    for mode, chunk in agent.stream(
        {"messages": [*messages, {"role": "user", "content": question}]},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            # chunk = (token, metadata)；只打印 respond 节点的字
            # 类比：过滤内部 Service 日志，只把 Controller 最终响应当输出
            # 避免把 understand（意图识别）/ tools（检索、查库）的内部输出刷到终端
            token, metadata = chunk
            if metadata.get("langgraph_node") != "respond":
                continue
            text = _message_text(token)
            if text:
                print(text, end="", flush=True)
                streamed_any = True
        elif mode == "values":
            # 完整 state 快照；用最后一次的 messages 覆盖本地 Session
            # clarify 节点不走 respond 流式，也靠这条拿到回复写入历史
            updated_messages = chunk.get("messages", updated_messages)

    # 流式通道没吐字时回退：读完整 body（≈ SSE 没帧时读 ResponseEntity）
    # 典型场景：clarify 模板回复，或模型一次性返回未走 token 流
    if not streamed_any and updated_messages:
        assistant = updated_messages[-1]
        content = (
            assistant.get("content", "")
            if isinstance(assistant, dict)
            else _message_text(assistant)
        )
        print(content, end="", flush=True)

    print()  # 本轮回复结束，换行
    # 调用方赋回 messages → 下一轮带着完整历史（含本轮 user + assistant）
    return updated_messages


def run_cli() -> None:
    """启动一个在当前进程内保留上下文的财经客服会话。

    生命周期：装配 Agent → 打印欢迎语 → Scanner 式循环读输入 →
    每轮交给图 stream → 用返回的 messages 更新内存 Session。
    """
    try:
        # 工厂方法：读 Settings、建 Embedding/FAISS、MySQL 仓储、workflow.compile()
        # 类比 @Configuration 里组装 ChatClient + VectorStore + JdbcTemplate，
        # 返回一个可调用的 StateMachine（CompiledStateGraph）
        agent = create_financial_agent()
    except (RuntimeError, ValueError) as exc:
        # 配置缺失（无 API Key / 非法参数）→ 启动失败，不进主循环
        # 类比 Spring Boot 启动失败，不进入业务 main
        print(f"启动失败：{exc}")
        return

    # 进程内对话历史（≈ 未外置的 HttpSession / List<ChatMessage>）
    # 刷新进程即丢；不是 Redis、不是 DB
    messages: list[Any] = []
    print("财经客服 Agent 已启动。输入“退出”结束会话。")
    # 真正的路由在 app/workflow.py；CLI 只负责 IO。流程概览见 README。
    print(
        "DeepSeek 负责意图识别；匹配资产查询则查 MySQL，"
        "否则检索 knowledge/ 并由 DeepSeek 基于证据回复。"
    )

    # 控制台 REPL，不是 Servlet 线程池；类似 while(scanner.hasNextLine())
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C / 管道结束 → 优雅退出（≈ 捕获 InterruptedException）
            print("\n会话已结束。")
            return

        if not question:
            continue  # 空行跳过，类比 Scanner 过滤空白
        if question.lower() in EXIT_COMMANDS:
            print("会话已结束。")
            return

        try:
            # 本轮 user + assistant 写回 Session，下一轮再带上
            messages = _stream_reply(agent, messages, question)
        except Exception as exc:  # 意图识别 / 回复生成 / 网络错误
            # 单轮失败不拆掉整个 REPL（连接还在）
            # 类比 Filter 里 catch 后返回 500 文案，会话继续
            print(f"\n处理失败：{exc}")
            continue
