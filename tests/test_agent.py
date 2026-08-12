from app.agent import create_financial_agent
from app.config import Settings


def test_first_version_agent_has_no_tools_node() -> None:
    agent = create_financial_agent(Settings(api_key="offline-test-key"))

    node_names = set(agent.get_graph().nodes)

    assert "model" in node_names
    assert "tools" not in node_names
