from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn

from search_agent.agent import root_agent

a2a_app = to_a2a(root_agent)


def main():
    # It is standard practice to run A2A remote agents on port 8001
    # locally to avoid conflicting with your primary client agents on 8000.
    uvicorn.run(a2a_app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
