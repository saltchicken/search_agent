import json
import logging

import uvicorn
from loguru import logger
import litellm  # <--- Import litellm directly
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from search_agent.agent import root_agent


class SmartInterceptor(logging.Handler):
    """Intercepts all logs, silences network spam, and extracts LLM thoughts."""
    def emit(self, record):
        # 1. Silently drop all DEBUG spam from these noisy networking libraries
        if record.name.startswith(("httpx", "httpcore", "urllib3", "asyncio", "a2a")):
            if record.levelno < logging.INFO:
                return

        msg = record.getMessage()

        # 2. Hunt for the LiteLLM RAW RESPONSE block
        if "RAW RESPONSE:" in msg:
            try:
                # Strip handles the tricky newline characters
                json_str = msg.split("RAW RESPONSE:")[1].strip()
                data = json.loads(json_str)
                
                # Extract the thinking/reasoning content
                message = data.get("message", {})
                thought = message.get("thinking") or message.get("reasoning_content")
                
                if thought:
                    logger.opt(colors=True).info(f"\n<magenta>{thought}</magenta>\n")
            except Exception:
                pass
            return  # We extracted the thought, so don't print the ugly raw JSON block

        # 3. For everything else (like Uvicorn startup logs or our tool logs), print normally
        if record.levelno >= logging.INFO:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            logger.opt(exception=record.exc_info).log(level, msg)


a2a_app = to_a2a(root_agent)

def main():
    # Force LiteLLM to spit out its internal debugging (which contains the thoughts)
    litellm.set_verbose = True

    # Force absolutely EVERYTHING through our smart interceptor
    logging.basicConfig(handlers=[SmartInterceptor()], level=0, force=True)

    # Run Uvicorn without its default logging config
    uvicorn.run(
        a2a_app, 
        host="0.0.0.0", 
        port=8001,
        log_config=None
    )


if __name__ == "__main__":
    main()
