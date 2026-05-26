import os
import json
import logging

# 1. Enable LiteLLM debug logging BEFORE imports so it uses standard Python logging
os.environ["LITELLM_LOG"] = "DEBUG"

import uvicorn
from loguru import logger
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from search_agent.agent import root_agent


class CleanInterceptor(logging.Handler):
    """Intercepts logs, silences all spam, and extracts clean LLM thoughts."""
    def emit(self, record):
        msg = record.getMessage()

        # 2. Hunt for the LiteLLM RAW RESPONSE block
        if "RAW RESPONSE:" in msg:
            try:
                json_str = msg.split("RAW RESPONSE:")[1].strip()
                data = json.loads(json_str)
                
                message = data.get("message", {})
                thought = message.get("thinking") or message.get("reasoning_content")
                
                if thought:
                    # raw=True completely removes the timestamps and log levels!
                    logger.opt(raw=True, colors=True).info(
                        f"<blue>[Search_Agent | thought]</blue>\n<magenta>{thought}</magenta>\n"
                    )
            except Exception:
                pass
            return  # Drop the raw JSON block

        # 3. Drop ALL debug/info spam from background libraries (including LiteLLM)
        if record.name.startswith(("httpx", "httpcore", "urllib3", "asyncio", "a2a", "LiteLLM", "litellm")):
            if record.levelno < logging.WARNING:
                return

        # 4. Keep Uvicorn startup logs normal
        if record.levelno >= logging.INFO:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            logger.opt(exception=record.exc_info).log(level, msg)


a2a_app = to_a2a(root_agent)

def main():
    # Force all python logging through our clean interceptor
    logging.basicConfig(handlers=[CleanInterceptor()], level=0, force=True)

    uvicorn.run(
        a2a_app, 
        host="0.0.0.0", 
        port=8001,
        log_config=None
    )


if __name__ == "__main__":
    main()
