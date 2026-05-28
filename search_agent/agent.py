import json
import textwrap
import trafilatura

from ddgs import DDGS
from loguru import logger
from pydantic import BaseModel, Field

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

DEFAULT_MODEL = "ollama_chat/gemma4:e4b"
llm_client = LiteLlm(model=DEFAULT_MODEL)


class ResearchReport(BaseModel):
    """Defines the strict output schema for the LLM."""
    topic: str = Field(description="The exact topic researched")
    executive_summary: str = Field(description="A dense, high-level summary of the findings")
    key_data_points: list[str] = Field(description="List of key facts discovered")
    sources_used: list[str] = Field(description="List of URLs used as sources")


def search_duckduckgo(query: str, time_limit: str = "w") -> str:
    """
    Searches the open web for a given query to find relevant URLs.
    
    Args:
        query: The specific topic or question to research.
        time_limit: Limits the freshness of results. Acceptable values are 'd' (past day), 
                    'w' (past week), 'm' (past month), or 'y' (past year).
        
    Returns:
        A JSON string containing the title, URL, and a brief snippet of the top results.
    """
    try:
        # Fetch top 3 results and apply the freshness constraint
        results = DDGS().text(query, max_results=3, timelimit=time_limit)
        return json.dumps(list(results))
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return f"Search failed: {str(e)}"


def scrape_website(url: str) -> str:
    """
    Scrapes a webpage and intelligently extracts the core textual content using trafilatura, 
    ignoring navigation menus, footers, and scripts.
    
    Args:
        url: The exact URL to scrape.
        
    Returns:
        The cleaned, extracted raw text from the webpage.
    """
    try:
        # Trafilatura handles the request and headers internally
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded is None:
            logger.warning(f"Trafilatura failed to fetch URL: {url}")
            return f"Scraping failed: Could not fetch {url}"

        # Extract the core text payload
        extracted_text = trafilatura.extract(downloaded)
        
        if not extracted_text:
            logger.warning(f"No text extracted from URL: {url}")
            return f"Scraping failed: No main text content found at {url}"

        # Smarter truncation preserving word boundaries to prevent context overflow
        shortened_text = textwrap.shorten(
            extracted_text, 
            width=5000, 
            placeholder="... [Content Truncated]"
        )
        return shortened_text
        
    except Exception as e:
        logger.error(f"Scraping exception for {url}: {e}")
        return f"Scraping failed for {url}: {str(e)}"


# Initialize the ADK Agent
root_agent = Agent(
    name="search_agent",
    model=llm_client,
    instruction=f"""You are an autonomous research compiler working in a machine-to-machine pipeline. 
    Your workflow:
    1. Receive a research query.
    2. Use the `search_duckduckgo` tool to find the most relevant sources. Evaluate the snippets returned to select the most authoritative and relevant URL.
    3. Use the `scrape_website` tool to extract the raw text from the most promising URL. Do not attempt to scrape PDF links.
    4. Synthesize the findings into a strict JSON payload.
    
    You must output ONLY valid JSON. Do not include markdown code blocks or conversational text.
    Your JSON must follow this exact JSON Schema:
    {json.dumps(ResearchReport.model_json_schema(), indent=2)}
    """,
    tools=[search_duckduckgo, scrape_website]
)
