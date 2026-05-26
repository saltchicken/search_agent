import json

from bs4 import BeautifulSoup
from ddgs import DDGS
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
import requests

DEFAULT_MODEL = "ollama_chat/gemma4:e4b"
llm_client = LiteLlm(model=DEFAULT_MODEL)


def search_duckduckgo(query: str) -> str:
    """
    Searches the open web for a given query to find relevant URLs.
    
    Args:
        query: The specific topic or question to research.
        
    Returns:
        A JSON string containing the title, URL, and a brief snippet of the top results.
    """
    try:
        # Fetch top 3 results to keep context windows manageable
        results = DDGS().text(query, max_results=3)
        return json.dumps(list(results))
    except Exception as e:
        return f"Search failed: {str(e)}"


def scrape_website(url: str) -> str:
    """
    Scrapes a webpage and intelligently extracts the core textual content, 
    ignoring navigation menus, footers, and scripts.
    
    Args:
        url: The exact URL to scrape.
        
    Returns:
        The cleaned, extracted raw text from the webpage.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Strip out noisy HTML elements
        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Target content-heavy tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        extracted_text = " ".join(
            [elem.get_text(strip=True) for elem in text_elements])

        # Truncate to the first 5000 characters to prevent LLM context overflow
        return extracted_text[:5000]
    except Exception as e:
        return f"Scraping failed for {url}: {str(e)}"


# Initialize the ADK Agent
root_agent = Agent(
    name="search_agent",
    model=llm_client,
    instruction=
    """You are an autonomous research compiler working in a machine-to-machine pipeline. 
    Your workflow:
    1. Receive a research query.
    2. Use the `search_duckduckgo` tool to find the most relevant sources.
    3. Use the `scrape_website` tool to extract the raw text from the most promising URLs.
    4. Synthesize the findings into a strict JSON payload.
    
    You must output ONLY valid JSON. Do not include markdown code blocks or conversational text.
    Your JSON must follow this exact schema:
    {
      "topic": "The exact topic researched",
      "executive_summary": "A dense, high-level summary of the findings",
      "key_data_points": ["fact 1", "fact 2", "fact 3"],
      "sources_used": ["url1", "url2"]
    }""",
    tools=[search_duckduckgo, scrape_website])
