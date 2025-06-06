import requests
import google.generativeai as genai
import os
from typing import Dict

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Cache for previously scraped URLs
url_cache: Dict[str, str] = {}

def summarize_html_with_ai(url: str) -> str:
    """
    Summarizes a web page using Gemini by extracting key content from HTML.
    """
    if url in url_cache:
        return url_cache[url]

    try:
        response = requests.get(url, timeout=10)
        html = response.text[:15000]  # Limit to avoid overly long prompts

        prompt = f"""
You are a helpful assistant reading a web page's HTML. Extract the key ideas, focusing on the business context, products, AI use, and strategies. Ignore all visual or structural markup.

HTML:
{html}

Summary:
"""
        result = gemini_model.generate_content(prompt)
        summary = result.text.strip()
        url_cache[url] = summary
        return summary

    except Exception as e:
        return f"Error scraping {url}: {e}"

def batch_scrape_and_summarize(urls: list) -> Dict[str, str]:
    """
    Process a batch of URLs and return their AI-generated summaries.
    """
    from concurrent.futures import ThreadPoolExecutor

    def safe_scrape(url):
        return url, summarize_html_with_ai(url)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(safe_scrape, urls))

    return {url: summary for url, summary in results}
