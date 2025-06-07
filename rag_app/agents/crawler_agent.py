import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

BASE_URL = "https://www.lexisnexis.com/en-us"
OUTPUT_FILE = Path("rag_app/data/url_index.json")

visited = set()
results = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def crawl_page(page, url, base_domain):
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(1000)

        title = await page.title()
        logger.info(f"✅ {url} | Title: {title}")
        results.append({"url": url, "title": title})

        anchors = await page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
        for link in anchors:
            if link.startswith(BASE_URL):
                parsed_link = urlparse(link).scheme + "://" + urlparse(link).netloc + urlparse(link).path
                if parsed_link not in visited and parsed_link.startswith(base_domain):
                    visited.add(parsed_link)
                    await crawl_page(page, parsed_link, base_domain)

    except Exception as e:
        logger.warning(f"⚠️ Failed to crawl {url}: {e}")

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        visited.add(BASE_URL)
        await crawl_page(page, BASE_URL, BASE_URL)
        await browser.close()

        logger.info(f"💾 Saving {len(results)} pages to {OUTPUT_FILE}")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(run())
