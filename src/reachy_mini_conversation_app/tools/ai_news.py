"""AI News tool - fetches trending AI papers from HuggingFace."""

import json
import asyncio
import logging
import urllib.error
import urllib.request
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

HUGGINGFACE_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def _fetch_papers_sync(limit: int = 5) -> list[dict[str, Any]]:
    """Fetch papers synchronously (to be run in executor)."""
    try:
        req = urllib.request.Request(
            HUGGINGFACE_DAILY_PAPERS_URL,
            headers={"User-Agent": "ReachyMini/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        papers = []
        for item in data[:limit]:
            paper = item.get("paper", {})
            papers.append({
                "title": paper.get("title", "Unknown"),
                "summary": paper.get("summary", "")[:300] + "..." if paper.get("summary") else "",
                "authors": [a.get("name", "") for a in paper.get("authors", [])[:3]],
                "upvotes": item.get("paper", {}).get("upvotes", 0),
            })
        return papers

    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch AI news: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI news response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching AI news: {e}")
        return []


class AiNews(Tool):
    """Fetch the latest trending AI research papers from HuggingFace."""

    name = "ai_news"
    description = (
        "Fetch the latest trending AI research papers and news from HuggingFace's daily papers. "
        "Use this when the user asks about AI news, recent AI research, or what's happening in AI. "
        "Returns paper titles, brief summaries, and authors."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of papers to fetch (1-10, default 3).",
            },
            "topic": {
                "type": "string",
                "description": "Optional topic to mention when presenting news (e.g., 'robotics', 'language models'). "
                "Note: filtering is not supported, but you can use this to contextualize your response.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch AI news from HuggingFace daily papers."""
        count = min(max(int(kwargs.get("count", 3)), 1), 10)
        topic = kwargs.get("topic", "")

        logger.info(f"Tool call: ai_news count={count} topic={topic}")

        # Run synchronous fetch in executor to avoid blocking
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(None, _fetch_papers_sync, count)

        if not papers:
            return {
                "status": "error",
                "message": "Could not fetch AI news. The service might be temporarily unavailable.",
            }

        return {
            "status": "success",
            "count": len(papers),
            "papers": papers,
            "source": "HuggingFace Daily Papers",
            "hint": "Share these papers enthusiastically! Consider using a dance or emotion to express excitement about interesting research.",
        }
