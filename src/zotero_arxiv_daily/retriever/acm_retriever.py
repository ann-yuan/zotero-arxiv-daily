"""Retrieve recent ACM publications through Crossref metadata."""

from datetime import datetime, timedelta, timezone
import html
import re
from time import sleep
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("acm")
class AcmRetriever(BaseRetriever):
    """Retrieve recent ACM DOI records without scraping paywalled pages."""

    api_url = "https://api.crossref.org/works"
    request_headers = {
        "User-Agent": "zotero-arxiv-daily (https://github.com/TideDra/zotero-arxiv-daily)",
        "Accept": "application/json",
    }
    allowed_types = {"journal-article", "proceedings-article", "book-chapter", "posted-content"}

    def __init__(self, config):
        super().__init__(config)
        self.lookback_hours = int(self.retriever_config.get("lookback_hours", 24))
        self.max_results = int(self.retriever_config.get("max_results", 100))
        self.historical = bool(self.retriever_config.get("historical", False))
        self.metadata_only = bool(self.retriever_config.get("metadata_only", False))

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(10):
            try:
                response = requests.get(
                    self.api_url,
                    params=params,
                    headers=self.request_headers,
                    timeout=60,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if attempt == 9:
                    raise
                delay = 10 * (attempt + 1)
                logger.warning(f"Failed to retrieve ACM metadata: {exc}. Retry in {delay} seconds.")
                sleep(delay)
        return {}  # pragma: no cover

    @staticmethod
    def _parse_created(item: dict[str, Any]) -> datetime | None:
        created = item.get("created", {})
        date_time = created.get("date-time")
        if date_time:
            try:
                return datetime.fromisoformat(date_time.replace("Z", "+00:00"))
            except ValueError:
                pass
        date_parts = created.get("date-parts", [[]])[0]
        if not date_parts:
            return None
        values = list(date_parts) + [1, 1]
        return datetime(*values[:3], tzinfo=timezone.utc)

    @staticmethod
    def _clean_text(value: str | None) -> str:
        if not value:
            return ""
        value = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", html.unescape(value)).strip()

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        if self.historical:
            return self._retrieve_historical_raw_papers()

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.lookback_hours)
        params = {
            "filter": f"prefix:10.1145,from-created-date:{cutoff.date()},until-created-date:{now.date()}",
            "sort": "created",
            "order": "desc",
            "rows": self.max_results,
        }
        result = self._get_json(params)
        items = result.get("message", {}).get("items", [])
        raw_papers = []
        for item in items:
            created = self._parse_created(item)
            if created is None or created < cutoff or created > now:
                continue
            if item.get("type") not in self.allowed_types:
                continue
            raw_papers.append(item)

        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def _retrieve_historical_raw_papers(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.lookback_hours)
        rows = min(self.max_results, 1000)
        cursor = "*"
        raw_papers: list[dict[str, Any]] = []

        while len(raw_papers) < self.max_results:
            params = {
                "filter": f"prefix:10.1145,from-created-date:{cutoff.date()},until-created-date:{now.date()}",
                "sort": "created",
                "order": "desc",
                "rows": rows,
                "cursor": cursor,
            }
            result = self._get_json(params)
            message = result.get("message", {})
            items = message.get("items", [])
            if not items:
                break
            for item in items:
                created = self._parse_created(item)
                if created is None or created < cutoff or created > now:
                    continue
                if item.get("type") not in self.allowed_types:
                    continue
                raw_papers.append(item)
                if len(raw_papers) >= self.max_results:
                    break
            next_cursor = message.get("next-cursor")
            if not next_cursor or next_cursor == cursor or len(items) < rows:
                break
            cursor = next_cursor

        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        logger.info("Historical ACM metadata yielded {} records before scope filtering", len(raw_papers))
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        doi = raw_paper.get("DOI")
        titles = raw_paper.get("title") or []
        title = self._clean_text(titles[0] if titles else "")
        if not doi or not title:
            return None

        authors: list[str] = []
        affiliations: list[str] = []
        for author in raw_paper.get("author", []):
            name = author.get("name") or " ".join(
                part for part in (author.get("given"), author.get("family")) if part
            )
            if name:
                authors.append(self._clean_text(name))
            affiliations.extend(
                self._clean_text(affiliation.get("name"))
                for affiliation in author.get("affiliation", [])
                if affiliation.get("name")
            )

        abstract = self._clean_text(raw_paper.get("abstract")) or title
        url = (raw_paper.get("resource") or {}).get("primary", {}).get("URL")
        url = url or f"https://dl.acm.org/doi/{doi}"
        pdf_url = f"https://dl.acm.org/doi/pdf/{doi}"
        unique_affiliations = list(dict.fromkeys(affiliations))
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None,
            affiliations=unique_affiliations or None,
            published_at=self._parse_created(raw_paper),
        )
