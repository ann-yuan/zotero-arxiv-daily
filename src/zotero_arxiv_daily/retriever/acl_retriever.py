"""Retrieve recently ingested papers from the ACL Anthology RSS feed."""

from datetime import datetime, timedelta, timezone
import gzip
import html
import json
import re
from time import struct_time
from typing import Any

import bibtexparser
import feedparser
import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("acl")
class AclRetriever(BaseRetriever):
    """Retrieve papers recently added to the ACL Anthology.

    The Anthology publishes an official RSS feed for newly added papers. The
    feed contains titles and links; the public landing page contains a
    machine-readable ``paper_params`` object with the abstract and authors.
    """

    feed_url = "https://aclanthology.org/papers/index.xml"
    bulk_bib_url = "https://aclanthology.org/anthology+abstracts.bib.gz"
    request_headers = {
        "User-Agent": "zotero-arxiv-daily (https://github.com/TideDra/zotero-arxiv-daily)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    }

    def __init__(self, config):
        super().__init__(config)
        self.lookback_hours = int(self.retriever_config.get("lookback_hours", 24))
        self.historical = bool(self.retriever_config.get("historical", False))
        self.max_results = int(self.retriever_config.get("max_results", 0))
        self.metadata_only = bool(self.retriever_config.get("metadata_only", False))

    @staticmethod
    def _entry_datetime(entry: Any) -> datetime | None:
        parsed: struct_time | None = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed is None:
            return None
        return datetime(*parsed[:6], tzinfo=timezone.utc)

    @staticmethod
    def _clean_text(value: str | None) -> str:
        if not value:
            return ""
        value = re.sub(r"<[^>]+>", " ", value)
        return re.sub(r"\s+", " ", html.unescape(value)).strip()

    @staticmethod
    def _decode_js_string(value: str) -> str:
        try:
            return json.loads(f'"{value}"')
        except (json.JSONDecodeError, TypeError):
            return value

    def _fetch_page_metadata(self, url: str) -> tuple[list[str], str]:
        response = requests.get(url, headers=self.request_headers, timeout=30)
        response.raise_for_status()
        page = response.text

        authors: list[str] = []
        authors_match = re.search(r"authors:\[(.*?)\],abstract:", page, flags=re.DOTALL)
        if authors_match:
            for first, last in re.findall(
                r'first:"((?:\\.|[^"\\])*)",last:"((?:\\.|[^"\\])*)"',
                authors_match.group(1),
            ):
                name = f"{self._decode_js_string(first)} {self._decode_js_string(last)}".strip()
                if name:
                    authors.append(name)

        abstract = ""
        abstract_match = re.search(
            r'paper_params=.*?abstract:"((?:\\.|[^"\\])*)"', page, flags=re.DOTALL
        )
        if abstract_match:
            abstract = self._clean_text(self._decode_js_string(abstract_match.group(1)))
        return authors, abstract

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        if self.historical:
            return self._retrieve_historical_raw_papers()

        feed = feedparser.parse(self.feed_url)
        entries = getattr(feed, "entries", [])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        raw_papers: list[dict[str, Any]] = []

        for entry in entries:
            published = self._entry_datetime(entry)
            if published is None or published < cutoff:
                continue

            title = self._clean_text(entry.get("title"))
            url = entry.get("link") or ""
            if not title or not url:
                continue

            description = self._clean_text(entry.get("description"))
            authors = []
            if " in " in description:
                authors = [a.strip() for a in description.split(" in ", 1)[0].split(",") if a.strip()]

            try:
                page_authors, abstract = self._fetch_page_metadata(url)
                if page_authors:
                    authors = page_authors
            except requests.RequestException as exc:
                logger.warning(f"Failed to retrieve ACL metadata for {url}: {exc}")
                abstract = ""

            # A few Anthology records do not yet expose an abstract. Keeping
            # the title gives the reranker/LLM a useful minimal representation.
            abstract = abstract or title
            raw_papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": url.rstrip("/"),
                    "pdf_url": f"{url.rstrip('/')}.pdf",
                    "published": published,
                }
            )

        raw_papers.sort(key=lambda paper: paper["published"], reverse=True)
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def _retrieve_historical_raw_papers(self) -> list[dict[str, Any]]:
        """Read ACL's official bulk BibTeX export for a multi-year run."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        response = requests.get(self.bulk_bib_url, headers=self.request_headers, timeout=180)
        response.raise_for_status()
        database = bibtexparser.loads(gzip.decompress(response.content).decode("utf-8"))
        raw_papers: list[dict[str, Any]] = []

        for entry in database.entries:
            try:
                year = int(str(entry.get("year", "0"))[:4])
            except ValueError:
                continue
            published = datetime(year, 1, 1, tzinfo=timezone.utc)
            if published < cutoff:
                continue

            title = self._clean_text(entry.get("title"))
            if not title:
                continue
            paper_id = entry.get("ID") or entry.get("id")
            url = entry.get("url") or (
                f"https://aclanthology.org/{paper_id}/" if paper_id else ""
            )
            if not url:
                continue
            author_text = str(entry.get("author", ""))
            authors = [author.strip() for author in re.split(r"\s+and\s+", author_text) if author.strip()]
            abstract = self._clean_text(entry.get("abstract")) or title
            raw_papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": url.rstrip("/"),
                    "pdf_url": f"{url.rstrip('/')}.pdf",
                    "published": published,
                }
            )

        raw_papers.sort(key=lambda paper: paper["published"], reverse=True)
        if self.max_results:
            raw_papers = raw_papers[:self.max_results]
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        logger.info("Historical ACL metadata yielded {} records before scope filtering", len(raw_papers))
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper:
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=raw_paper["authors"],
            abstract=raw_paper["abstract"],
            url=raw_paper["url"],
            pdf_url=raw_paper["pdf_url"],
            full_text=None,
            published_at=raw_paper.get("published"),
        )
