"""Keyword-based scope matching for the sign-language paper collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Mapping

from loguru import logger

from .protocol import Paper


@dataclass(frozen=True)
class ScopeCategory:
    name: str
    label: str
    strong: tuple[str, ...]
    contextual: tuple[str, ...]


def _normalise_text(value: str) -> str:
    """Normalise case and whitespace while preserving keyword punctuation."""
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _as_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values if str(value).strip())


class ScopeMatcher:
    """Classify papers using configured strong/contextual keywords.

    Strong keywords are sufficient by themselves. Contextual keywords require
    at least one global sign-language anchor, which prevents generic phrases
    such as ``motion generation`` from admitting unrelated papers.
    """

    def __init__(self, config: Mapping[str, Any] | None):
        self.enabled = bool(config.get("enabled", False)) if config else False
        self.drop_unmatched = bool(config.get("drop_unmatched", True)) if config else True
        self.anchors: tuple[str, ...] = ()
        self.categories: tuple[ScopeCategory, ...] = ()

        if not config:
            return

        sign_language = config.get("sign_language", {})
        self.anchors = _as_strings(sign_language.get("anchors", []))
        categories: list[ScopeCategory] = []
        for name, category_config in sign_language.items():
            if name == "anchors":
                continue
            categories.append(
                ScopeCategory(
                    name=str(name),
                    label=str(category_config.get("label", name)),
                    strong=_as_strings(category_config.get("strong", [])),
                    contextual=_as_strings(category_config.get("contextual", [])),
                )
            )
        self.categories = tuple(categories)

    @property
    def category_labels(self) -> dict[str, str]:
        return {category.name: category.label for category in self.categories}

    @staticmethod
    def _find_matches(text: str, keywords: tuple[str, ...]) -> list[str]:
        return [keyword for keyword in keywords if _normalise_text(keyword) in text]

    def classify_text(self, title: str, abstract: str = "") -> tuple[list[str], dict[str, list[str]]]:
        """Classify metadata text without requiring a fully converted Paper."""
        if not self.enabled:
            return [], {}

        text = _normalise_text(f"{title}\n{abstract or ''}")
        anchor_matches = self._find_matches(text, self.anchors)
        categories: list[str] = []
        matched_keywords: dict[str, list[str]] = {}

        for category in self.categories:
            strong_matches = self._find_matches(text, category.strong)
            contextual_matches = (
                self._find_matches(text, category.contextual) if anchor_matches else []
            )
            matches = list(dict.fromkeys(strong_matches + contextual_matches))
            if matches:
                categories.append(category.name)
                matched_keywords[category.name] = matches

        return categories, matched_keywords

    def classify(self, paper: Paper) -> tuple[list[str], dict[str, list[str]]]:
        """Assign all matching categories and record the matched keywords."""
        categories, matched_keywords = self.classify_text(paper.title, paper.abstract)
        paper.categories = categories
        paper.matched_keywords = matched_keywords
        return categories, matched_keywords

    def filter_papers(self, papers: list[Paper]) -> list[Paper]:
        """Classify and optionally drop papers outside the configured scope."""
        if not self.enabled:
            return papers

        kept: list[Paper] = []
        dropped = 0
        for paper in papers:
            categories, _ = self.classify(paper)
            if categories or not self.drop_unmatched:
                kept.append(paper)
            else:
                dropped += 1

        logger.info(
            "Scope filter kept {} papers and dropped {} papers outside the configured sign-language scope",
            len(kept),
            dropped,
        )
        for category in self.categories:
            count = sum(category.name in paper.categories for paper in kept)
            logger.info("Scope category {} ({}) matched {} papers", category.name, category.label, count)
        return kept

    def rank_standalone(self, papers: list[Paper]) -> list[Paper]:
        """Rank scoped papers without a Zotero corpus.

        Strong keyword matches receive twice the weight of contextual matches;
        papers matching more categories receive a small diversity bonus, and
        newer records receive a recency bonus. Scores are normalised to the
        same 0-10 range used by the embedding reranker.
        """
        if not papers:
            return []

        strong_keywords = {
            category.name: {_normalise_text(keyword) for keyword in category.strong}
            for category in self.categories
        }
        raw_scores: list[float] = []
        now = datetime.now(timezone.utc)
        for paper in papers:
            keyword_score = 0.0
            for category, keywords in paper.matched_keywords.items():
                strong = strong_keywords.get(category, set())
                keyword_score += sum(
                    2.0 if _normalise_text(keyword) in strong else 1.0
                    for keyword in keywords
                )

            category_bonus = min(len(paper.categories), len(self.categories)) * 0.5
            recency_bonus = 0.0
            if paper.published_at is not None:
                published_at = paper.published_at
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                age_days = max(
                    0.0,
                    (now - published_at.astimezone(timezone.utc)).total_seconds() / 86400,
                )
                recency_bonus = 1.0 / (1.0 + age_days)
            raw_scores.append(keyword_score + category_bonus + recency_bonus)

        maximum = max(raw_scores, default=0.0)
        for paper, raw_score in zip(papers, raw_scores):
            paper.score = 10.0 * raw_score / maximum if maximum else 0.0
        ranked = sorted(
            papers,
            key=lambda paper: (
                paper.score or 0.0,
                paper.published_at.timestamp() if paper.published_at is not None else float("-inf"),
            ),
            reverse=True,
        )
        logger.info("Standalone scope ranking scored {} papers without Zotero", len(ranked))
        return ranked
