"""Tests for the ACM Crossref retriever."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.retriever import get_retriever_cls
from zotero_arxiv_daily.retriever.acm_retriever import AcmRetriever


FIXED_NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


@pytest.fixture()
def fixed_now(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.acm_retriever.datetime", _FixedDatetime)


def _configure(config, lookback_hours=24, max_results=100):
    with open_dict(config.source):
        config.source.acm = {"lookback_hours": lookback_hours, "max_results": max_results}
    return AcmRetriever(config)


def _item(doi, created, item_type="proceedings-article"):
    return {
        "DOI": doi,
        "type": item_type,
        "created": {"date-time": created},
        "title": ["An ACM <i>paper</i>"],
        "author": [
            {
                "given": "Ada",
                "family": "Lovelace",
                "affiliation": [{"name": "Analytical Engine Lab"}],
            }
        ],
        "abstract": "<jats:p>We evaluate &amp; compare.</jats:p>",
        "resource": {"primary": {"URL": "https://dl.acm.org/doi/" + doi}},
    }


def _patch_crossref(monkeypatch, items):
    calls = []

    def _get(url, **kwargs):
        calls.append(kwargs.get("params", {}))
        return SimpleNamespace(
            json=lambda: {"message": {"items": items}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("zotero_arxiv_daily.retriever.acm_retriever.requests.get", _get)
    return calls


def test_acm_is_registered():
    assert get_retriever_cls("acm") is AcmRetriever


def test_acm_retrieve_filters_date_and_type(config, monkeypatch, fixed_now):
    recent = _item("10.1145/123", "2026-08-22T10:00:00Z")
    old = _item("10.1145/old", "2026-08-20T10:00:00Z")
    invalid = _item("10.1145/invalid", "2026-08-22T09:00:00Z", item_type="dataset")
    calls = _patch_crossref(monkeypatch, [recent, old, invalid])
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    papers = _configure(config).retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title == "An ACM paper"
    assert papers[0].authors == ["Ada Lovelace"]
    assert papers[0].abstract == "We evaluate & compare."
    assert papers[0].affiliations == ["Analytical Engine Lab"]
    assert papers[0].url == "https://dl.acm.org/doi/10.1145/123"
    assert papers[0].pdf_url == "https://dl.acm.org/doi/pdf/10.1145/123"
    assert calls[0]["filter"].startswith("prefix:10.1145,")


def test_acm_falls_back_to_doi_url_and_title_abstract(config):
    retriever = _configure(config)
    raw = _item("10.1145/fallback", "2026-08-22T10:00:00Z")
    raw.pop("resource")
    raw.pop("abstract")
    paper = retriever.convert_to_paper(raw)
    assert paper.url == "https://dl.acm.org/doi/10.1145/fallback"
    assert paper.abstract == "An ACM paper"


def test_acm_skips_missing_title(config):
    retriever = _configure(config)
    raw = _item("10.1145/no-title", "2026-08-22T10:00:00Z")
    raw["title"] = []
    assert retriever.convert_to_paper(raw) is None


def test_acm_debug_truncates(config, monkeypatch, fixed_now):
    items = [_item(f"10.1145/{i}", "2026-08-22T10:00:00Z") for i in range(15)]
    _patch_crossref(monkeypatch, items)
    config.executor.debug = True
    assert len(_configure(config)._retrieve_raw_papers()) == 10


def test_acm_historical_uses_cursor_pagination(config, monkeypatch, fixed_now):
    with open_dict(config.source):
        config.source.acm = {
            "historical": True,
            "lookback_hours": 43800,
            "max_results": 2,
        }
    items = [
        _item("10.1145/one", "2026-08-22T10:00:00Z"),
        _item("10.1145/two", "2026-08-22T09:00:00Z"),
    ]
    calls = []

    def _get(url, **kwargs):
        calls.append(kwargs["params"])
        return SimpleNamespace(
            json=lambda: {"message": {"items": items, "next-cursor": "next"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("zotero_arxiv_daily.retriever.acm_retriever.requests.get", _get)
    raw = AcmRetriever(config)._retrieve_raw_papers()

    assert len(raw) == 2
    assert calls[0]["cursor"] == "*"
