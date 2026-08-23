"""Tests for the ACL Anthology RSS retriever."""

from datetime import datetime, timezone
from time import gmtime
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.retriever import get_retriever_cls
from zotero_arxiv_daily.retriever.acl_retriever import AclRetriever


FIXED_NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


@pytest.fixture()
def fixed_now(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.acl_retriever.datetime", _FixedDatetime)


def _configure(config, lookback_hours=24):
    with open_dict(config.source):
        config.source.acl = {"lookback_hours": lookback_hours}
    return AclRetriever(config)


def _entry(title, url, published):
    return {
        "title": title,
        "link": url,
        "description": "Ada Lovelace, Grace Hopper in Findings of ACL",
        "published_parsed": gmtime(published.timestamp()),
    }


def _patch_feed_and_pages(monkeypatch, entries):
    page = (
        '<script>paper_params={anthology_id:"2026.acl-1.1",'
        'title:"An ACL paper",authors:[{first:"Ada",last:"Lovelace"},'
        '{first:"Grace",last:"Hopper"}],abstract:"We study <b>sign language</b> \\"systems\\"."}</script>'
    )

    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.acl_retriever.feedparser.parse",
        lambda _: SimpleNamespace(entries=entries),
    )

    def _get(url, **kwargs):
        return SimpleNamespace(text=page, raise_for_status=lambda: None)

    monkeypatch.setattr("zotero_arxiv_daily.retriever.acl_retriever.requests.get", _get)


def test_acl_is_registered():
    assert get_retriever_cls("acl") is AclRetriever


def test_acl_retrieve_filters_by_date_and_reads_landing_page(config, monkeypatch, fixed_now):
    recent = _entry("Recent ACL", "https://aclanthology.org/2026.acl-1.1/", FIXED_NOW)
    old = _entry("Old ACL", "https://aclanthology.org/2026.acl-0.1/", FIXED_NOW.replace(day=20))
    _patch_feed_and_pages(monkeypatch, [recent, old])
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    papers = _configure(config).retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title == "Recent ACL"
    assert papers[0].authors == ["Ada Lovelace", "Grace Hopper"]
    assert papers[0].abstract == 'We study sign language "systems".'
    assert papers[0].url == "https://aclanthology.org/2026.acl-1.1"
    assert papers[0].pdf_url == "https://aclanthology.org/2026.acl-1.1.pdf"


def test_acl_falls_back_when_abstract_is_missing(config, monkeypatch, fixed_now):
    entry = _entry("Fallback ACL", "https://aclanthology.org/2026.acl-2.1/", FIXED_NOW)
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.acl_retriever.feedparser.parse",
        lambda _: SimpleNamespace(entries=[entry]),
    )

    def _get(url, **kwargs):
        return SimpleNamespace(
            text='paper_params={authors:[{first:"Ada",last:"Lovelace"}],abstract:""}',
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("zotero_arxiv_daily.retriever.acl_retriever.requests.get", _get)
    paper = _configure(config).retrieve_papers()[0]
    assert paper.abstract == "Fallback ACL"
    assert paper.authors == ["Ada Lovelace"]


def test_acl_debug_truncates(config, monkeypatch, fixed_now):
    entries = [
        _entry(f"ACL {i}", f"https://aclanthology.org/2026.acl-{i}.1/", FIXED_NOW)
        for i in range(15)
    ]
    _patch_feed_and_pages(monkeypatch, entries)
    config.executor.debug = True
    assert len(_configure(config)._retrieve_raw_papers()) == 10


def test_acl_historical_reads_bulk_bibtex(config, monkeypatch, fixed_now):
    with open_dict(config.source):
        config.source.acl = {
            "historical": True,
            "lookback_hours": 43800,
            "max_results": 100,
        }
    bib = b'''@inproceedings{2025.sign-1.1,
      author = {Ada Lovelace and Grace Hopper},
      title = {Sign Language Systems},
      abstract = {We study sign language systems.},
      year = {2025},
      url = {https://aclanthology.org/2025.sign-1.1/}
    }
    @inproceedings{2020.old-1.1,
      author = {Old Author},
      title = {An Old Paper},
      year = {2020}
    }'''

    import gzip

    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.acl_retriever.requests.get",
        lambda *args, **kwargs: SimpleNamespace(
            content=gzip.compress(bib),
            raise_for_status=lambda: None,
        ),
    )

    papers = AclRetriever(config)._retrieve_raw_papers()

    assert len(papers) == 1
    assert papers[0]["title"] == "Sign Language Systems"
    assert papers[0]["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert papers[0]["abstract"] == "We study sign language systems."
