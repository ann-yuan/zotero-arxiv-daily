"""Tests for sign-language scope matching and grouped email rendering."""

from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.scope import ScopeMatcher


def _config():
    return {
        "enabled": True,
        "drop_unmatched": True,
        "sign_language": {
            "anchors": ["sign language", "deaf"],
            "recognition": {
                "label": "识别",
                "strong": ["continuous sign language recognition"],
                "contextual": ["handshape recognition"],
            },
            "datasets": {
                "label": "语料库 / 数据集 / 标注",
                "strong": ["sign language dataset"],
                "contextual": ["gloss annotation"],
            },
            "generation": {
                "label": "生成",
                "strong": [],
                "contextual": ["motion generation"],
            },
        },
    }


def _paper(title, abstract):
    return Paper(
        source="test",
        title=title,
        authors=["Author"],
        abstract=abstract,
        url="https://example.com/paper",
    )


def test_scope_strong_keywords_match_without_contextual_keyword():
    matcher = ScopeMatcher(_config())
    paper = _paper("Continuous sign language recognition", "A benchmark study.")

    matcher.filter_papers([paper])

    assert paper.categories == ["recognition"]
    assert paper.matched_keywords == {
        "recognition": ["continuous sign language recognition"]
    }


def test_contextual_keywords_require_a_global_anchor():
    matcher = ScopeMatcher(_config())
    unrelated = _paper("Motion generation with diffusion", "We generate human motion.")
    related = _paper("Motion generation for deaf signers", "A sign language study.")

    assert matcher.filter_papers([unrelated]) == []
    assert matcher.filter_papers([related]) == [related]
    assert related.categories == ["generation"]
    assert related.matched_keywords == {"generation": ["motion generation"]}


def test_scope_keeps_multiple_matching_categories_and_keywords():
    matcher = ScopeMatcher(_config())
    paper = _paper(
        "A sign language dataset for continuous sign language recognition",
        "We provide gloss annotation for the benchmark.",
    )

    matcher.filter_papers([paper])

    assert paper.categories == ["recognition", "datasets"]
    assert paper.matched_keywords["recognition"] == ["continuous sign language recognition"]
    assert paper.matched_keywords["datasets"] == ["sign language dataset", "gloss annotation"]


def test_scope_can_keep_unmatched_when_configured():
    config = _config()
    config["drop_unmatched"] = False
    matcher = ScopeMatcher(config)
    paper = _paper("An unrelated paper", "No sign language content.")

    assert matcher.filter_papers([paper]) == [paper]
    assert paper.categories == []


def test_render_email_groups_papers_by_category_and_shows_matches():
    paper = _paper("A sign language dataset", "A corpus.")
    paper.score = 8.2
    paper.tldr = "A dataset paper."
    paper.categories = ["recognition", "datasets"]
    paper.matched_keywords = {
        "recognition": ["continuous sign language recognition"],
        "datasets": ["sign language dataset"],
    }

    html = render_email(
        [paper],
        category_labels={"recognition": "识别", "datasets": "语料库 / 数据集 / 标注"},
    )

    assert "识别" in html
    assert "语料库 / 数据集 / 标注" in html
    assert "Categories:" in html
    assert "Matched keywords:" in html
    assert html.count("A sign language dataset") == 2
