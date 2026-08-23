from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, RawPaperItem
from tqdm import tqdm
from typing import Type
from time import sleep
from loguru import logger
from ..scope import ScopeMatcher


class BaseRetriever(ABC):
    name: str
    def __init__(self, config:DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source,self.name)

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[RawPaperItem]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper:RawPaperItem) -> Paper | None:
        pass

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        scope_matcher: ScopeMatcher | None = getattr(self, "scope_matcher", None)
        if scope_matcher is not None and scope_matcher.enabled:
            scoped_raw_papers = []
            dropped = 0
            for raw_paper in raw_papers:
                title, abstract = self._raw_metadata(raw_paper)
                categories, _ = scope_matcher.classify_text(title, abstract)
                if categories or not scope_matcher.drop_unmatched:
                    scoped_raw_papers.append(raw_paper)
                else:
                    dropped += 1
            logger.info(
                "Scope prefilter kept {} raw papers and dropped {} before conversion",
                len(scoped_raw_papers),
                dropped,
            )
            raw_papers = scoped_raw_papers
        logger.info("Processing papers...")
        papers = []
        for raw_paper in tqdm(raw_papers, total=len(raw_papers), desc="Converting papers"):
            try:
                paper = self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"Skipping paper {getattr(raw_paper, 'title', raw_paper)}: {exc}")
                continue
            if paper is not None:
                papers.append(paper)
            # Historical metadata-only runs do not need the daily source-rate
            # pacing or full-text downloads; keep the daily behavior unchanged.
            if not getattr(self, "metadata_only", False):
                sleep(1)
        return papers

    @staticmethod
    def _raw_metadata(raw_paper: RawPaperItem) -> tuple[str, str]:
        """Extract title/abstract metadata from common retriever raw formats."""
        if isinstance(raw_paper, dict):
            title = raw_paper.get("title", "")
            abstract = raw_paper.get("abstract", "")
            if isinstance(title, list):
                title = title[0] if title else ""
            return str(title or ""), str(abstract or "")

        title = getattr(raw_paper, "title", "")
        abstract = getattr(raw_paper, "summary", "")
        return str(title or ""), str(abstract or "")

registered_retrievers = {}

def register_retriever(name:str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls
    return decorator

def get_retriever_cls(name:str) -> Type[BaseRetriever]:
    if name not in registered_retrievers:
        raise ValueError(f"Retriever {name} not found")
    return registered_retrievers[name]
