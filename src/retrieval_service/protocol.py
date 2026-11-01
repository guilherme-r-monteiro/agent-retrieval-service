from typing import Protocol

from retrieval_service.corpus import Document
from retrieval_service.hybrid import Mode


class Searcher(Protocol):
    def search(self, query: str, k: int, mode: Mode) -> list[Document]: ...
