from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def body(self) -> str:
        return f"{self.title}. {self.text}"


def load_documents(path: Path) -> list[Document]:
    frame = pd.read_parquet(path)
    return [Document(str(row["_id"]), row["title"], row["text"]) for _, row in frame.iterrows()]


def load_queries(path: Path, qrels_path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    queries = pd.read_parquet(path)
    qrels = pd.read_csv(qrels_path, sep="\t", dtype=str)
    relevant: dict[str, set[str]] = {}
    for row in qrels.itertuples():
        if int(row.score) > 0:
            relevant.setdefault(row._1, set()).add(row._2)
    texts = {str(r["_id"]): r["text"] for _, r in queries.iterrows() if str(r["_id"]) in relevant}
    return texts, relevant
