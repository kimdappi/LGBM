from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import ToolCard


@dataclass
class ToolRAGIndex:
    cards: List[ToolCard]
    vectorizer: TfidfVectorizer
    matrix: object  # scipy sparse


def build_toolrag_index(cards: List[ToolCard]) -> ToolRAGIndex:
    texts = [c.to_text() for c in cards]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts) if texts else None
    return ToolRAGIndex(cards=cards, vectorizer=vectorizer, matrix=matrix)


def retrieve_tool_cards(index: ToolRAGIndex, query: str, top_n: int = 5) -> List[Tuple[ToolCard, float]]:
    if not index.cards or index.matrix is None:
        return []
    q = index.vectorizer.transform([query])
    sims = cosine_similarity(q, index.matrix)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
    out: List[Tuple[ToolCard, float]] = []
    for idx, score in ranked:
        out.append((index.cards[idx], float(score)))
    return out

