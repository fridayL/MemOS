# memos/reranker/strategies/single_turn.py
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .dialogue_common import DialogueRankingTracker
from .single_turn import SingleTurnStrategy


class SingleTurnOutFilterStrategy(SingleTurnStrategy):
    """
    Single turn dialogue strategy.

    This strategy processes dialogue pairs by concatenating user and assistant
    messages into single strings for ranking. Each dialogue pair becomes a
    separate document for ranking.
    example:
        >>> documents = ["chat_time: 2025-01-01 12:00:00\nuser: hello\nassistant: hi there"]
        >>> output memory item: ["Memory:xxx \n\n chat_time: 2025-01-01 12:00:00\nuser: hello\nassistant: hi there"]
    """

    def prepare_documents(
        self,
        query: str,
        graph_results: list,
        top_k: int,
        **kwargs,
    ) -> tuple[DialogueRankingTracker, dict[str, Any], list[str]]:
        """
        Prepare documents based on single turn concatenation strategy.

        Args:
            query: The search query
            graph_results: List of graph results
            top_k: Maximum number of items to return

        Returns:
            tuple[DialogueRankingTracker, dict[str, Any], list[str]]:
            - Tracker: DialogueRankingTracker instance
            - original_items: Dict mapping memory_id to original TextualMemoryItem
            - documents: List of text documents ready for ranking
        """
        return super().prepare_documents(query, graph_results, top_k, **kwargs)

    def reconstruct_items(
        self,
        ranked_indices: list[int],
        scores: list[float],
        tracker: DialogueRankingTracker,
        original_items: dict[str, Any],
        top_k: int,
        **kwargs,
    ) -> list[tuple[Any, float]]:
        """
        Reconstruct TextualMemoryItem objects from ranked dialogue pairs.

        Args:
            ranked_indices: List of dialogue pair indices sorted by relevance
            scores: Corresponding relevance scores
            tracker: DialogueRankingTracker instance
            original_items: Dict mapping memory_id to original TextualMemoryItem
            top_k: Maximum number of items to return

        Returns:
            List of (reconstructed_memory_item, aggregated_score) tuples
        """
        filter_sources = {}
        reconstructed_items = []
        for idx, score in zip(ranked_indices, scores, strict=False):
            dialogue_pair = tracker.get_dialogue_pair_by_index(idx)
            if dialogue_pair and (dialogue_pair.memory_id in original_items):
                original_item = original_items[dialogue_pair.memory_id]
                reconstructed_item = deepcopy(original_item)
                if dialogue_pair.combined_text in filter_sources:
                    reconstructed_items.append((reconstructed_item, score))
                else:
                    filter_sources[dialogue_pair.combined_text] = True
                    reconstructed_item.memory = (
                        dialogue_pair.memory
                        + "\n\nsources-dialogue-pairs"
                        + dialogue_pair.combined_text
                    )
                    reconstructed_items.append((reconstructed_item, score))

        # Sort by aggregated score and return top_k
        reconstructed_items.sort(key=lambda x: x[1], reverse=True)
        return reconstructed_items[:top_k]