
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

# --- Distance Metrics ---
def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    return -np.linalg.norm(vector_a - vector_b)  # Negative for sorting (higher = better)

def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    return -np.sum(np.abs(vector_a - vector_b))

def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    return np.dot(vector_a, vector_b)

def chebyshev_distance(vector_a: np.array, vector_b: np.array) -> float:
    return -np.max(np.abs(vector_a - vector_b))

DISTANCE_METRICS = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "dot_product": dot_product_similarity,
    "chebyshev": chebyshev_distance,
}

# --- VectorDatabase Class ---
class VectorDatabase:
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        distance_metric: str = "cosine"
    ):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.distance_metric_name = distance_metric

        if distance_metric not in DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        self.distance_function = DISTANCE_METRICS[distance_metric]

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def _matches_filter(self, item_metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        for key, value in filter_criteria.items():
            if key not in item_metadata:
                return False
            if isinstance(value, list):
                if item_metadata[key] not in value:
                    return False
            elif item_metadata[key] != value:
                return False
        return True

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        distance_measure = distance_measure or self.distance_function
        scores = []
        for key, vector in self.vectors.items():
            if metadata_filter and not self._matches_filter(self.metadata.get(key, {}), metadata_filter):
                continue
            score = distance_measure(query_vector, vector)
            scores.append((key, score, self.metadata.get(key, {})))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = None,
        return_as_text: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        distance_measure = distance_measure or self.distance_function
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, metadata_filter)
        if return_as_text:
            return [result[0] for result in results]
        return results

    def search_by_distance_metric(
        self,
        query_text: str,
        k: int,
        metric_name: str = "cosine",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {metric_name}. Available: {list(DISTANCE_METRICS.keys())}")
        distance_func = DISTANCE_METRICS[metric_name]
        return self.search_by_text(query_text, k, distance_func, metadata_filter=metadata_filter)

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, Any]]:
        return self.vectors.get(key, None), self.metadata.get(key, {})

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.metadata)

    def filter_by_metadata(self, filter_criteria: Dict[str, Any]) -> List[str]:
        return [key for key, meta in self.metadata.items() if self._matches_filter(meta, filter_criteria)]

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

    async def abuild_from_list_with_metadata(
        self, 
        texts_with_metadata: List[Tuple[str, Dict[str, Any]]]
    ) -> "VectorDatabase":
        texts = [text for text, _ in texts_with_metadata]
        embeddings = await self.embedding_model.async_get_embeddings(texts)
        for (text, metadata), embedding in zip(texts_with_metadata, embeddings):
            self.insert(text, np.array(embedding), metadata)
        return self

