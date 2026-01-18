"""Evaluation metrics for movie recommendation systems."""

import pandas as pd
import numpy as np
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_sentence_transformer = None

def _get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_transformer


def genre_precision_at_k(
    recommendations: pd.DataFrame,
    query_genres: List[str],
    k: int
) -> float:
    """Calculate precision@K based on genre overlap.

    A recommendation is relevant if it shares at least one genre with query.

    Arguments:
        recommendations: DataFrame with 'genres_list' column.
        query_genres: List of genres from query movie.
        k: Number of top recommendations to consider.

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if len(recommendations) == 0 or k == 0:
        return 0.0

    query_set = set(query_genres)
    top_k = recommendations.head(k)

    relevant = 0
    for _, row in top_k.iterrows():
        rec_genres = set(row['genres_list'])
        if len(query_set & rec_genres) > 0:
            relevant += 1

    return relevant / k


def genre_jaccard_at_k(
    recommendations: pd.DataFrame,
    query_genres: List[str],
    k: int
) -> float:
    """Calculate average Jaccard similarity for genres in top K.

    Measures genre similarity as intersection over union.

    Arguments:
        recommendations: DataFrame with genres_list column.
        query_genres: List of genres from query movie.
        k: Number of top recommendations to consider.

    Returns:
        Average Jaccard similarity (0.0 to 1.0).
    """
    if len(recommendations) == 0 or k == 0:
        return 0.0

    query_set = set(query_genres)
    top_k = recommendations.head(k)

    jaccards = []
    for _, row in top_k.iterrows():
        rec_genres = set(row['genres_list'])
        intersection = len(query_set & rec_genres)
        union = len(query_set | rec_genres)
        jaccard = intersection / union if union > 0 else 0.0
        jaccards.append(jaccard)

    return np.mean(jaccards)


def content_diversity_at_k(
    retriever,
    recommendations: pd.DataFrame,
    k: int
) -> float:
    """Calculate content diversity as average pairwise distance.

    Measures how different the recommended movies are from each other.
    Uses semantic embeddings for fair comparison across all methods.

    Arguments:
        retriever: Fitted retriever instance (with df attribute).
        recommendations: DataFrame with recommended movies.
        k: Number of top recommendations to consider.

    Returns:
        Average pairwise cosine distance (0.0 to 1.0).
    """

    top_k = recommendations.head(k)

    if len(top_k) < 2:
        return 0.0

    indices = top_k.index.tolist()

    model = _get_sentence_transformer()

    contents = [retriever.df.iloc[i]['content'] for i in indices]
    embeddings = model.encode(contents,
                              convert_to_numpy=True,
                              show_progress_bar=False)

    similarities = cosine_similarity(embeddings, embeddings)

    distances = []
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            distance = 1 - similarities[i][j]
            distances.append(distance)

    if len(distances) == 0:
        return 0.0

    return np.mean(distances)


def evaluate_retriever(
    retriever,
    test_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Evaluate retriever on test set with precision and diversity.

    Arguments:
        retriever: Fitted retriever instance.
        test_df: Test DataFrame with movies to query.
        k_values: List of K values to evaluate.

    Returns:
        Dictionary with metric names and average scores.
    """
    results = {f'precision@{k}': [] for k in k_values}
    results.update({f'jaccard@{k}': [] for k in k_values})
    results.update({f'content_div@{k}': [] for k in k_values})
    results['query_time_ms'] = []

    for _, row in test_df.iterrows():
        title = row['original_title']
        query_genres = row['genres_list']

        try:
            start = time.perf_counter()
            recs = retriever.recommend(title, top_k=max(k_values))
            query_time = (time.perf_counter() - start) * 1000
            results['query_time_ms'].append(query_time)

            for k in k_values:
                prec = genre_precision_at_k(recs, query_genres, k)
                results[f'precision@{k}'].append(prec)

                jacc = genre_jaccard_at_k(recs, query_genres, k)
                results[f'jaccard@{k}'].append(jacc)

                content_div = content_diversity_at_k(retriever, recs, k)
                results[f'content_div@{k}'].append(content_div)
        except:
            continue

    return {k: np.mean(v) for k, v in results.items()}
