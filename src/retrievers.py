"""Movie recommendation retriever classes."""

# TODO: fix dulpicated code between classes with abstract class or standalone helpers

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class TFIDFRecommender:
    """A movie recommender based on TF-IDF and cosine similarity."""

    def __init__(self,
                 df: pd.DataFrame,
                 use_lemmatization: bool = True,
                 remove_stopwords: bool = True,
                 ngram_range: tuple = (1, 1),
                 min_length: int = 3,
                 use_alpha_filter: bool = True):
        
        self.df = df.copy().reset_index(drop=True)
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.ngram_range = ngram_range
        self.min_length = min_length
        self.use_alpha_filter = use_alpha_filter
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    def get_variant_name(self) -> str:
        lemma = "lemma" if self.use_lemmatization else "no_lemma"
        stop = "no_stop" if self.remove_stopwords else "keep_stop"
        ngram = "unigram" if self.ngram_range == (1, 1) else "bigram"
        min_len = f"min{self.min_length}"
        alpha = "alpha" if self.use_alpha_filter else "no_alpha"
        return f"tfidf_{lemma}_{stop}_{ngram}_{min_len}_{alpha}"

    def preprocess(self, text: str):
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if self.use_alpha_filter and not token.is_alpha:
                continue

            if len(token.text) < self.min_length:
                continue

            if self.remove_stopwords and token.is_stop:
                continue

            if self.use_lemmatization:
                tokens.append(token.lemma_.lower())
            else:
                tokens.append(token.text.lower())

        return tokens

    def fit(self):
        tfidf = TfidfVectorizer(
            tokenizer=self.preprocess,
            lowercase=False,
            token_pattern=None,
            ngram_range=self.ngram_range
        )

        self.tfidf_matrix = tfidf.fit_transform(self.df['content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['original_title']).drop_duplicates()

    def recommend(self, title: str, top_k: int = 5) -> pd.DataFrame:
        """Get movie recommendations based on similarity.

        Arguments:
            title: Movie title to find similarities for.
            top_k: Number of recommendations to return.

        Returns:
            DataFrame with recommended movies and similarity scores.
        """
        if title not in self.indices:
            print(f"Movie {title} not found in data")
            return pd.DataFrame()

        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.df.iloc[movie_indices].copy()
        results['similarity_score'] = scores

        cols = ['original_title', 'similarity_score', 'genres_list', 'overview']
        return results[cols]


class SemanticRecommender:
    """A movie recommender based on semantic embeddings and cosine similarity."""

    def __init__(self, df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2", token: str = None):
        self.df = df.copy().reset_index(drop=True)
        self.model_name = model_name
        self.token = token
        self.model = None
        self.embeddings = None
        self.cosine_sim = None
        self.indices = None

    def get_variant_name(self) -> str:
        name = self.model_name.replace("-", "_")
        return f"semantic_{name}"

    def encode_all(self):
        """Encode all movies and precompute similarity matrix."""

        self.model = SentenceTransformer(self.model_name, token=self.token)

        self.embeddings = self.model.encode(
            self.df['content'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.cosine_sim = cosine_similarity(self.embeddings, self.embeddings)

        self.indices = pd.Series(
            self.df.index,
            index=self.df['original_title']
        ).drop_duplicates()

    def recommend(self, title: str, top_k: int = 5) -> pd.DataFrame:
        """Get movie recommendations based on semantic similarity.

        Arguments:
            title: Movie title to find similarities for.
            top_k: Number of recommendations to return.

        Returns:
            DataFrame with recommended movies and similarity scores.
        """
        if title not in self.indices:
            print(f"Movie {title} not found in data.")
            return pd.DataFrame()

        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.df.iloc[movie_indices].copy()
        results['similarity_score'] = scores

        cols = ['original_title', 'similarity_score', 'genres_list', 'overview']
        return results[cols]


class BM25Recommender:
    """A movie recommender based on BM25 ranking algorithm."""

    def __init__(self,
                 df: pd.DataFrame,
                 k1: float = 1.5,
                 b: float = 0.75,
                 use_lemmatization: bool = True,
                 remove_stopwords: bool = True,
                 ngram_range: tuple = (1, 1),
                 min_length: int = 3,
                 use_alpha_filter: bool = True):

        self.df = df.copy().reset_index(drop=True)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
        self.k1 = k1
        self.b = b
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.ngram_range = ngram_range
        self.min_length = min_length
        self.use_alpha_filter = use_alpha_filter
        self.bm25 = None
        self.tokenized_corpus = None
        self.cosine_sim = None
        self.indices = None

    def get_variant_name(self) -> str:
        lemma = "lemma" if self.use_lemmatization else "no_lemma"
        stop = "no_stop" if self.remove_stopwords else "keep_stop"
        ngram = "unigram" if self.ngram_range == (1, 1) else "bigram"
        min_len = f"min{self.min_length}"
        alpha = "alpha" if self.use_alpha_filter else "no_alpha"
        return f"bm25_k{self.k1}_b{self.b}_{lemma}_{stop}_{ngram}_{min_len}_{alpha}"

    def preprocess(self, text: str):
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if self.use_alpha_filter and not token.is_alpha:
                continue

            if len(token.text) < self.min_length:
                continue

            if self.remove_stopwords and token.is_stop:
                continue

            if self.use_lemmatization:
                tokens.append(token.lemma_.lower())
            else:
                tokens.append(token.text.lower())

        return tokens

    def fit(self):
        """Fit BM25 model and precompute similarity matrix."""
        self.tokenized_corpus = [self.preprocess(doc) for doc in self.df['content']]

        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        n_docs = len(self.df)
        self.cosine_sim = np.zeros((n_docs, n_docs))

        for i in range(n_docs):
            query_tokens = self.tokenized_corpus[i]
            scores = self.bm25.get_scores(query_tokens)
            self.cosine_sim[i] = scores

        self.indices = pd.Series(self.df.index, index=self.df['original_title']).drop_duplicates(keep='first')

    def recommend(self, title: str, top_k: int = 5) -> pd.DataFrame:
        """Get movie recommendations based on BM25 similarity.

        Arguments:
            title: Movie title to find similarities for.
            top_k: Number of recommendations to return.

        Returns:
            DataFrame with recommended movies and similarity scores.
        """
        if title not in self.indices:
            print(f"Movie {title} not found in data")
            return pd.DataFrame()

        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.df.iloc[movie_indices].copy()
        results['similarity_score'] = scores

        cols = ['original_title', 'similarity_score', 'genres_list', 'overview']
        return results[cols]

class HybridRecommender:
    """A movie recommender that combines BM25 and semantic similarity scores."""

    def __init__(self, bm25_retriever, semantic_retriever, alpha=0.5):
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.alpha = alpha
        self.df = bm25_retriever.df
        self.indices = bm25_retriever.indices

    def get_variant_name(self):
        return f"hybrid_bm25_gemma_alpha{self.alpha}"

    def recommend(self, title: str, top_k: int = 5) -> pd.DataFrame:
        """Get movie recommendations based on hybrid similarity.

        Arguments:
            title: Movie title to find similarities for.
            top_k: Number of recommendations to return.

        Returns:
            DataFrame with recommended movies and similarity scores.
        """
        if title not in self.indices:
            print(f"Movie {title} not found in data")
            return pd.DataFrame()

        idx = self.indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        bm25_scores = self.bm25.cosine_sim[idx]
        semantic_scores = self.semantic.cosine_sim[idx]

        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)

        hybrid_scores = self.alpha * bm25_norm + (1 - self.alpha) * semantic_norm

        sim_scores = list(enumerate(hybrid_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        results = self.df.iloc[movie_indices].copy()
        results['similarity_score'] = scores

        cols = ['original_title', 'similarity_score', 'genres_list', 'overview']
        return results[cols]
