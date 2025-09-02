import os
import glob
import re
from typing import List, Dict, Tuple, Optional, Set, Counter
from collections import defaultdict
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hdbscan

    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

from app.utils import detect_language, get_stop_words


class TopicClusterer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.files_data = {}
        self.cluster_explanations = {}

    def _detect_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load_files(self, input_dir: str, pattern: str = "*.txt") -> int:
        """Load all text files from directory, return count of loaded files"""
        paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
        self.files_data = {}

        for path in paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        filename = os.path.basename(path)
                        self.files_data[filename] = content
            except Exception as e:
                print(f"Error loading {path}: {e}")

        return len(self.files_data)

    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def extract_topic_segments(self, topic: str, context_sentences: int = 3, min_chars: int = 50) -> Dict[
        str, List[str]]:
        """Extract segments discussing the given topic"""
        topic_segments = {}
        topic_lower = topic.lower()

        for filename, content in self.files_data.items():
            segments = []
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]

            topic_indices = []
            for i, sentence in enumerate(sentences):
                if topic_lower in sentence.lower():
                    topic_indices.append(i)

            for idx in topic_indices:
                start_idx = max(0, idx - context_sentences)
                end_idx = min(len(sentences), idx + context_sentences + 1)
                segment = ' '.join(sentences[start_idx:end_idx]).strip()

                if len(segment) >= min_chars and segment not in segments:
                    segments.append(segment)

            if segments:
                topic_segments[filename] = segments

        return topic_segments

    def cluster_segments(self, topic_segments: Dict[str, List[str]], method: str = "kmeans",
                         n_clusters: Optional[int] = None) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """Cluster the extracted segments semantically"""
        if not topic_segments:
            raise ValueError("No segments to cluster")

        segments = []
        source_files = []

        for filename, file_segments in topic_segments.items():
            for segment in file_segments:
                segments.append(segment)
                source_files.append(filename)

        self._load_model()
        embeddings = self.model.encode(segments, convert_to_numpy=True)
        embeddings = normalize(embeddings, axis=1)

        n_segments = len(segments)
        if n_clusters is None:
            n_clusters = min(max(2, n_segments // 3), 8)
        n_clusters = min(n_clusters, n_segments - 1) if n_segments > 1 else 1

        if method == "hdbscan" and _HDBSCAN_AVAILABLE and n_segments > 4:
            min_cluster_size = max(2, n_segments // 4)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=1,
                                        metric="euclidean")
            labels = clusterer.fit_predict(embeddings)

            if len(set(labels)) <= 1 or np.all(labels == -1):
                method = "kmeans"

        if method == "kmeans" or not _HDBSCAN_AVAILABLE:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

        return segments, source_files, labels, embeddings

    def extract_cluster_keywords(self, segments: List[str], labels: np.ndarray, top_k: int = 5) -> Dict[int, List[str]]:
        """Extract top keywords for each cluster using TF-IDF"""
        if not segments:
            return {}

        language = detect_language(" ".join(segments[:10]))
        stop_words = list(get_stop_words(language))

        try:
            vectorizer = TfidfVectorizer(stop_words=stop_words,
                                         max_features=10000,
                                         ngram_range=(1, 2),
                                         min_df=1,
                                         max_df=0.8)
            tfidf_matrix = vectorizer.fit_transform(segments)
            feature_names = vectorizer.get_feature_names_out()

            cluster_keywords = {}
            for cluster_id in sorted(set(labels)):
                if cluster_id == -1:
                    cluster_keywords[cluster_id] = ["noise", "outliers"]
                    continue

                cluster_mask = labels == cluster_id
                cluster_tfidf = tfidf_matrix[cluster_mask]
                mean_scores = np.asarray(cluster_tfidf.mean(axis=0)).flatten()

                top_indices = mean_scores.argsort()[-top_k:][::-1]
                cluster_keywords[cluster_id] = [feature_names[i] for i in top_indices if len(feature_names[i]) > 2]

            return cluster_keywords

        except Exception:
            return self._simple_keyword_extraction(segments, labels, top_k, stop_words)

    def _simple_keyword_extraction(self, segments: List[str], labels: np.ndarray,
                                   top_k: int, stop_words: Set[str]) -> Dict[int, List[str]]:
        """Simple keyword extraction fallback"""
        cluster_keywords = {}
        for cluster_id in sorted(set(labels)):
            cluster_segments = [segments[i] for i, label in enumerate(labels) if label == cluster_id]

            word_counts = Counter()
            for segment in cluster_segments:
                words = re.findall(r'\b[\wÀ-ÖØ-öø-ÿ]+\b', segment.lower())
                words = [w for w in words if len(w) > 2 and w not in stop_words]
                word_counts.update(words)

            cluster_keywords[cluster_id] = [word for word, _ in word_counts.most_common(top_k)]

        return cluster_keywords

    def explain_clusters(self, segments: List[str], labels: np.ndarray, embeddings: np.ndarray,
                         cluster_keywords: Dict[int, List[str]]) -> Dict[int, Dict[str, any]]:
        """Generate explanations for why segments are grouped together"""
        explanations = {}

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip outliers
                explanations[cluster_id] = {
                    'theme': 'Outliers/Noise',
                    'explanation': 'These segments were too different from others to form coherent clusters.',
                    'coherence_score': 0.0,
                    'common_phrases': [],
                    'semantic_similarity': 'Low'
                }
                continue

            # Get segments in this cluster
            cluster_mask = labels == cluster_id
            cluster_segments = [segments[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
            cluster_embeddings = embeddings[cluster_mask]

            # Calculate intra-cluster similarity
            if len(cluster_embeddings) > 1:
                similarity_matrix = cosine_similarity(cluster_embeddings)
                # Remove diagonal (self-similarity)
                mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
                avg_similarity = similarity_matrix[mask].mean()
                coherence_score = float(avg_similarity)
            else:
                coherence_score = 1.0

            # Find common phrases/patterns
            common_phrases = self._find_common_phrases(cluster_segments)

            # Generate thematic explanation
            theme = self._generate_cluster_theme(cluster_keywords.get(cluster_id, []), common_phrases)

            # Generate detailed explanation
            explanation = self._generate_detailed_explanation(
                cluster_segments, cluster_keywords.get(cluster_id, []),
                common_phrases, coherence_score
            )

            explanations[cluster_id] = {
                'theme': theme,
                'explanation': explanation,
                'coherence_score': coherence_score,
                'common_phrases': common_phrases,
                'semantic_similarity': 'High' if coherence_score > 0.7 else 'Medium' if coherence_score > 0.5 else 'Low',
                'segment_count': len(cluster_segments),
                'keywords': cluster_keywords.get(cluster_id, [])
            }

        return explanations

    def _find_common_phrases(self, segments: List[str], min_length: int = 3, max_phrases: int = 5) -> List[str]:
        """Find common phrases across segments in a cluster"""
        if len(segments) < 2:
            return []

        # Extract potential phrases (3-5 words)
        phrase_counts = Counter()

        for segment in segments:
            words = re.findall(r'\b[\wÀ-ÖØ-öø-ÿ]+\b', segment.lower())
            # Generate n-grams
            for n in range(min_length, min(6, len(words))):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i + n])
                    if len(phrase) > 10:  # Only consider substantial phrases
                        phrase_counts[phrase] += 1

        # Filter phrases that appear in multiple segments
        min_occurrences = max(2, len(segments) // 2)
        common_phrases = [phrase for phrase, count in phrase_counts.items()
                          if count >= min_occurrences]

        return common_phrases[:max_phrases]

    def _generate_cluster_theme(self, keywords: List[str], common_phrases: List[str]) -> str:
        """Generate a thematic name/description for the cluster"""
        if not keywords and not common_phrases:
            return "Mixed Content"

        # Combine keywords and phrases for theme generation
        all_terms = keywords[:3] + [phrase.split()[0] for phrase in common_phrases[:2]]

        if len(all_terms) >= 2:
            return f"Focus on {', '.join(all_terms[:2])}"
        elif all_terms:
            return f"Related to {all_terms[0]}"
        else:
            return "Thematically Related Content"

    def _generate_detailed_explanation(self, segments: List[str], keywords: List[str],
                                       common_phrases: List[str], coherence_score: float) -> str:
        """Generate detailed explanation for why segments are clustered together"""
        explanation_parts = []

        # Start with cluster composition
        explanation_parts.append(f"This cluster contains {len(segments)} segments that were grouped together because:")

        # Semantic similarity explanation
        if coherence_score > 0.7:
            explanation_parts.append("• They show high semantic similarity in their content and meaning")
        elif coherence_score > 0.5:
            explanation_parts.append("• They share moderate semantic similarity in their topics and themes")
        else:
            explanation_parts.append("• They share some common thematic styled_elements, though less tightly related")

        # Keyword-based explanation
        if keywords:
            explanation_parts.append(f"• They frequently discuss: {', '.join(keywords[:3])}")

        # Common phrases explanation
        if common_phrases:
            explanation_parts.append(f"• They share similar expressions or phrases")

        # Content pattern explanation
        if len(set(len(seg.split()) for seg in segments)) <= 2:
            explanation_parts.append("• They have similar content length and structure")

        # Conceptual coherence
        explanation_parts.append(f"• Coherence score: {coherence_score:.2f} (higher = more similar content)")

        return "\n".join(explanation_parts)

    def save_results(self, topic: str, segments: List[str], source_files: List[str],
                     labels: np.ndarray, cluster_keywords: Dict[int, List[str]],
                     output_file: str, explanations: Optional[Dict[int, Dict[str, any]]] = None):
        """Save results to CSV file with explanations"""
        data = []
        for i, (segment, source_file, cluster_id) in enumerate(zip(segments, source_files, labels)):
            keywords = ", ".join(cluster_keywords.get(cluster_id, []))

            # Add explanation data if available
            explanation_data = explanations.get(cluster_id, {}) if explanations else {}

            data.append({
                'segment_id': i,
                'source_file': source_file,
                'cluster': int(cluster_id),
                'cluster_theme': explanation_data.get('theme', ''),
                'cluster_explanation': explanation_data.get('explanation', ''),
                'coherence_score': explanation_data.get('coherence_score', 0.0),
                'semantic_similarity': explanation_data.get('semantic_similarity', ''),
                'cluster_keywords': keywords,
                'common_phrases': '; '.join(explanation_data.get('common_phrases', [])),
                'segment_text': segment,
                'segment_preview': segment[:200] + "..." if len(segment) > 200 else segment
            })

        df = pd.DataFrame(data)
        df = df.sort_values(['cluster', 'source_file'])
        df.to_csv(output_file, index=False)