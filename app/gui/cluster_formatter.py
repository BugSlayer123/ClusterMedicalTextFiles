class ClusterResultFormatter:
    def __init__(self, result, topic_text, method, context_sentences):
        self.result = result
        self.topic_text = topic_text
        self.method = method
        self.context_sentences = context_sentences

    def generate_summary_report(self):
        n_clusters = len(set(self.result['labels']))
        n_segments = len(self.result['segments'])
        n_files = len(set(self.result['source_files']))

        summary = f"CLUSTERING ANALYSIS REPORT\n"
        summary += "=" * 50 + "\n\n"
        summary += f"OVERVIEW:\n"
        summary += f"• {n_segments} segments found across {n_files} files\n"
        summary += f"• {n_clusters} clusters identified\n"
        summary += f"• Topic: '{self.topic_text}'\n\n"

        for cluster_id in sorted(set(self.result['labels'])):
            summary += self._format_cluster_section(cluster_id)

        return summary

    def _format_cluster_section(self, cluster_id):
        cluster_mask = self.result['labels'] == cluster_id
        cluster_segments = [self.result['segments'][i] for i in range(len(self.result['segments'])) if cluster_mask[i]]
        cluster_files = [self.result['source_files'][i] for i in range(len(self.result['source_files'])) if
                         cluster_mask[i]]
        explanation = self.result['explanations'].get(cluster_id, {})

        section = f"CLUSTER {cluster_id}" if cluster_id != -1 else "OUTLIERS"
        section += f" - {explanation.get('theme', 'Unknown')}\n"
        section += "-" * 40 + "\n"

        section += f"GROUPING RATIONALE:\n"
        section += f"{explanation.get('explanation', 'No explanation available')}\n\n"

        section += f"METRICS:\n"
        section += f"• Segments: {len(cluster_segments)}\n"
        section += f"• Coherence Score: {explanation.get('coherence_score', 0):.3f}\n"
        section += f"• Semantic Similarity: {explanation.get('semantic_similarity', 'Unknown')}\n"
        section += f"• Source Files: {', '.join(set(cluster_files))}\n\n"

        keywords = self.result['cluster_keywords'].get(cluster_id, [])
        if keywords:
            section += f"KEY TERMS: {', '.join(keywords[:5])}\n"

        common_phrases = explanation.get('common_phrases', [])
        if common_phrases:
            section += f"COMMON PHRASES: {'; '.join(common_phrases[:3])}\n"

        section += f"\nEXAMPLE SEGMENTS:\n"
        for i, segment in enumerate(cluster_segments[:2]):
            preview = segment[:150] + "..." if len(segment) > 150 else segment
            section += f"  {i + 1}. {preview}\n\n"

        if len(cluster_segments) > 2:
            section += f"  ... and {len(cluster_segments) - 2} more segments\n\n"

        section += "\n" + "=" * 50 + "\n\n"
        return section

    def generate_explanations_report(self):
        content = "CLUSTER FORMATION EXPLANATIONS\n"
        content += "=" * 50 + "\n\n"
        content += "This section explains the AI's reasoning for creating each cluster:\n\n"

        for cluster_id in sorted(set(self.result['labels'])):
            if cluster_id == -1:
                continue
            content += self._format_explanation_section(cluster_id)

        return content

    def _format_explanation_section(self, cluster_id):
        explanation = self.result['explanations'].get(cluster_id, {})
        cluster_mask = self.result['labels'] == cluster_id
        cluster_size = sum(cluster_mask)

        section = f"CLUSTER {cluster_id}: {explanation.get('theme', 'Unknown Theme')}\n"
        section += "-" * 30 + "\n"

        section += f"AI REASONING:\n"
        section += f"{explanation.get('explanation', 'No detailed explanation available.')}\n\n"

        section += f"SIMILARITY ANALYSIS:\n"
        coherence = explanation.get('coherence_score', 0)
        section += self._interpret_coherence_score(coherence)

        section += f"• Cluster contains {cluster_size} segments\n"
        section += f"• Semantic relationship strength: {explanation.get('semantic_similarity', 'Unknown')}\n\n"

        if explanation.get('common_phrases'):
            section += f"SHARED LANGUAGE PATTERNS:\n"
            for phrase in explanation.get('common_phrases', [])[:3]:
                section += f"• \"{phrase}\"\n"
            section += "\n"

        section += "\n" + "-" * 50 + "\n\n"
        return section

    def _interpret_coherence_score(self, coherence):
        if coherence > 0.7:
            return f"• Very high content similarity ({coherence:.3f}) - segments discuss very similar concepts\n"
        elif coherence > 0.5:
            return f"• Moderate content similarity ({coherence:.3f}) - segments share common themes\n"
        else:
            return f"• Lower content similarity ({coherence:.3f}) - segments grouped by weaker associations\n"


class ClusterDetailGenerator:
    def __init__(self, result, cluster_id, method, context_sentences, topic_text):
        self.result = result
        self.cluster_id = cluster_id
        self.method = method
        self.context_sentences = context_sentences
        self.topic_text = topic_text

    def generate_details(self):
        if self.cluster_id == -1:
            return self._generate_outlier_details()
        return self._generate_cluster_details()

    def _generate_outlier_details(self):
        cluster_mask = self.result['labels'] == self.cluster_id
        cluster_segments = [self.result['segments'][i] for i in range(len(self.result['segments'])) if cluster_mask[i]]
        cluster_files = [self.result['source_files'][i] for i in range(len(self.result['source_files'])) if
                         cluster_mask[i]]

        details = "OUTLIERS CLUSTER\n"
        details += "=" * 30 + "\n\n"
        details += "These segments were identified as outliers because they don't fit well into any coherent thematic group.\n\n"
        details += self._add_file_distribution(cluster_files, cluster_segments)
        details += self._add_sample_segments(cluster_segments, cluster_files)

        return details

    def _generate_cluster_details(self):
        explanation = self.result['explanations'].get(self.cluster_id, {})
        cluster_mask = self.result['labels'] == self.cluster_id
        cluster_segments = [self.result['segments'][i] for i in range(len(self.result['segments'])) if cluster_mask[i]]
        cluster_files = [self.result['source_files'][i] for i in range(len(self.result['source_files'])) if
                         cluster_mask[i]]

        details = f"CLUSTER {self.cluster_id}: {explanation.get('theme', 'Unknown Theme')}\n"
        details += "=" * 50 + "\n\n"

        details += "CLUSTER RATIONALE:\n"
        details += "-" * 25 + "\n"
        details += f"{explanation.get('explanation', 'No explanation available')}\n\n"

        details += self._add_analysis_metrics(explanation, cluster_segments, cluster_files)
        details += self._add_keywords_and_phrases(explanation)
        details += self._add_file_distribution(cluster_files, cluster_segments)
        details += self._add_sample_segments(cluster_segments, cluster_files)
        details += self._add_technical_insights(explanation, cluster_segments)

        return details

    def _add_analysis_metrics(self, explanation, cluster_segments, cluster_files):
        details = "CLUSTER ANALYSIS:\n"
        details += "-" * 18 + "\n"
        coherence = explanation.get('coherence_score', 0)
        details += f"• Coherence Score: {coherence:.4f}\n"
        details += f"• Interpretation: {self._interpret_coherence(coherence)}\n"
        details += f"• Semantic Similarity: {explanation.get('semantic_similarity', 'Unknown')}\n"
        details += f"• Number of Segments: {len(cluster_segments)}\n"
        details += f"• Source Files: {len(set(cluster_files))}\n\n"
        return details

    def _interpret_coherence(self, coherence):
        if coherence > 0.8:
            return "Extremely coherent - segments are very similar"
        elif coherence > 0.7:
            return "Highly coherent - segments discuss very similar topics"
        elif coherence > 0.6:
            return "Well coherent - segments share clear common themes"
        elif coherence > 0.5:
            return "Moderately coherent - segments have some thematic overlap"
        elif coherence > 0.4:
            return "Loosely coherent - segments weakly related"
        else:
            return "Low coherence - segments may be diverse or outliers"

    def _add_keywords_and_phrases(self, explanation):
        details = ""
        keywords = self.result['cluster_keywords'].get(self.cluster_id, [])
        if keywords:
            details += "DISTINGUISHING KEYWORDS:\n"
            details += "-" * 25 + "\n"
            for i, keyword in enumerate(keywords[:8], 1):
                details += f"{i}. {keyword}\n"
            details += "\n"

        common_phrases = explanation.get('common_phrases', [])
        if common_phrases:
            details += "RECURRING PHRASES:\n"
            details += "-" * 20 + "\n"
            for i, phrase in enumerate(common_phrases[:5], 1):
                details += f"{i}. \"{phrase}\"\n"
            details += "\n"
        return details

    def _add_file_distribution(self, cluster_files, cluster_segments):
        file_counts = {}
        for file in cluster_files:
            file_counts[file] = file_counts.get(file, 0) + 1

        details = "SOURCE FILE DISTRIBUTION:\n"
        details += "-" * 28 + "\n"
        for file, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(cluster_segments)) * 100
            details += f"• {file}: {count} segments ({percentage:.1f}%)\n"
        details += "\n"
        return details

    def _add_sample_segments(self, cluster_segments, cluster_files):
        details = "REPRESENTATIVE SEGMENTS:\n"
        details += "-" * 28 + "\n"
        for i, segment in enumerate(cluster_segments[:3], 1):
            details += f"SEGMENT {i}:\n"
            details += f"Source: {cluster_files[cluster_segments.index(segment)]}\n"
            details += f"Content: {segment[:300]}{'...' if len(segment) > 300 else ''}\n\n"

        if len(cluster_segments) > 3:
            details += f"... and {len(cluster_segments) - 3} more segments in this cluster.\n\n"
        return details

    def _add_technical_insights(self, explanation, cluster_segments):
        coherence = explanation.get('coherence_score', 0)
        details = "TECHNICAL INSIGHTS:\n"
        details += "-" * 21 + "\n"
        details += f"• Clustering Method: {self.method.upper()}\n"
        details += f"• Context Window: ±{self.context_sentences} sentences\n"
        details += f"• Topic Search: '{self.topic_text}'\n"
        details += f"• This cluster was formed because the AI detected semantic patterns\n"
        details += f"  that distinguish these {len(cluster_segments)} segments from others.\n"
        details += f"• The coherence score of {coherence:.3f} indicates "
        details += "strong thematic unity.\n" if coherence > 0.6 else "moderate thematic connection.\n"
        return details