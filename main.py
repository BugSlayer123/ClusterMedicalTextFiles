#!/usr/bin/env python3
"""
Semantic grouping of text files using sentence-transformers.

Two recommended strategies:
  - Default: use a multilingual SBERT model (no translation) for cross-language clustering.
  - If --translate: attempt local translation (requires transformers + sentencepiece + torch).

Usage:
    python semantic_group_files_fixed.py --input_dir ./texts --output clusters.csv
    python semantic_group_files_fixed.py --input_dir ./texts --output clusters.csv --translate
    python semantic_group_files_fixed.py --input_dir ./texts --output clusters.csv --model paraphrase-multilingual-MiniLM-L12-v2

Notes:
  - If you want translation to work locally, install: pip install transformers sentencepiece torch
  - Best quick fix: use a multilingual model like "paraphrase-multilingual-MiniLM-L12-v2".
"""

import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# language detection
try:
    from langdetect import detect
    _LANGDETECT_AVAILABLE = True
except Exception:
    _LANGDETECT_AVAILABLE = False

# local translation (optional)
try:
    from transformers import pipeline
    import sentencepiece
    _TRANSLATION_AVAILABLE = True
except Exception:
    _TRANSLATION_AVAILABLE = False

# HDBSCAN for automatic cluster count detection
try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False


def load_text_files(input_dir, pattern="*.txt"):
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    texts = []
    filenames = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
            if len(txt) == 0:
                continue
            texts.append(txt)
            filenames.append(os.path.basename(p))
    return filenames, texts


def detect_language_for_docs(docs):
    """
    Try langdetect if available; otherwise fallback to the simple heuristic.
    Returns list of language codes: 'en', 'nl', 'fr', ...
    """
    langs = []
    if _LANGDETECT_AVAILABLE:
        for d in docs:
            try:
                lang = detect(d[:1000])  # short sample
                langs.append(lang)
            except Exception:
                langs.append("unknown")
    else:
        for doc in docs:
            txt = doc[:1000].lower()
            dutch_indicators = ["het", "een", "van", "de", "dat", "en", "is", "arts", "patiënt", "ik", "bij", "voor", "maar"]
            french_indicators = ["le", "de", "et", "à", "un", "il", "être", "en", "avoir", "que", "pour", "dans", "médecin", "patient"]
            english_indicators = ["the", "and", "of", "to", "in", "that", "for", "with", "patient", "doctor"]
            dcount = sum(1 for w in dutch_indicators if w in txt)
            fcount = sum(1 for w in french_indicators if w in txt)
            ecount = sum(1 for w in english_indicators if w in txt)
            mx = max(dcount, fcount, ecount)
            if mx == 0:
                langs.append("unknown")
            elif fcount == mx:
                langs.append("fr")
            elif dcount == mx:
                langs.append("nl")
            else:
                langs.append("en")
    return langs


def translate_to_english_local(texts, source_languages):
    """
    Translate texts to English using local Hugging Face models (Helsinki-NLP OPUS-MT).
    Requires: transformers and sentencepiece installed.
    If translator cannot be loaded or sentencepiece missing, returns originals and languages unchanged.
    """
    if not _TRANSLATION_AVAILABLE:
        print("Warning: local translation not available. Install transformers, sentencepiece and torch to enable it:")
        print("  pip install transformers sentencepiece torch")
        return texts, source_languages

    print("Loading translation pipelines (local)...")
    translators = {}

    model_map = {
        'nl': 'Helsinki-NLP/opus-mt-nl-en',
        'fr': 'Helsinki-NLP/opus-mt-fr-en',
    }

    unique_langs = set(source_languages)
    for code in unique_langs:
        if code in model_map:
            model_name = model_map[code]
            try:
                translators[code] = pipeline('translation', model=model_name, device=-1)
                print(f"Loaded translator for {code} -> en ({model_name})")
            except Exception as e:
                print(f"Could not load translator {model_name}: {e}")
                translators[code] = None

    translated_texts = []
    final_langs = []

    for txt, lang in zip(texts, source_languages):
        if lang not in translators or translators.get(lang) is None:
            # no translator for this language or translator failed: keep original
            translated_texts.append(txt)
            final_langs.append(lang)
            continue

        try:
            sentences = [s.strip() for s in txt.split('\n') if s.strip()]
            out_sentences = []
            for s in sentences:
                if len(s) == 0:
                    continue
                # truncate extremely long lines to avoid model issues
                piece = s[:800]
                res = translators[lang](piece)
                if isinstance(res, list) and len(res) > 0:
                    # translation pipeline returns dict with 'translation_text'
                    out_sentences.append(res[0].get('translation_text', piece))
                else:
                    out_sentences.append(piece)
            translated_texts.append("\n".join(out_sentences))
            final_langs.append('en')
        except Exception as e:
            print(f"Translation failed for a document (lang={lang}): {e}. Keeping original.")
            translated_texts.append(txt)
            final_langs.append(lang)

    return translated_texts, final_langs


def chunk_text(text, max_chars=1000):
    """
    Split by paragraphs (double-newline) then windows over long paragraphs.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text]
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                if end < len(p):
                    back = p.rfind(" ", start, end)
                    if back > start + int(max_chars * 0.4):
                        end = back
                chunk = p[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == start:
                    break
                start = end
    return chunks


def build_embeddings_for_documents(model, docs, max_chars=1000, batch_size=32, device=None, normalize_embeddings=True):
    """
    Chunk documents, embed chunks, aggregate to a single vector per doc (length-weighted mean).
    """
    all_chunks = []
    doc_idx = []
    chunks_per_doc = []

    for i, d in enumerate(docs):
        chunks = chunk_text(d, max_chars=max_chars)
        if len(chunks) == 0:
            chunks = [d[:max_chars]]
        chunks_per_doc.append(chunks)
        for c in chunks:
            all_chunks.append(c)
            doc_idx.append(i)

    if len(all_chunks) == 0:
        raise ValueError("No text chunks to embed")

    chunk_embeddings = model.encode(all_chunks,
                                    batch_size=batch_size,
                                    show_progress_bar=True,
                                    convert_to_numpy=True,
                                    device=device)

    # length-weighted aggr
    dim = chunk_embeddings.shape[1]
    n_docs = len(docs)
    doc_embeddings = np.zeros((n_docs, dim), dtype=np.float32)
    lengths = np.array([max(1, len(c)) for c in all_chunks], dtype=np.float32)

    for i, emb in enumerate(chunk_embeddings):
        d_i = doc_idx[i]
        doc_embeddings[d_i] += emb * lengths[i]

    total_len = np.zeros((n_docs,), dtype=np.float32)
    for idx, L in zip(doc_idx, lengths):
        total_len[idx] += L

    for i in range(n_docs):
        if total_len[i] > 0:
            doc_embeddings[i] /= total_len[i]
        else:
            doc_embeddings[i] = np.mean([chunk_embeddings[j] for j, k in enumerate(doc_idx) if k == i], axis=0)

    if normalize_embeddings:
        doc_embeddings = normalize(doc_embeddings, axis=1)

    return doc_embeddings, chunks_per_doc


def cluster_embeddings(doc_embeddings, method="hdbscan", kmeans_k=None, hdbscan_min_cluster_size=2):
    n_docs = doc_embeddings.shape[0]

    if method == "hdbscan":
        if not _HDBSCAN_AVAILABLE:
            print("hdbscan not installed, falling back to kmeans.")
            method = "kmeans"
        else:
            # choose parameters robustly for small datasets
            min_cluster_size = max(2, min(hdbscan_min_cluster_size, max(2, n_docs // 4)))
            min_samples = max(1, min_cluster_size - 1)
            print(f"Using HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       min_samples=min_samples,
                                       metric="euclidean",
                                       cluster_selection_epsilon=0.05,
                                       prediction_data=False)
            labels = clusterer.fit_predict(doc_embeddings)
            if len(set(labels)) <= 1 or np.all(labels == -1):
                print("HDBSCAN produced no meaningful clusters; falling back to K-Means")
                method = "kmeans"

    if method == "kmeans":
        # estimate k if not provided
        if kmeans_k is None:
            k = min(max(2, n_docs // 2), max(2, min(8, n_docs)))
        else:
            k = min(int(kmeans_k), max(2, n_docs))
        if k >= n_docs:
            k = max(2, n_docs - 1)
        print(f"Using KMeans with k={k}")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(doc_embeddings)

    return labels


def detect_language_and_get_stop_words(docs):
    langs = detect_language_for_docs(docs)
    from collections import Counter
    c = Counter(langs)
    main = c.most_common(1)[0][0]
    print("Detected languages:", dict(c))
    # return stop words list or 'english' string
    if main == 'fr':
        french_stop_words = [
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son',
            'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'mais', 'par', 'plus', 'pouvoir', 'dire', 'me',
            'on', 'mon', 'lui', 'nous', 'comme', 'mes', 'te', 'ses', 'tes', 'leur'
        ]
        return french_stop_words, langs
    elif main == 'nl':
        dutch_stop_words = [
            'de', 'het', 'een', 'van', 'en', 'dat', 'die', 'in', 'op', 'voor', 'met', 'als', 'zijn', 'hebben', 'door',
            'aan', 'bij', 'na', 'over', 'deze', 'dit', 'maar', 'ook', 'niet', 'zo'
        ]
        return dutch_stop_words, langs
    else:
        return "english", langs


def extract_top_keywords_per_cluster(docs, labels, top_n=8, ngram_range=(1, 2)):
    stop_words, detected_languages = detect_language_and_get_stop_words(docs)
    vectorizer = TfidfVectorizer(stop_words=stop_words,
                                 max_features=20000,
                                 ngram_range=ngram_range,
                                 min_df=1,
                                 max_df=0.8)
    try:
        X = vectorizer.fit_transform(docs)
        terms = np.array(vectorizer.get_feature_names_out())
    except Exception as e:
        print("TF-IDF failed:", e)
        return extract_keywords_simple(docs, labels, top_n)

    cluster_keywords = {}
    for cl in sorted(set(labels)):
        mask = (labels == cl)
        if np.sum(mask) == 0:
            cluster_keywords[cl] = []
            continue
        summed = np.asarray(X[mask].sum(axis=0)).ravel()
        top_idx = np.argsort(summed)[::-1][:top_n]
        kws = [terms[i] for i in top_idx if len(terms[i]) > 2]
        cluster_keywords[cl] = kws[:top_n]
    return cluster_keywords


def extract_keywords_simple(docs, labels, top_n=8):
    from collections import Counter
    import re
    detected_stop_words, _ = detect_language_and_get_stop_words(docs)
    if isinstance(detected_stop_words, list):
        stop_words = set(detected_stop_words[:40])
    else:
        stop_words = set(['the', 'and', 'of', 'to', 'in', 'le', 'de', 'et', 'het', 'een', 'van'])
    cluster_keywords = {}
    for cl in sorted(set(labels)):
        mask = (labels == cl)
        docs_in = [docs[i] for i, m in enumerate(mask) if m]
        if not docs_in:
            cluster_keywords[cl] = []
            continue
        text = " ".join(docs_in).lower()
        words = re.findall(r'\b[\wÀ-ÖØ-öø-ÿ]+\b', text)
        words = [w for w in words if w not in stop_words and len(w) > 2]
        counter = Counter(words)
        cluster_keywords[cl] = [w for w, _ in counter.most_common(top_n)]
    return cluster_keywords


def print_similarity_matrix(doc_embeddings, filenames):
    sim = cosine_similarity(doc_embeddings)
    print("\nSimilarity matrix (cosine):")
    short_names = [f if len(f) <= 12 else f[:9] + "..." for f in filenames]
    print("Files:", short_names)
    for i, row in enumerate(sim):
        print(f"{short_names[i]:>12}: {' '.join([f'{v:.2f}' for v in row])}")


def main(args):
    print("Loading files...")
    filenames, docs = load_text_files(args.input_dir, pattern=args.pattern)
    if len(docs) == 0:
        raise SystemExit("No files found in directory.")
    print(f"Found {len(docs)} documents")

    # language detection
    detected_langs = detect_language_for_docs(docs)
    print("Per-document detected languages (sample):", detected_langs[:10])

    if args.translate:
        print("Translate flag enabled: attempting local translation to English...")
        docs, final_langs = translate_to_english_local(docs, detected_langs)
        print("Translation finished. Final languages:", final_langs[:10])

    print("Loading model:", args.model)
    # detect device automatically for sentence-transformers (encode device param)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Torch device:", device)
    except Exception:
        device = None

    model = SentenceTransformer(args.model)

    print("Building document embeddings...")
    doc_embeddings, chunks_per_doc = build_embeddings_for_documents(
        model, docs,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        device=device,
        normalize_embeddings=True
    )

    if args.debug:
        print_similarity_matrix(doc_embeddings, filenames)

    print("Clustering embeddings using method:", args.method)
    labels = cluster_embeddings(doc_embeddings,
                                method=args.method,
                                kmeans_k=args.kmeans_k,
                                hdbscan_min_cluster_size=args.hdbscan_min_cluster_size)

    print("Extracting keywords per cluster...")
    cluster_keywords = extract_top_keywords_per_cluster(docs, labels, top_n=args.top_n_keywords)

    rows = []
    for f, txt, lab in zip(filenames, docs, labels):
        preview = txt.strip().replace("\n", " ")[:400]
        rows.append({"filename": f, "cluster": int(lab), "preview": preview})

    df = pd.DataFrame(rows)
    df["cluster_keywords"] = df["cluster"].map(lambda c: ", ".join(cluster_keywords.get(c, [])))
    df = df.sort_values(['cluster', 'filename'])
    df.to_csv(args.output, index=False)
    print(f"Saved cluster CSV: {args.output}")

    print("\nCluster summary:")
    for cl in sorted(set(labels)):
        count = int(np.sum(labels == cl))
        kw = cluster_keywords.get(cl, [])
        files_in = [fn for fn, lb in zip(filenames, labels) if lb == cl]
        name = f"Cluster {cl}" if cl != -1 else "Noise/Outliers"
        print(f"  {name}: {count} files")
        print(f"    Files: {', '.join(files_in)}")
        print(f"    Keywords: {', '.join(kw[:6])}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory containing text files")
    p.add_argument("--pattern", default="*.txt", help="glob pattern for files")
    p.add_argument("--output", default="clusters.csv", help="Output CSV path")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model")
    p.add_argument("--method", default="kmeans", choices=["hdbscan", "kmeans"], help="Clustering method")
    p.add_argument("--kmeans_k", default=None, type=int, help="K for KMeans (auto-detected if not specified)")
    p.add_argument("--hdbscan_min_cluster_size", default=2, type=int, help="min cluster size for HDBSCAN")
    p.add_argument("--max_chars", default=1000, type=int, help="max chars per chunk")
    p.add_argument("--batch_size", default=32, type=int, help="batch size for embedding")
    p.add_argument("--top_n_keywords", default=8, type=int, help="top keywords per cluster")
    p.add_argument("--debug", action="store_true", help="Print similarity matrix for debugging")
    p.add_argument("--translate", action="store_true", help="If True, try to translate non-English docs to English (requires transformers + sentencepiece)")
    args = p.parse_args()
    main(args)
