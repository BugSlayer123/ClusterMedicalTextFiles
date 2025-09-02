# Semantic Grouping of Text Files

This tool clusters text documents into semantically similar groups using [SentenceTransformers](https://www.sbert.net/) embeddings and clustering methods such as **KMeans** or **HDBSCAN**.  
It supports multilingual documents, with optional local translation to English.

---

## Features

- Load and process `.txt` files from a directory  
- Multilingual support with [SBERT multilingual models](https://www.sbert.net/docs/pretrained_models.html)  
- Optional translation (using Hugging Face Helsinki-NLP models, requires extra dependencies)  
- Document chunking for long texts  
- Clustering with KMeans (default) or HDBSCAN (auto cluster count)  
- Extract top keywords per cluster using TF-IDF or simple frequency analysis  
- Output a structured `.csv` with filenames, cluster assignments, previews, and keywords  

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-repo/semantic-group-files.git
cd semantic-group-files
pip install -r requirements.txt

pip install -r requirements.txt
```
## Example usage
```
python main.py --input_dir ./texts --output clusters.csv --translate 
```
Or with a multilingual model:
```
python main.py --input_dir ./texts --model "paraphrase-multilingual-MiniLM-L12-v2" --method kmeans --kmeans_k 6
```
