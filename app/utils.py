import re
from typing import List, Set
from collections import Counter


def detect_language(text: str) -> str:
    """Simple language detection based on common words"""
    text_lower = text.lower()

    dutch_indicators = ["het", "een", "van", "de", "dat", "en", "is", "voor", "maar", "ook", "niet", "zo"]
    english_indicators = ["the", "and", "of", "to", "in", "that", "for", "with", "patient", "doctor"]

    dutch_count = sum(1 for w in dutch_indicators if w in text_lower)
    english_count = sum(1 for w in english_indicators if w in text_lower)

    return "dutch" if dutch_count > english_count else "english"


def get_stop_words(language: str) -> Set[str]:
    """Get stop words for a given language"""
    if language == "dutch":
        return {"het", "een", "van", "de", "dat", "en", "is", "voor", "maar", "ook",
                "niet", "zo", "die", "dit", "deze", "zijn", "hebben", "wordt", "kan",
                "wel", "nog", "bij", "op", "over"}
    else:
        return {"the", "and", "of", "to", "in", "that", "for", "with", "patient", "doctor",
                "is", "are", "was", "were", "have", "has", "had", "will", "would", "could",
                "should", "this", "that", "these", "those"}