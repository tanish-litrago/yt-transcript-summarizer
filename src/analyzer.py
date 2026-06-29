import re
from collections import Counter

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","this","that","it","is","was","are","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","can","not","no","so","if","as","up","out",
    "he","she","they","we","you","i","me","us","him","her","them","his",
    "its","our","their","my","your","also","just","more","very","about",
    "like","than","then","when","where","which","who","what","how","all",
    "any","each","into","over","such","even","only","now","well","back",
    "after","here","there","because","through","during","before","both",
    "these","those","going","want","know","think","right","said","okay",
    "yeah","uh","um","actually","basically","really","kind","gonna","let",
    "say","see","look","get","got","one","two","three","four","five","way",
}


def word_frequency(text: str, top_n: int = 15) -> list:
    words = re.findall(r"[a-z]{3,}", text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    counts = Counter(filtered).most_common(top_n)
    return [{"word": w, "count": c} for w, c in counts]


def compression_ratio(original: str, summary: str) -> dict:
    orig_words = len(original.split())
    summ_words = len(summary.split())
    orig_chars = len(original)
    summ_chars = len(summary)
    ratio = round((1 - summ_words / max(orig_words, 1)) * 100, 1)
    return {
        "original_words" : orig_words,
        "summary_words"  : summ_words,
        "original_chars" : orig_chars,
        "summary_chars"  : summ_chars,
        "compression_pct": ratio,
    }


def readability(text: str) -> dict:
    if not HAS_TEXTSTAT:
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "reading_level": "N/A",
            "avg_sentence_length": 0,
            "avg_word_length": 0,
        }
    score = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)

    if score >= 90:   level = "Very Easy"
    elif score >= 70: level = "Easy"
    elif score >= 60: level = "Standard"
    elif score >= 50: level = "Fairly Difficult"
    elif score >= 30: level = "Difficult"
    else:             level = "Very Difficult"

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    avg_sent = round(len(words) / max(len(sentences), 1), 1)
    avg_word = round(sum(len(w) for w in words) / max(len(words), 1), 1)

    return {
        "flesch_reading_ease" : round(score, 1),
        "flesch_kincaid_grade": round(grade, 1),
        "reading_level"       : level,
        "avg_sentence_length" : avg_sent,
        "avg_word_length"     : avg_word,
    }


def sentiment_timeline(chunks: list) -> list:
    results = []
    positive_words = {
        "good","great","excellent","amazing","wonderful","best","better",
        "important","useful","helpful","clear","effective","powerful",
        "success","successful","improve","improved","learn","understand",
        "simple","easy","fast","efficient","accurate","correct","perfect",
        "interesting","exciting","remarkable","significant","strong",
    }
    negative_words = {
        "bad","worse","worst","difficult","hard","problem","issue","error",
        "fail","failure","wrong","incorrect","slow","expensive","complex",
        "confusing","unclear","impossible","never","not","no","cannot",
        "lose","lost","broken","missing","lack","without","limited",
    }
    for i, chunk in enumerate(chunks):
        words = re.findall(r"[a-z]+", chunk.lower())
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)
        total = max(pos + neg, 1)
        score = round((pos - neg) / total * 100, 1)
        if score > 10:   label = "Positive"
        elif score < -10: label = "Negative"
        else:             label = "Neutral"
        results.append({
            "section": f"Section {i+1}",
            "score"  : score,
            "label"  : label,
            "pos"    : pos,
            "neg"    : neg,
        })
    return results


def entity_type_distribution(entities_typed: list) -> list:
    """
    Groups Gemma's typed entities (from gemma_engine.extract_keywords)
    into counts per type for the Named Entity Types chart.

    entities_typed: list of {"name": str, "type": str} dicts.
    Replaces the old spaCy-based NER grouping removed in v2.0.
    """
    if not entities_typed:
        return [{"type": "None found", "count": 0, "pct": 0}]

    type_counts = Counter(e.get("type", "Other") for e in entities_typed)
    total = sum(type_counts.values())
    return [
        {"type": t, "count": c, "pct": round(c / total * 100, 1)}
        for t, c in type_counts.most_common()
    ]


def speaking_pace(transcript: str, duration_str: str) -> dict:
    words = len(transcript.split())
    try:
        parts = duration_str.replace(":", " ").split()
        if len(parts) == 2:
            minutes = int(parts[0]) + int(parts[1]) / 60
        else:
            minutes = int(parts[0]) / 60
        if minutes > 0:
            wpm = round(words / minutes)
            if wpm < 100:   pace = "Slow"
            elif wpm < 150: pace = "Normal"
            elif wpm < 200: pace = "Fast"
            else:           pace = "Very Fast"
            return {"wpm": wpm, "pace": pace, "duration_min": round(minutes, 1)}
    except Exception:
        pass
    return {"wpm": 0, "pace": "Unknown", "duration_min": 0}


def run_all(transcript: str, summary: str, chunks: list,
            entities_typed: list, duration_str: str) -> dict:
    return {
        "word_freq"   : word_frequency(transcript, 15),
        "compression" : compression_ratio(transcript, summary),
        "readability" : readability(summary),
        "sentiment"   : sentiment_timeline(chunks),
        "entity_types": entity_type_distribution(entities_typed),
        "pace"        : speaking_pace(transcript, duration_str),
    }
