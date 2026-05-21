#!/usr/bin/env python3
"""
Step 3: Sentence tokenization and contrastive-connector splitting

Tokenizes each speech with spaCy, then splits sentences on contrastive
connectors (but / however / while / although / ;) to isolate mixed-tone
clauses into separate scoreable fragments.

Rationale: a sentence like "The labor market is strong, but inflation has
fallen below target" contains opposing signals. Splitting it produces two
clean fragments that can each be scored independently.

Output (data/processed/):
    sentences_bernanke.csv
    sentences_yellen.csv
    sentences_powell.csv

Usage:
    python3 src/build_sentence_corpus.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

try:
    import spacy
except ImportError:
    sys.exit("pip install spacy && python3 -m spacy download en_core_web_sm")

ROOT       = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
SPEECH_DIR = ROOT / "data" / "raw" / "speeches"

CHAIRS = ["bernanke", "yellen", "powell"]

# ── Contrastive-connector split pattern ───────────────────────────────────────
# Each sub-pattern is designed to match the connector WITH surrounding
# punctuation so that neither resulting fragment inherits the connector word.
#
# Semicolons:  always separate independent clauses → always split
# but:         word boundary avoids "rebuttal"; comma optional
# however:     only split when preceded by , or ; — this avoids splitting
#              mid-sentence parentheticals like "X, however, Y" where
#              however modifies the whole clause rather than introducing
#              a contrasting second clause
# while:       comma required — avoids temporal "while we were doing X"
# although:    comma optional — usually concessive/contrastive in policy text

SPLIT_RE = re.compile(
    r';\s*|'                           # semicolons
    r',?\s+\bbut\b\s+|'               # ", but " / " but "
    r'[,;]\s+\bhowever\b[,\s]+|'      # ", however," / "; however,"
    r',\s+\bwhile\b\s+|'              # ", while "  (comma required)
    r',?\s+\balthough\b\s+',          # ", although " / " although "
    re.IGNORECASE
)

MIN_WORDS = 8  # discard fragments shorter than this (filters headers, stubs)

# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_speech_text(raw: str) -> str:
    """
    Strip the 5-line header (date / title / speaker / venue / 'Share'),
    remove inline footnote reference numbers, and normalise whitespace.
    """
    lines = raw.split("\n")
    body  = "\n".join(lines[5:])                      # skip header
    body  = re.sub(r"(?<!\w)\d+(?!\w)", " ", body)   # lone digits = footnote refs
    body  = re.sub(r"\s+", " ", body).strip()
    return body


# ── Splitting ──────────────────────────────────────────────────────────────────

def split_sentence(sentence: str) -> list[str]:
    """
    Apply contrastive-connector regex to one sentence.
    Returns only fragments that meet the minimum word threshold.
    """
    parts = SPLIT_RE.split(sentence)
    return [p.strip() for p in parts if len(p.split()) >= MIN_WORDS]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading spaCy model…")
    # Keep tok2vec + parser for accurate sentence detection; disable unused pipes
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "attribute_ruler", "lemmatizer"])
    # Increase max length just in case any speech is unusually long
    nlp.max_length = 2_000_000

    for chair in CHAIRS:
        speeches  = pd.read_csv(PROCESSED / f"speeches_{chair}.csv", parse_dates=["date"])
        rows      = []

        for _, speech in speeches.iterrows():
            fpath = SPEECH_DIR / speech["filename"]
            if not fpath.exists():
                print(f"  WARNING: missing file {speech['filename']}")
                continue

            raw  = fpath.read_text(errors="ignore")
            body = clean_speech_text(raw)
            doc  = nlp(body)

            for sent_id, sent in enumerate(doc.sents):
                fragments = split_sentence(sent.text)
                for frag_id, frag in enumerate(fragments):
                    rows.append({
                        "date":      speech["date"].date(),
                        "chair_key": chair,
                        "title":     speech["title"],
                        "filename":  speech["filename"],
                        "sent_id":   sent_id,
                        "frag_id":   frag_id,
                        "text":      frag,
                    })

        df       = pd.DataFrame(rows)
        out_path = PROCESSED / f"sentences_{chair}.csv"
        df.to_csv(out_path, index=False)

        # Summary stats
        speeches_n  = len(speeches)
        fragments_n = len(df)
        split_n     = len(df[df["frag_id"] > 0])          # fragments born from a split
        avg_per_sp  = fragments_n / speeches_n if speeches_n else 0

        print(
            f"{chair}: {speeches_n} speeches → {fragments_n} fragments "
            f"({split_n} from connector splits, avg {avg_per_sp:.0f}/speech) "
            f"→ {out_path.name}"
        )


if __name__ == "__main__":
    main()
