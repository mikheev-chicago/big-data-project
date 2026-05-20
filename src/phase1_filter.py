#!/usr/bin/env python3
"""
Phase 1 — Filter non-monetary-policy speeches.

Two-stage classifier:
  Stage 1 (title rules):  fast, no API calls — catches clear non-monetary-policy
                          speeches by keyword matching on the title
  Stage 2 (LLM):          claude-opus-4-7 classifies ambiguous speeches using
                          title + first 400 words of speech text

FOMC-day check:
  Speeches whose date matches an FOMC meeting date are flagged. FOMC press
  conferences only began in April 2011; if a post-2011 FOMC-day speech has
  < 1,000 words it may be a fragment — those are excluded as a precaution.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python src/phase1_filter.py

Output:
    data/processed/speeches_filtered.csv        — all 242 rows with filter columns
    data/processed/speeches_monetary_policy.csv — kept speeches only
"""

import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import anthropic

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPEECHES_DIR  = BASE_DIR / "data" / "raw" / "speeches"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXCERPT_WORDS = 400

# ── Stage 1: title-based keyword rules ────────────────────────────────────────
# Conservative list — only titles that are unambiguously off-topic.
# Borderline cases fall through to Stage 2 (LLM).

_TITLE_EXCLUDE = [
    # Graduation / ceremonial
    "commencement",
    "baccalaureate",
    "class day",
    # Health policy
    "health-care",
    "healthcare",
    "health care reform",
    # Financial literacy (public education programs, not policy)
    "financial literacy",
    "jump$tart",
    "financial and economic education",
    # Education / childhood policy
    "early childhood education",
    # Workforce / social programs
    "workforce development",
    # Diversity / social
    "women's history month",
    "historically black colleges",
    "women's participation in the economy",
    # Administrative ceremonies
    "dedication of the new",
    "swearing-in ceremony",
    "ceremonial swearing-in",
    # Known off-topic one-offs
    "the economics of happiness",
    "asset building for low and middle income",
    "promoting research and development: the government",
]

_TITLE_INCLUDE = [
    # Unambiguous monetary-policy titles — skip LLM call
    "monetary policy",
    "economic outlook",
    "price stability",
    "restoring price stability",
    "inflation",
    "interest rate",
    "labor market dynamics",
    "federal reserve's balance sheet",
    "quantitative easing",
    "forward guidance",
    "the economy",
    "economic policy",
    "macroeconomic",
    "financial stability",
    "financial crisis",
    "economic recovery",
    "economic conditions",
]


def stage1_check(title: str) -> tuple[str, str]:
    """
    Returns:
      ('EXCLUDE', reason)   — title is clearly off-topic
      ('INCLUDE', reason)   — title is clearly monetary-policy relevant
      ('UNCERTAIN', '')     — ambiguous; needs Stage 2
    """
    lower = title.lower()
    for kw in _TITLE_EXCLUDE:
        if kw in lower:
            return "EXCLUDE", f"title keyword: '{kw}'"
    for kw in _TITLE_INCLUDE:
        if kw in lower:
            return "INCLUDE", f"title keyword: '{kw}'"
    return "UNCERTAIN", ""


# ── Stage 2: LLM classifier ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research assistant helping filter a corpus of Federal Reserve Chair speeches
for a study of monetary-policy hawkishness and its effect on 2-year Treasury yields.

Your task: decide whether each speech is RELEVANT or IRRELEVANT to monetary policy.

────────────────────────────────────────────────────────────────
RELEVANT — INCLUDE if the speech primarily discusses any of:
  • Monetary policy stance (interest rates, hiking/cutting cycles, rate path)
  • Economic outlook (GDP growth, inflation, unemployment, labor market)
  • Federal Reserve tools (QE, QT, forward guidance, balance sheet policy)
  • Financial stability from a policy standpoint (systemic risk, credit conditions,
    how financial stress affects the policy outlook)
  • Central banking philosophy, frameworks, mandates (dual mandate, AIT)
  • International economic conditions as they feed into U.S. policy decisions
  • Historical Fed policy analysis or lessons for current policymaking
  • Fiscal–monetary interactions that affect the Fed's policy space

IRRELEVANT — EXCLUDE if the speech is primarily about:
  • Financial literacy programs or public education for consumers
  • Community development grants, housing assistance, minority entrepreneurship
  • Workforce training, education pipeline, or social-mobility programs
  • Healthcare or social policy
  • Pure regulatory/supervisory procedures (e.g., stress-test methodology, bank
    examination practices) with no monetary policy discussion
  • Diversity, equity, or institutional culture at the Fed
  • Administrative ceremonies: building dedications, award acceptances
  • Conference welcoming/opening/closing remarks with no substantive content
    (brief introductions that merely set the stage for other speakers)
────────────────────────────────────────────────────────────────

NUANCES:
  • "Community banking" that discusses credit availability and monetary transmission
    → INCLUDE.
  • Short welcoming remarks that only introduce a conference and contain no
    original monetary-policy argument → EXCLUDE.
  • "Financial regulation" speeches that discuss systemic risk and its implications
    for policy → INCLUDE; those focused only on exam procedures → EXCLUDE.
  • When genuinely uncertain, lean toward INCLUDE — false positives are less
    harmful than false negatives for this research.

Respond with EXACTLY one of these two formats (nothing else):
  INCLUDE: <reason in 10 words or fewer>
  EXCLUDE: <reason in 10 words or fewer>
"""


def get_text_excerpt(filename: str) -> str:
    path = SPEECHES_DIR / filename
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return " ".join(text.split()[:EXCERPT_WORDS])


def stage2_llm(
    title: str,
    excerpt: str,
    client: anthropic.Anthropic,
    retries: int = 3,
) -> tuple[str, str]:
    """Call claude-opus-4-7. Returns ('INCLUDE'|'EXCLUDE', reason)."""
    user_msg = f"Speech title: {title}\n\nFirst {EXCERPT_WORDS} words of text:\n{excerpt}"

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=64,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text.strip()
            break
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            log.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        except anthropic.APIError as exc:
            if attempt == retries - 1:
                raise
            log.warning(f"API error ({exc}) — retrying")
            time.sleep(2 ** attempt)
    else:
        return "INCLUDE", "LLM unavailable — included by default"

    if text.upper().startswith("INCLUDE"):
        decision = "INCLUDE"
        reason = text.split(":", 1)[1].strip() if ":" in text else "LLM: relevant"
    elif text.upper().startswith("EXCLUDE"):
        decision = "EXCLUDE"
        reason = text.split(":", 1)[1].strip() if ":" in text else "LLM: not relevant"
    else:
        log.warning(f"Unexpected LLM response: {text!r} — defaulting to INCLUDE")
        decision = "INCLUDE"
        reason = "LLM ambiguous response — included by default"

    return decision, reason


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY not set.\n"
            "Get a key at: https://console.anthropic.com/settings/keys\n"
            "Then run:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    speeches = pd.read_csv(PROCESSED_DIR / "speeches.csv")
    fomc     = pd.read_csv(PROCESSED_DIR / "fomc_calendar.csv")
    speeches["date"] = pd.to_datetime(speeches["date"])
    fomc["date"]     = pd.to_datetime(fomc["date"])
    fomc_dates = set(fomc["date"])

    FOMC_PRESS_CONF_START = pd.Timestamp("2011-04-27")  # first FOMC presser

    results = []
    counts = {"s1_include": 0, "s1_exclude": 0, "s2_include": 0, "s2_exclude": 0, "fomc_exclude": 0}

    log.info(f"Processing {len(speeches)} speeches through Phase 1 filter...")

    for i, (_, row) in enumerate(speeches.iterrows(), 1):
        title    = row["title"]
        date     = row["date"]
        filename = row["filename"]
        fomc_day = date in fomc_dates

        decision = reason = stage = None

        # ── FOMC-day press conference check ──────────────────────────────────
        if fomc_day and date >= FOMC_PRESS_CONF_START and row["word_count"] < 1000:
            decision = "EXCLUDE"
            reason   = "FOMC-day speech post-Apr-2011 with <1,000 words — possible press conf fragment"
            stage    = "fomc_check"
            counts["fomc_exclude"] += 1

        else:
            # ── Stage 1: title keywords ───────────────────────────────────────
            s1, s1_reason = stage1_check(title)
            if s1 == "EXCLUDE":
                decision, reason, stage = "EXCLUDE", s1_reason, "stage1"
                counts["s1_exclude"] += 1
            elif s1 == "INCLUDE":
                decision, reason, stage = "INCLUDE", s1_reason, "stage1"
                counts["s1_include"] += 1
            else:
                # ── Stage 2: LLM ──────────────────────────────────────────────
                excerpt = get_text_excerpt(filename)
                try:
                    decision, reason = stage2_llm(title, excerpt, client)
                    stage = "stage2"
                except Exception as exc:
                    log.error(f"LLM failed for '{title}': {exc}")
                    decision, reason, stage = "INCLUDE", "LLM error — included by default", "stage2_error"
                if decision == "INCLUDE":
                    counts["s2_include"] += 1
                else:
                    counts["s2_exclude"] += 1
                log.info(f"  [{i:3d}/{len(speeches)}] stage2 {decision}: {title[:58]}")

        results.append({
            **row.to_dict(),
            "fomc_day":         fomc_day,
            "filter_stage":     stage,
            "filter_decision":  decision,
            "filter_reason":    reason,
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    df      = pd.DataFrame(results)
    kept    = df[df["filter_decision"] == "INCLUDE"]
    excluded = df[df["filter_decision"] == "EXCLUDE"]

    out_all  = PROCESSED_DIR / "speeches_filtered.csv"
    out_kept = PROCESSED_DIR / "speeches_monetary_policy.csv"
    df.to_csv(out_all,  index=False)
    kept.to_csv(out_kept, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*62}")
    log.info(f"Phase 1 complete — {len(df)} speeches processed")
    log.info(f"  Kept:     {len(kept):3d}  ({len(kept)/len(df)*100:.0f}%)")
    log.info(f"  Excluded: {len(excluded):3d}  ({len(excluded)/len(df)*100:.0f}%)")
    log.info(f"    Stage 1 keyword → include: {counts['s1_include']}  exclude: {counts['s1_exclude']}")
    log.info(f"    Stage 2 LLM     → include: {counts['s2_include']}  exclude: {counts['s2_exclude']}")
    log.info(f"    FOMC-day check  → exclude: {counts['fomc_exclude']}")
    log.info(f"  FOMC-day speeches (flagged): {df['fomc_day'].sum()}")

    log.info(f"\nExcluded speeches ({len(excluded)} total):")
    for _, r in excluded.sort_values("date").iterrows():
        log.info(f"  {str(r['date'].date()):10s}  [{r['filter_stage']:10s}]  {r['title'][:55]}")

    # ── Example of an excluded speech ────────────────────────────────────────
    if len(excluded) > 0:
        # Pick a Stage 2 exclusion if available (more interesting than a Stage 1 keyword hit)
        s2_excl = excluded[excluded["filter_stage"] == "stage2"]
        example = s2_excl.iloc[0] if len(s2_excl) > 0 else excluded.sort_values("date").iloc[0]
        excerpt = get_text_excerpt(example["filename"])

        log.info(f"\n{'='*62}")
        log.info("EXAMPLE EXCLUDED SPEECH")
        log.info(f"  Title:    {example['title']}")
        log.info(f"  Date:     {example['date'].date()}")
        log.info(f"  Chair:    {example['chair']}")
        log.info(f"  Stage:    {example['filter_stage']}")
        log.info(f"  Reason:   {example['filter_reason']}")
        log.info(f"  Words:    {example['word_count']}")
        log.info(f"  Text (first 300 chars):")
        log.info(f"    {excerpt[:300]}")

    log.info(f"\nOutputs:")
    log.info(f"  {out_all}   (all decisions)")
    log.info(f"  {out_kept}  (kept only)")


if __name__ == "__main__":
    main()
