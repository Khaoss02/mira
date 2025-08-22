from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TextBlob (subjectivity)
# TextBlob sometimes needs corpora; we’ll fail gracefully if not available
try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except Exception:
    _HAS_TEXTBLOB = False


@dataclass
class SentimentResult:
    compound: float
    pos: float
    neu: float
    neg: float
    subjectivity: Optional[float] = None
    dominant_emotion: Optional[str] = None


class SentimentEngine:
    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        vs = self.vader.polarity_scores(text)
        subjectivity = None
        if _HAS_TEXTBLOB:
            try:
                subjectivity = float(TextBlob(text).sentiment.subjectivity)
            except Exception:
                subjectivity = None

        dominant_emotion = self._coarse_emotion(vs["compound"])
        return SentimentResult(
            compound=vs["compound"],
            pos=vs["pos"],
            neu=vs["neu"],
            neg=vs["neg"],
            subjectivity=subjectivity,
            dominant_emotion=dominant_emotion,
        )

    @staticmethod
    def _coarse_emotion(compound: float) -> str:
        if compound >= 0.6:
            return "joy"
        if compound >= 0.2:
            return "positive"
        if compound <= -0.6:
            return "anger/sadness"
        if compound <= -0.2:
            return "negative"
        return "neutral"

    def as_dict(self, result: SentimentResult) -> Dict[str, Any]:
        return {
            "compound": result.compound,
            "pos": result.pos,
            "neu": result.neu,
            "neg": result.neg,
            "subjectivity": result.subjectivity,
            "dominant_emotion": result.dominant_emotion,
        }
