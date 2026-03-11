import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from openai import OpenAI


LEET_MAP = str.maketrans({
    "@": "а",
    "4": "ч",
    "3": "з",
    "0": "о",
    "1": "і",
    "$": "с",
    "6": "б",
    "8": "в",
    "x": "х",
})

PARENT_INSULT_PATTERNS = [
    r"\bмамк[ауеийоы]\b",
    r"\bтво(ю|я)\s+мать\b",
    r"\bебал\s+(твою\s+)?мать\b",
    r"\bмать\s+твою\b",
    r"\bсын\s+(шлюхи|проститутки)\b",
    r"\bтво(его|их|ю)\s+(батю|отца|мать|маму|родаков)\b",
]


@dataclass
class Verdict:
    flagged: bool
    category: str
    score: float
    source: str
    details: Dict[str, Any]


class HybridModerator:
    def __init__(self, model_path: str = "models/text_moderation_pipeline.joblib"):
        self.model_path = Path(model_path)
        self.local_model = joblib.load(self.model_path) if self.model_path.exists() else None
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().translate(LEET_MAP)
        text = re.sub(r"https?://\S+", " link ", text)
        text = re.sub(r"[_\-\.|,;:!?/\\+*`~\[\](){}<>'\"]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?<=\b[а-яіїєґa-z])\s+(?=[а-яіїєґa-z]\b)", "", text)
        return text.strip()

    def regex_parent_insult(self, text: str) -> Optional[Verdict]:
        normalized = self.normalize_text(text)
        for pattern in PARENT_INSULT_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return Verdict(True, "parent_insult", 0.99, "regex", {"pattern": pattern})
        return None

    def classify_text_local(self, text: str) -> Optional[Verdict]:
        if not self.local_model:
            return None
        normalized = self.normalize_text(text)
        probs = self.local_model.predict_proba([normalized])[0]
        labels = list(self.local_model.classes_)
        top_idx = int(probs.argmax())
        top_label = labels[top_idx]
        top_score = float(probs[top_idx])
        flagged = top_label != "clean" and top_score >= 0.70
        return Verdict(
            flagged=flagged,
            category=top_label,
            score=top_score,
            source="local_model",
            details={"probs": {label: float(prob) for label, prob in zip(labels, probs)}},
        )

    def moderate_text_openai(self, text: str) -> Optional[Verdict]:
        if not self.openai_client:
            return None
        response = self.openai_client.moderations.create(
            model="omni-moderation-latest",
            input=text[:4000],
        )
        result = response.results[0]
        scores = result.category_scores
        mapping = [
            "harassment",
            "harassment_threatening",
            "hate",
            "hate_threatening",
            "sexual",
            "sexual_minors",
            "violence",
            "violence_graphic",
            "illicit",
            "illicit/violent",
        ]
        best_category, best_score = "none", 0.0
        for name in mapping:
            attr = name.replace("/", "_").replace("-", "_")
            value = getattr(scores, name, None)
            if value is None:
                value = getattr(scores, attr, None)
            if isinstance(value, (int, float)) and value > best_score:
                best_score = float(value)
                best_category = name
        return Verdict(
            flagged=bool(result.flagged) and best_score >= 0.60,
            category=best_category,
            score=best_score,
            source="openai_text",
            details={
                "categories": result.categories.model_dump() if hasattr(result.categories, "model_dump") else {},
                "scores": result.category_scores.model_dump() if hasattr(result.category_scores, "model_dump") else {},
            },
        )

    def moderate_text(self, text: str) -> Verdict:
        hard = self.regex_parent_insult(text)
        if hard:
            return hard

        local = self.classify_text_local(text)
        if local and local.flagged and local.category in {"parent_insult", "threat", "spam"}:
            return local

        remote = self.moderate_text_openai(text)
        if remote and remote.flagged:
            return remote

        if local:
            return local

        return Verdict(False, "clean", 0.0, "fallback", {})

    @staticmethod
    def _file_to_data_url(path: str) -> str:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def moderate_image(self, path: str, caption: str = "") -> Verdict:
        if not self.openai_client:
            return Verdict(False, "unavailable", 0.0, "image_disabled", {})
        payload = [{"type": "image_url", "image_url": {"url": self._file_to_data_url(path)}}]
        if caption:
            payload.insert(0, {"type": "text", "text": caption[:4000]})
        response = self.openai_client.moderations.create(
            model="omni-moderation-latest",
            input=payload,
        )
        result = response.results[0]
        scores = result.category_scores
        best_category, best_score = "none", 0.0
        for name in ["sexual", "sexual_minors", "violence", "violence_graphic"]:
            attr = name.replace("/", "_").replace("-", "_")
            value = getattr(scores, name, None)
            if value is None:
                value = getattr(scores, attr, None)
            if isinstance(value, (int, float)) and value > best_score:
                best_score = float(value)
                best_category = name
        return Verdict(
            flagged=bool(result.flagged) and best_score >= 0.70,
            category=best_category,
            score=best_score,
            source="openai_image",
            details={
                "categories": result.categories.model_dump() if hasattr(result.categories, "model_dump") else {},
                "scores": result.category_scores.model_dump() if hasattr(result.category_scores, "model_dump") else {},
            },
        )
