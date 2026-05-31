from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from wordcloud import STOPWORDS

LABEL_ORDER = ["Negative", "Neutral", "Positive"]
RANDOM_STATE = 42
TFIDF_CONFIG = {
    "max_features": 15000,
    "ngram_range": (1, 3),
    "min_df": 3,
    "sublinear_tf": False,
}
CLASSIFIER_CONFIG = {
    "loss": "log_loss",
    "class_weight": "balanced",
    "max_iter": 3000,
    "tol": 1e-3,
    "random_state": RANDOM_STATE,
}
VISUAL_STOPWORDS = STOPWORDS.union(
    {
        "product",
        "products",
        "review",
        "reviews",
        "flipkart",
        "good",
        "great",
        "very",
        "really",
        "much",
        "one",
        "use",
        "using",
        "used",
        "buy",
        "bought",
        "would",
        "could",
        "also",
        "please",
        "dont",
        "doesnt",
        "im",
        "ive",
    }
)


@dataclass
class SentimentArtifacts:
    pipeline: Pipeline
    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_score: np.ndarray
    accuracy: float
    macro_f1: float
    weighted_f1: float
    neutral_recall: float
    report_df: pd.DataFrame
    confusion_matrix: np.ndarray
    feature_count: int
    class_labels: list[str]


def map_sentiment(rating: float) -> str:
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_flipkart_data(csv_path: str = "flipkart.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_columns = {"Product_name", "Review", "Rating"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path}: {missing_text}")

    df = df.copy()
    df["Sentiment"] = df["Rating"].apply(map_sentiment)
    df["Cleaned_Review"] = df["Review"].apply(clean_text)
    df["Review_Word_Count"] = df["Cleaned_Review"].str.split().str.len()
    df["Review_Char_Count"] = df["Review"].astype(str).str.len()
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("classifier", SGDClassifier(**CLASSIFIER_CONFIG)),
        ]
    )


def train_sentiment_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> SentimentArtifacts:
    X = df["Cleaned_Review"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(X_train, y_train)
    y_pred = evaluation_pipeline.predict(X_test)
    y_score = evaluation_pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.loc[LABEL_ORDER, ["precision", "recall", "f1-score"]]
    confusion = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    feature_count = len(final_pipeline.named_steps["tfidf"].get_feature_names_out())
    class_labels = list(final_pipeline.named_steps["classifier"].classes_)

    return SentimentArtifacts(
        pipeline=final_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_score=y_score,
        accuracy=accuracy,
        macro_f1=report["macro avg"]["f1-score"],
        weighted_f1=report["weighted avg"]["f1-score"],
        neutral_recall=report["Neutral"]["recall"],
        report_df=report_df,
        confusion_matrix=confusion,
        feature_count=feature_count,
        class_labels=class_labels,
    )


def predict_review(review: str, pipeline: Pipeline) -> dict[str, object]:
    cleaned = clean_text(review)
    probabilities = pipeline.predict_proba([cleaned])[0]
    labels = list(pipeline.named_steps["classifier"].classes_)
    prediction = pipeline.predict([cleaned])[0]
    probability_map = {
        label: float(probability)
        for label, probability in zip(labels, probabilities)
    }
    return {
        "prediction": prediction,
        "cleaned_review": cleaned,
        "probabilities": probability_map,
    }


def top_words(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    tokens: list[str] = []
    for review in df["Cleaned_Review"]:
        tokens.extend(
            token
            for token in review.split()
            if len(token) > 2 and token not in VISUAL_STOPWORDS
        )
    return pd.Series(Counter(tokens)).sort_values(ascending=False).head(top_n)


def sentiment_rating_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["Rating"], df["Sentiment"]).reindex(
        index=[1, 2, 3, 4, 5],
        columns=LABEL_ORDER,
        fill_value=0,
    )
