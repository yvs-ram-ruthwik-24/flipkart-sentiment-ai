from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud

from sentiment_utils import (
    LABEL_ORDER,
    VISUAL_STOPWORDS,
    load_flipkart_data,
    predict_review,
    sentiment_rating_table,
    top_words,
    train_sentiment_model,
)

sns.set_theme(style="whitegrid", context="talk")


df = load_flipkart_data("flipkart.csv")
print(f"Loaded {len(df):,} reviews from flipkart.csv")
print(f"Available columns: {', '.join(df.columns)}\n")

print("Sample rows:")
print(df[["Product_name", "Rating", "Sentiment"]].head(5).to_string(index=False))

print("\nSentiment breakdown:")
print(df["Sentiment"].value_counts().reindex(LABEL_ORDER).to_string())

print("\nRating breakdown:")
print(df["Rating"].value_counts().sort_index().to_string())

print("\nRating vs sentiment table:")
print(sentiment_rating_table(df).to_string())


sentiment_counts = df["Sentiment"].value_counts().reindex(LABEL_ORDER)
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar", color=["#dc2626", "#f59e0b", "#16a34a"])
plt.title("Distribution of Customer Sentiments on Flipkart", fontsize=16)
plt.xlabel("Sentiment Category", fontsize=12)
plt.ylabel("Total Number of Reviews", fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


rating_counts = df["Rating"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
rating_counts.plot(kind="bar", color="#4f46e5")
plt.title("Rating Distribution", fontsize=16)
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Total Number of Reviews", fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


rating_sentiment = sentiment_rating_table(df)
plt.figure(figsize=(8, 5))
sns.heatmap(
    rating_sentiment,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    cbar=False,
)
plt.title("Ratings and Sentiment Alignment", fontsize=16)
plt.xlabel("Sentiment")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()


print("\nGenerating top words chart...")
word_counts = top_words(df, top_n=20)
plt.figure(figsize=(10, 7))
word_counts.sort_values().plot(kind="barh", color="#38bdf8")
plt.title("Top 20 Meaningful Words in Flipkart Reviews", fontsize=16)
plt.xlabel("Number of Occurrences", fontsize=12)
plt.ylabel("Words", fontsize=12)
plt.tight_layout()
plt.show()


def build_wordcloud(text: str, colormap: str) -> WordCloud:
    return WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap=colormap,
        stopwords=VISUAL_STOPWORDS,
        collocations=False,
    ).generate(text)


positive_text = " ".join(df[df["Sentiment"] == "Positive"]["Cleaned_Review"])
negative_text = " ".join(df[df["Sentiment"] == "Negative"]["Cleaned_Review"])

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(build_wordcloud(positive_text, "Greens"), interpolation="bilinear")
plt.title("Word Cloud: Positive Reviews", fontsize=16)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(build_wordcloud(negative_text, "Reds"), interpolation="bilinear")
plt.title("Word Cloud: Negative Reviews", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()


artifacts = train_sentiment_model(df)
print("\nModel trained with a stratified split and then refit on all reviews for live use.")
print(f"Holdout accuracy: {artifacts.accuracy * 100:.2f}%")
print(f"Macro F1: {artifacts.macro_f1 * 100:.2f}%")
print(f"Neutral recall: {artifacts.neutral_recall * 100:.2f}%\n")

print("Classification report:")
print(classification_report(artifacts.y_test, artifacts.y_pred, zero_division=0))


print("Evaluation heatmaps...")
plt.figure(figsize=(7, 5))
sns.heatmap(
    artifacts.confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=artifacts.class_labels,
    yticklabels=artifacts.class_labels,
)
plt.title("Model Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Sentiment", fontsize=12)
plt.ylabel("Actual Sentiment", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
sns.heatmap(
    artifacts.report_df,
    annot=True,
    cmap="viridis",
    vmin=0,
    vmax=1,
)
plt.title("Classification Report Metrics", fontsize=16)
plt.tight_layout()
plt.show()


y_test_bin = label_binarize(artifacts.y_test, classes=artifacts.class_labels)
y_score = artifacts.y_score

fpr = {}
tpr = {}
roc_auc = {}

for index, label in enumerate(artifacts.class_labels):
    fpr[index], tpr[index], _ = roc_curve(y_test_bin[:, index], y_score[:, index])
    roc_auc[index] = auc(fpr[index], tpr[index])

plt.figure(figsize=(9, 6))
colors = cycle(["red", "gray", "green"])
for index, color in zip(range(len(artifacts.class_labels)), colors):
    plt.plot(
        fpr[index],
        tpr[index],
        color=color,
        lw=2,
        label=f"ROC curve: {artifacts.class_labels[index]} (AUC = {roc_auc[index]:0.2f})",
    )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


print("Live demonstration:")
demo_reviews = [
    "This laptop is amazing, very fast and smooth!",
    "Very bad experience, display quality is poor.",
    "Worst product please dont buy, start hanging within a week.",
    "Camera quality is not too good.",
]

for review in demo_reviews:
    result = predict_review(review, artifacts.pipeline)
    probabilities = result["probabilities"]
    print(f"Review: '{review}'")
    print(f"Predicted Sentiment: ---> **{result['prediction']}** <---")
    print(
        "Confidence: "
        f"Positive {probabilities.get('Positive', 0) * 100:.1f}%, "
        f"Neutral {probabilities.get('Neutral', 0) * 100:.1f}%, "
        f"Negative {probabilities.get('Negative', 0) * 100:.1f}%"
    )
    print("-" * 60)
