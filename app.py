from __future__ import annotations

import pandas as pd
import streamlit as st

from sentiment_utils import (
    LABEL_ORDER,
    load_flipkart_data,
    predict_review,
    sentiment_rating_table,
    train_sentiment_model,
)

st.set_page_config(
    page_title="Flipkart Sentiment AI",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #06b6d4 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 2.15rem;
    }
    .hero-card p {
        margin: 0.35rem 0 0;
        color: rgba(255, 255, 255, 0.92);
        font-size: 0.98rem;
    }
    .soft-card {
        padding: 1rem 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }
    .small-copy {
        color: #475569;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    dataframe = load_flipkart_data("flipkart.csv")
    artifacts = train_sentiment_model(dataframe)
    return dataframe, artifacts


df, artifacts = load_assets()
pipeline = artifacts.pipeline

st.markdown(
    """
    <div class="hero-card">
        <div class="small-copy" style="color: rgba(255,255,255,0.78);">Data Science Capstone Project 6</div>
        <h1>Flipkart Sentiment AI</h1>
        <p>Analyze customer reviews, compare sentiment with ratings, and inspect confidence scores with a polished live dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
with top_col1:
    st.metric("Reviews", f"{len(df):,}")
with top_col2:
    st.metric("Products", f"{df['Product_name'].nunique():,}")
with top_col3:
    st.metric("Accuracy", f"{artifacts.accuracy * 100:.2f}%")
with top_col4:
    st.metric("Macro F1", f"{artifacts.macro_f1 * 100:.2f}%")

st.caption(
    "The model is trained on flipkart.csv, cached for fast reruns, and uses the same preprocessing flow as the analysis script."
)

with st.sidebar:
    st.header("Dashboard Snapshot")
    st.metric("Positive", int((df["Sentiment"] == "Positive").sum()))
    st.metric("Neutral", int((df["Sentiment"] == "Neutral").sum()))
    st.metric("Negative", int((df["Sentiment"] == "Negative").sum()))
    st.divider()
    st.subheader("Model details")
    st.write("TF-IDF vectorizer with 1-3 word n-grams")
    st.write("SGD classifier using log-loss")
    st.write("Stratified holdout validation for reporting")
    st.divider()
    st.subheader("Quick note")
    st.info("Try the sample buttons first if you want a fast demo. The model handles obvious positive and negative reviews well, but mixed or sarcastic text can still be tricky.")

tab1, tab2 = st.tabs(["🔮 Live Predictor", "🧠 Model Overview"])

if "review_input" not in st.session_state:
    st.session_state.review_input = "This laptop is amazing, very fast and smooth!"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def clear_analysis() -> None:
    st.session_state.analysis_result = None


sample_reviews = {
    "Happy": "This laptop is amazing, very fast and smooth!",
    "Negative": "Very bad experience, display quality is poor.",
    "Mixed": "Worst product please dont buy, start hanging within a week.",
    "Neutral": "Camera quality is not too good.",
}


with tab1:
    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.subheader("Test the AI")
        st.write("Load a sample or type your own review below.")

        sample_button_cols = st.columns(4)
        for index, (label, sample_text) in enumerate(sample_reviews.items()):
            with sample_button_cols[index]:
                if st.button(label, use_container_width=True):
                    st.session_state.review_input = sample_text
                    st.session_state.analysis_result = None
                    st.rerun()

        user_review = st.text_area(
            "Enter a customer review here:",
            key="review_input",
            height=180,
            on_change=clear_analysis,
        )

        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            if st.button("🤖 Analyze Sentiment", type="primary", use_container_width=True):
                if user_review.strip():
                    st.session_state.analysis_result = predict_review(user_review, pipeline)
                else:
                    st.error("Please enter a review first.")
        with action_col2:
            if st.button("Reset", use_container_width=True):
                st.session_state.review_input = ""
                st.session_state.analysis_result = None
                st.rerun()

        st.caption("The buttons above are just shortcuts. You can type any new review and rerun the analysis instantly.")

    with right_col:
        st.subheader("Prediction Result")
        result = st.session_state.analysis_result
        if result is None:
            st.markdown(
                """
                <div class="soft-card">
                    <strong>No prediction yet.</strong><br/>
                    Type a review on the left and press <em>Analyze Sentiment</em> to see the result and confidence breakdown.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            prediction = result["prediction"]
            probabilities = result["probabilities"]
            cleaned_review = result["cleaned_review"]

            if prediction == "Positive":
                st.success(f"🟢 **{prediction}** — the customer is happy.")
            elif prediction == "Negative":
                st.error(f"🔴 **{prediction}** — the customer is unhappy.")
            else:
                st.warning(f"🟡 **{prediction}** — the model sees a mixed or neutral tone.")

            probability_frame = pd.DataFrame(
                {
                    "Sentiment": LABEL_ORDER,
                    "Probability": [probabilities.get(label, 0.0) for label in LABEL_ORDER],
                }
            ).set_index("Sentiment")
            st.bar_chart(probability_frame, use_container_width=True)

            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Positive", f"{probabilities.get('Positive', 0.0) * 100:.1f}%")
            with metric_cols[1]:
                st.metric("Neutral", f"{probabilities.get('Neutral', 0.0) * 100:.1f}%")
            with metric_cols[2]:
                st.metric("Negative", f"{probabilities.get('Negative', 0.0) * 100:.1f}%")

            with st.expander("View the cleaned text"):
                st.write("Raw review")
                st.text(user_review)
                st.write("Cleaned review")
                st.text(cleaned_review if cleaned_review else "(empty after cleaning)")

    st.divider()
    st.subheader("Model confidence tips")
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    with tip_col1:
        st.info("Positive reviews usually contain strong praise words such as *amazing*, *fast*, or *smooth*.")
    with tip_col2:
        st.info("Negative reviews are usually easier to catch when they contain clear complaint words such as *poor*, *bad*, or *worst*.")
    with tip_col3:
        st.info("Neutral and mixed reviews are the hardest because TF-IDF still depends on word frequency, not full context.")


with tab2:
    overview_col, metrics_col = st.columns([1, 1])

    with overview_col:
        st.subheader("How the model works")
        st.markdown(
            """
            1. The review text is cleaned and standardized.
            2. TF-IDF converts words and short phrases into numeric features.
            3. The classifier learns from those features to predict Positive, Neutral, or Negative sentiment.
            4. A stratified holdout split checks that the model performs well on unseen reviews.
            """
        )

        st.subheader("What makes this version better")
        st.markdown(
            """
            - The training split is stratified, so the evaluation is more representative.
            - The same preprocessing is used in the analysis script and the app.
            - The live model is trained with stronger text features and balanced class handling.
            """
        )

    with metrics_col:
        st.subheader("Validation metrics")
        metric_row_1 = st.columns(3)
        with metric_row_1[0]:
            st.metric("Accuracy", f"{artifacts.accuracy * 100:.2f}%")
        with metric_row_1[1]:
            st.metric("Macro F1", f"{artifacts.macro_f1 * 100:.2f}%")
        with metric_row_1[2]:
            st.metric("Neutral Recall", f"{artifacts.neutral_recall * 100:.2f}%")

        metric_row_2 = st.columns(3)
        with metric_row_2[0]:
            st.metric("Weighted F1", f"{artifacts.weighted_f1 * 100:.2f}%")
        with metric_row_2[1]:
            st.metric("Vocabulary", f"{artifacts.feature_count:,}")
        with metric_row_2[2]:
            st.metric("Labels", len(artifacts.class_labels))

        st.dataframe(artifacts.report_df, use_container_width=True)

    st.divider()

    summary_col1, summary_col2 = st.columns([1, 1])
    with summary_col1:
        st.subheader("Rating ↔ Sentiment snapshot")
        st.dataframe(sentiment_rating_table(df), use_container_width=True)

    with summary_col2:
        st.subheader("Dataset preview")
        st.dataframe(
            df[["Product_name", "Rating", "Sentiment"]].head(8),
            use_container_width=True,
        )

    st.warning(
        "This is a strong classic machine-learning baseline, but sarcasm, subtle negation, and highly mixed opinions can still be difficult."
    )
