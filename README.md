# Flipkart Sentiment AI

Flipkart Sentiment AI is a Streamlit dashboard that classifies customer reviews as **Positive**, **Neutral**, or **Negative** and shows how sentiment relates to product ratings.

**Live app:** [Open the deployed app](https://flipkart-sentiment-ai-gkngcehhabybasp7jmfdza.streamlit.app/)

## What it does

- Predicts sentiment from a review text in real time
- Shows confidence scores for each sentiment class
- Compares rating patterns against sentiment labels
- Reuses the same preprocessing and model pipeline in both the app and analysis script
- Includes sample reviews for quick classroom or demo use

## Project files

- `app.py` — Streamlit dashboard
- `Ds_Capstone_p.py` — analysis and evaluation script
- `sentiment_utils.py` — shared preprocessing and model code
- `flipkart.csv` — dataset used for training and analysis
- `requirements.txt` — Python dependencies for local use and Streamlit Cloud

## Tech stack

- Python
- Streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
- wordcloud

## How the model works

1. Review text is cleaned and normalized.
2. TF-IDF converts the text into numeric features.
3. A linear classifier learns sentiment patterns from the review data.
4. The app displays the predicted label and confidence breakdown.

## Dataset

The dataset in `flipkart.csv` contains Flipkart product reviews with product names and ratings. The analysis script maps the reviews into sentiment labels and uses them to train and evaluate the model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```bash
http://localhost:8501
```

## Streamlit Cloud deployment

This repository is already configured for Streamlit Community Cloud.

- Repository: `yvs-ram-ruthwik-24/flipkart-sentiment-ai`
- Main file path: `app.py`

## Notes

- The model performs best on clear opinions.
- Mixed, sarcastic, or highly ambiguous reviews can still be difficult.
- The app is designed as a capstone project demo and classroom presentation tool.
