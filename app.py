import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
import io
from wordcloud import WordCloud
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------------------
# Config & Helpers
# ---------------------------
st.set_page_config(page_title="News Category Prediction", page_icon="📰", layout="wide")
DATASET_PATH = "dataset/BBC_News_Train.csv"
DEFAULT_MODEL_PATH = "news_category_prediction.pkl"
FEEDBACK_CSV = "feedback.csv"

CATEGORY_ICONS = {
    "business": "💼",
    "entertainment": "🎬",
    "politics": "🏛️",
    "sport": "🏆",
    "tech": "💻",
    # fallback
    "other": "📰"
}

BADGE_COLORS = {
    "business": "#1f77b4",
    "entertainment": "#ff7f0e",
    "politics": "#2ca02c",
    "sport": "#d62728",
    "tech": "#9467bd",
    "other": "#7f7f7f"
}

# Apply simple dark/light mode CSS
def apply_css(dark_mode: bool):
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: #d6d6d6; }
            .stButton>button { background-color: #2b6ea3; color: white; }
            .stDownloadButton>button { background-color: #2b6ea3; color: white; }
            .stTextInput>div>div>input, .stTextArea>div>div>textarea { background-color: #1a1f26; color: #d6d6d6; }
            .stRadio>div>label, .stCheckbox>div>label { color: #d6d6d6; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #ffffff; color: #0f1724; }
            .stButton>button { background-color: #ff4b4b; color: white; }
            .stDownloadButton>button { background-color: #ff4b4b; color: white; }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Utility: load dataset if present
@st.cache_data
def load_dataset(path=DATASET_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Ensure expected columns exist
        if "Text" in df.columns and "Category" in df.columns:
            return df
    return None

# Utility: save feedback row
def save_feedback(input_text, predicted, correct, correct_label=None):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_text": input_text,
        "predicted": predicted,
        "correct": correct,
        "correct_label": correct_label or ""
    }
    df = pd.DataFrame([row])
    if not os.path.exists(FEEDBACK_CSV):
        df.to_csv(FEEDBACK_CSV, index=False)
    else:
        df.to_csv(FEEDBACK_CSV, mode="a", header=False, index=False)

# Utility: build PDF report bytes
def create_pdf_report(input_text, predicted, prob_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("📰 News Category Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Input Text:</b> {input_text}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Predicted Category:</b> {predicted}", styles["Heading2"]))
    story.append(Spacer(1, 12))

    prob_data = [["Category", "Probability"]] + prob_df.values.tolist()
    table = Table(prob_data, colWidths=[250, 150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(Paragraph("<b>Category Probabilities:</b>", styles["Heading3"]))
    story.append(table)
    story.append(Spacer(1, 20))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------------
# Model Loading & Training Helpers
# ---------------------------
@st.cache_resource
def load_or_train_models():
    """
    Attempts to load pre-saved models. If not available, trains simple models from dataset.
    Returns a dict of name -> pipeline model
    """
    models = {}
    # Try load default pipeline
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            pipeline = joblib.load(DEFAULT_MODEL_PATH)
            models["Logistic Regression"] = pipeline
        except Exception as e:
            st.warning(f"Could not load default model: {e}")

    # If dataset exists, train NB and SVM (if not already persisted)
    df = load_dataset()
    if df is not None:
        X = df["Text"].fillna("").astype(str)
        y = df["Category"]
        # If NB missing, or we want fresh versions, train
        try:
            # MultinomialNB pipeline
            nb_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.8)),
                ("nb", MultinomialNB(alpha=1.0))
            ])
            nb_pipe.fit(X, y)
            models["Naive Bayes"] = nb_pipe
        except Exception as e:
            st.warning(f"Failed to train Naive Bayes: {e}")

        try:
            # SVC pipeline (with probability=True to get probabilities)
            svc_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.8)),
                ("svc", SVC(C=1.0, kernel="linear", probability=True))
            ])
            svc_pipe.fit(X, y)
            models["SVM"] = svc_pipe
        except Exception as e:
            st.warning(f"Failed to train SVM: {e}")
    else:
        st.info("Dataset not found locally - fallback only to pre-saved model (if available).")

    # If logistic wasn't loaded but dataset exists, train LR
    if "Logistic Regression" not in models and df is not None:
        try:
            lr_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.8)),
                ("lr", LogisticRegression(max_iter=1000))
            ])
            lr_pipe.fit(X, y)
            models["Logistic Regression"] = lr_pipe
        except Exception as e:
            st.warning(f"Failed to train Logistic Regression: {e}")

    return models

# ---------------------------
# UI Layout
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dataset Insights", "Model Insights", "Feedback"])

dark_mode = st.sidebar.checkbox("Dark mode", value=True)
apply_css(dark_mode)

st.title("📰 News Category Prediction")

# Load models (cached)
with st.spinner("Loading models..."):
    models = load_or_train_models()
model_names = list(models.keys())
if not model_names:
    st.error("No models available. Ensure `news_category_prediction.pkl` or dataset exists.")
    st.stop()

# ---------------------------
# Prediction Page
# ---------------------------
if page == "Prediction":
    st.header("🔎 Predict a News Article Category")
    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_area("Enter news text to classify:", height=220)
        if st.button("Clear"):
            input_text = ""
            st.experimental_rerun()
    with col2:
        st.markdown("**Model & Options**")
        chosen_model_name = st.selectbox("Choose model:", model_names, index=0)
        show_topk = st.slider("Show top K probabilities", 1, min(5, len(models[chosen_model_name].classes_)), 3)
        show_top_words = st.checkbox("Show top influential words for predicted class", value=True)

    if st.button("Classify"):

        if not input_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            model = models[chosen_model_name]
            try:
                pred = model.predict([input_text])[0]
                probs = model.predict_proba([input_text])[0]
                classes = model.classes_
            except Exception as e:
                st.error(f"Model failed to predict: {e}")
                st.stop()

            # Display predicted badge
            icon = CATEGORY_ICONS.get(pred.lower(), CATEGORY_ICONS["other"])
            color = BADGE_COLORS.get(pred.lower(), BADGE_COLORS["other"])

            st.markdown(f"### Predicted Category: <span style='color:{color}; font-size:26px'>{icon}  {pred}</span>", unsafe_allow_html=True)

            # Build probabilities DataFrame
            prob_df = pd.DataFrame({"Category": classes, "Probability": probs})
            prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
            st.write("#### Top probabilities")
            st.dataframe(prob_df.head(show_topk).style.format({"Probability": "{:.3f}"}))

            # Plotly bar chart for probabilities
            fig = px.bar(prob_df, x="Category", y="Probability", title="Category probabilities", text=prob_df["Probability"].round(3))
            st.plotly_chart(fig, use_container_width=True)

            # Top influential words if available (for linear models or vectorizer-based pipelines)
            if show_top_words:
                # Try to extract vectorizer and classifier
                try:
                    # If pipeline
                    if isinstance(model, Pipeline):
                        vect = None
                        clf = None
                        for name, step in model.steps:
                            if 'tfidf' in name:
                                vect = step
                            if name in ("lr", "svc", "nb", "classifier", "clf"):
                                clf = step
                        if vect is not None and clf is not None:
                            feature_names = vect.get_feature_names_out()
                            topk = 10
                            # Logistic Regression / Linear SVC -> coef_
                            if hasattr(clf, "coef_"):
                                class_index = list(model.classes_).index(pred)
                                coefs = clf.coef_[class_index]
                                top_idx = np.argsort(coefs)[-topk:][::-1]
                                top_features = [(feature_names[i], coefs[i]) for i in top_idx]
                                st.write(f"Top {topk} influential words for **{pred}** (approx.):")
                                st.write(pd.DataFrame(top_features, columns=["word", "score"]))
                            # MultinomialNB -> feature_log_prob_
                            elif hasattr(clf, "feature_log_prob_"):
                                class_index = list(model.classes_).index(pred)
                                probs_feat = clf.feature_log_prob_[class_index]
                                top_idx = np.argsort(probs_feat)[-10:][::-1]
                                top_features = [(feature_names[i], probs_feat[i]) for i in top_idx]
                                st.write(f"Top words for **{pred}** (NB approx.):")
                                st.write(pd.DataFrame(top_features, columns=["word", "log_prob"]))
                except Exception as e:
                    st.info("Top words not available for this model: " + str(e))

            # CSV / PDF downloads
            result_df = pd.DataFrame({"Input Text": [input_text], "Predicted Category": [pred]})
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download result as CSV", data=csv, file_name="prediction.csv", mime="text/csv")

            pdf_buffer = create_pdf_report(input_text, pred, prob_df)
            st.download_button("Download result as PDF", data=pdf_buffer, file_name="prediction_report.pdf", mime="application/pdf")

            # Provide feedback quick buttons
            st.write("---")
            st.write("Was the prediction correct?")
            fcol1, fcol2, fcol3 = st.columns([1,1,2])
            with fcol1:
                if st.button("Yes — correct"):
                    save_feedback(input_text, pred, True)
                    st.success("Thanks for the feedback! (Saved)")
            with fcol2:
                if st.button("No — incorrect"):
                    # ask for correct label
                    df_local = load_dataset()
                    if df_local is not None:
                        labels = sorted(df_local["Category"].unique())
                        correct_label = st.selectbox("Select correct label", labels, key="correct_label_select")
                    else:
                        correct_label = st.text_input("Enter correct label", key="correct_label_text")
                    if st.button("Submit correct label"):
                        save_feedback(input_text, pred, False, correct_label)
                        st.success("Thanks — feedback saved.")
            with fcol3:
                st.info("Your feedback helps improve models when retraining.")

# ---------------------------
# Dataset Insights
# ---------------------------
elif page == "Dataset Insights":
    st.header("📊 Dataset Insights")
    df = load_dataset()
    if df is None:
        st.error("Dataset not found at `dataset/BBC_News_Train.csv`. Place the CSV there to use insights features.")
    else:
        st.write("### Dataset preview")
        st.dataframe(df.head(10))
        st.write("### Category distribution")
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, values="Count", names="Category", title="Category distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Wordcloud examples")
        cat = st.selectbox("Choose category for wordcloud", sorted(df["Category"].unique()))
        text = " ".join(df[df["Category"] == cat]["Text"].astype(str).tolist())
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color="white", max_words=200).generate(text)
            plt.figure(figsize=(12, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("No text available for this category.")

# ---------------------------
# Model Insights
# ---------------------------
elif page == "Model Insights":
    st.header("🧠 Model Insights & Evaluation")
    df = load_dataset()
    if df is None:
        st.error("Dataset not found locally. Place `dataset/BBC_News_Train.csv` to compute evaluation metrics.")
    else:
        X = df["Text"].fillna("").astype(str)
        y = df["Category"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        chosen = st.selectbox("Choose model to evaluate", model_names)
        model = models[chosen]

        # If the model was trained on full dataset earlier, evaluate on test split
        with st.spinner("Computing predictions on test split..."):
            try:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")
                st.write("#### Classification report")
                cls_report = classification_report(y_test, y_pred, output_dict=True)
                cr_df = pd.DataFrame(cls_report).transpose()
                st.dataframe(cr_df)
                st.write("#### Confusion matrix")
                cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                fig_cm, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                st.pyplot(fig_cm)
            except Exception as e:
                st.error(f"Failed to evaluate model: {e}")

        # Show top features per class for linear models
        if st.checkbox("Show top features per class (if available)"):
            try:
                if isinstance(model, Pipeline):
                    vect = None
                    clf = None
                    for name, step in model.steps:
                        if 'tfidf' in name:
                            vect = step
                        if name in ("lr", "svc", "nb", "classifier", "clf"):
                            clf = step
                    if vect is not None and clf is not None:
                        feature_names = vect.get_feature_names_out()
                        topn = st.slider("Top N features", 3, 30, 10)
                        cols = st.columns(len(model.classes_))
                        for idx, cls in enumerate(model.classes_):
                            with cols[idx % len(cols)]:
                                st.write(f"**{cls}**")
                                if hasattr(clf, "coef_"):
                                    coefs = clf.coef_[idx]
                                    top_idx = np.argsort(coefs)[-topn:][::-1]
                                    top_features = [feature_names[i] for i in top_idx]
                                    st.write(", ".join(top_features))
                                elif hasattr(clf, "feature_log_prob_"):
                                    probs_feat = clf.feature_log_prob_[idx]
                                    top_idx = np.argsort(probs_feat)[-topn:][::-1]
                                    top_features = [feature_names[i] for i in top_idx]
                                    st.write(", ".join(top_features))
                else:
                    st.info("Top features require a pipeline with TF-IDF and a classifier supporting coef_ or feature_log_prob_.")
            except Exception as e:
                st.error("Could not compute top features: " + str(e))

# ---------------------------
# Feedback page
# ---------------------------
elif page == "Feedback":
    st.header("🗳️ Feedback & Retraining Data")
    st.write("This page shows stored user feedback which you can use to improve models later.")
    if os.path.exists(FEEDBACK_CSV):
        fdf = pd.read_csv(FEEDBACK_CSV)
        st.dataframe(fdf.tail(200))
        st.download_button("Download feedback CSV", data=fdf.to_csv(index=False).encode("utf-8"), file_name="feedback.csv", mime="text/csv")
    else:
        st.info("No feedback collected yet. Users can provide feedback from the Prediction page.")

