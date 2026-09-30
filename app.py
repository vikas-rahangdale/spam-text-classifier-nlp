# app.py
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import streamlit as st
import pickle
import re, nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="MessageGuard | Spam Classifier", page_icon="✉️", layout="centered")

# --- preprocessing function ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --- pretrained model functions ---
def load_pretrained():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_pretrained(classifier, text):
    return classifier(text, return_all_scores=True)

# --- Streamlit UI ---
st.markdown("""
<style>
        .stApp { background: #f5f7fb; }
        .block-container { max-width: 900px; padding-top: 2.5rem; padding-bottom: 3rem; }
        .hero { background: linear-gradient(120deg, #102a43, #176b87); color: white;
                        padding: 2rem 2.2rem; border-radius: 16px; margin-bottom: 1.4rem; }
        .hero h1 { margin: 0; font-size: 2rem; }
        .hero p { color: #d9e9f2; margin: .55rem 0 0; }
        div[data-testid="stTextArea"] textarea { border-radius: 10px; }
        div[data-testid="stJson"] { background: white; border: 1px solid #e2e8f0;
                                                                    border-radius: 10px; padding: .5rem; }
</style>
<div class="hero">
    <h1>✉️ MessageGuard</h1>
    <p>Spam classification for text messages, powered by  trained model and a transformer benchmark.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Enter a message below to review predictions from both classification approaches.")

# Load best trained ML model
try:
    best_model = pickle.load(open("models/best_model.pkl", "rb"))
except:
    best_model = None

# Load vectorizer
try:
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
except:
    vectorizer = None

# Load pretrained transformer
try:
    classifier = load_pretrained()
except Exception as exc:
    classifier = None
    st.warning(f"The pretrained transformer could not be loaded: {exc}")

# Text input
user_text = st.text_area(
    "Message text",
    placeholder="Paste or type a message here…",
    height=150,
    help="Your message is used only to generate the predictions shown below.",
)

if user_text:
    st.divider()
    st.subheader("Classification results")
    left, right = st.columns(2, gap="medium")
    if vectorizer is not None and best_model is not None:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        pred = best_model.predict(vec)[0]
        label = "Spam" if pred == 1 else "Ham"
        ml_result = {"model": "Best Trained ML Model", "label": label}
        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(vec)[0]
            class_index = list(best_model.classes_).index(pred)
            ml_result["confidence"] = float(probabilities[class_index])
        with left:
            st.markdown("#### Trained ML model")
            st.json(ml_result)
    else:
        with left:
            st.markdown("#### Trained ML model")
            st.info("Model files are unavailable. Add the trained model and vectorizer to the `models` directory.")

    with right:
        st.markdown("#### Transformer benchmark")
        if classifier is None:
            st.info("Transformer predictions are unavailable right now.")
        else:
            try:
                prediction = predict_pretrained(classifier, user_text)
                if prediction and isinstance(prediction[0], dict):
                    scores = prediction
                elif prediction:
                    scores = prediction[0]
                else:
                    scores = []
                score_by_label = {item["label"].upper(): item["score"] for item in scores}
                positive_score = score_by_label.get("POSITIVE", 0.0)
                negative_score = score_by_label.get("NEGATIVE", 0.0)
                transformer_label = "Spam" if negative_score > positive_score else "Ham"
                st.json({
                    "model": "Pretrained Transformer",
                    "label": transformer_label,
                    "confidence": float(max(positive_score, negative_score)),
                })
            except Exception as exc:
                st.error(f"Transformer prediction failed: {exc}")
else:
    st.info("Predictions will appear here after you enter a message.")
