import pickle
import string
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from email import policy
from email.parser import BytesParser

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SpamShield | Email Checker",
    page_icon="🛡️",
    layout="centered",
)

# -----------------------------
# Download NLTK resources
# -----------------------------
@st.cache_resource
def setup_nltk():
    for resource in ("stopwords", "punkt_tab", "punkt"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    return (
        PorterStemmer(),
        set(stopwords.words("english")),
    )


ps, stop_words = setup_nltk()


# -----------------------------
# Load trained model files
# -----------------------------
@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent

    with open(base_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(base_dir / "spam_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


# -----------------------------
# Text preprocessing
# Must match training
# -----------------------------
def clean_new_text(text):
    text = text.lower()
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    tokens = nltk.word_tokenize(text)

    tokens = [
        ps.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(tokens)


# -----------------------------
# Extract uploaded email text
# -----------------------------
def read_uploaded_email(uploaded_file):
    raw = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".eml"):
        message = BytesParser(policy=policy.default).parsebytes(raw)

        parts = []
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        parts.append(content)
                except Exception:
                    continue

        return "\n\n".join(parts)

    return raw.decode("utf-8", errors="replace")


# -----------------------------
# Predict email class
# -----------------------------
def classify_email(text, vectorizer, model):
    cleaned = clean_new_text(text)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]

    # Support models whose labels are 0 = Ham, 1 = Spam
    if prediction == 1:
        label = "Spam"
    elif prediction == 0:
        label = "Ham"
    else:
        label = str(prediction)

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if prediction in classes:
            confidence = float(
                probabilities[classes.index(prediction)]
            )

    return label, confidence


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ SpamShield")
st.caption("AI-powered email spam detection")
st.write(
    "Paste an email or upload an email file to check "
    "whether it is spam or legitimate."
)

try:
    vectorizer, model = load_artifacts()
except Exception as exc:
    st.error(
        "Could not load the model files. Make sure both "
        "pickle files are in the same folder as app.py."
    )
    st.exception(exc)
    st.stop()

st.subheader("Check an email")

uploaded_file = st.file_uploader(
    "Upload an email file (optional)",
    type=["txt", "eml"],
    help="Upload a plain-text email or an .eml email file.",
)

default_text = ""

if uploaded_file is not None:
    try:
        default_text = read_uploaded_email(uploaded_file)
        if not default_text.strip():
            st.warning(
                "No plain-text email body was found in this file."
            )
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")

email_text = st.text_area(
    "Email content",
    value=default_text,
    height=240,
    placeholder=(
        "Paste the email subject and body here...\n\n"
        "Example: Congratulations! You have won a free prize."
    ),
)

check_clicked = st.button(
    "🔍 Check for Spam",
    type="primary",
    use_container_width=True,
)

if check_clicked:
    if not email_text.strip():
        st.warning("Please enter an email or upload a file first.")
    else:
        try:
            with st.spinner("Analyzing email..."):
                label, confidence = classify_email(
                    email_text, vectorizer, model
                )

            st.divider()
            st.subheader("Analysis result")

            if label == "Spam":
                st.error("🚨 Spam detected")
                st.write(
                    "This email was classified as spam by your model."
                )
            elif label == "Ham":
                st.success("✅ Ham — Not Spam")
                st.write(
                    "This email was classified as legitimate "
                    "by your model."
                )
            else:
                st.info(f"Prediction: {label}")

            if confidence is not None:
                st.metric(
                    "Prediction confidence",
                    f"{confidence:.2%}",
                )
                st.progress(confidence)

            st.caption(
                "Predictions are model estimates, not a guarantee "
                "that an email is safe."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with st.expander("About this app"):
    st.write(
        "SpamShield uses a saved TF-IDF vectorizer and your "
        "trained machine-learning classifier. Email text is "
        "preprocessed using lowercase conversion, punctuation "
        "removal, stopword filtering, and Porter stemming."
    )