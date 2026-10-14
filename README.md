# SMS Spam Classification Project

This project establishes an end-to-end Machine Learning pipeline to accurately classify text messages as either **Ham** (legitimate/normal messages) or **Spam** (unsolicited/junk messages). 

---

## 1. Data Ingestion & Exploration (EDA)

* **The Dataset:** We loaded a classic SMS dataset containing **5,572 messages** (4,825 Ham and 747 Spam). This configuration highlights a stark class imbalance typical for real-world spam filtering systems.
* **Feature Engineering:** Simple text statistics were engineered to identify unique differences between classes:
    * **Length & Word Count:** On average, Spam messages are significantly longer and contain more words, frequently hovering around 150 characters to accommodate promotional details.
    * **Punctuation Density:** Spam messages typically contain a higher volume of punctuation marks, particularly exclamation points (`!`) and dollar signs (`$`).
* **Visualizations:** We analyzed length distributions via density plots, reviewed class distributions with count plots, and generated Word Clouds. The visual text analysis showed that words like *"FREE"*, *"call"*, *"txt"*, and *"claim"* heavily dominate spam, while phrases like *"ok"*, *"go"*, and *"love"* characterize ham messages.

---

## 2. Text Preprocessing

To transform raw text into structural formats compatible with machine learning algorithms, data was cleaned and normalized using a custom `clean_text` function:

1. **Lowercasing:** Standardised all text variations to lowercase strings.
2. **Punctuation Removal:** Stripped noisy punctuation markers (such as `!`, `?`, and `.`).
3. **Tokenization:** Segmented block sentences into individual words (tokens).
4. **Stopwords Removal:** Dropped generic English filler terms (e.g., *"the"*, *"is"*, *"at"*) that carry minimal class-specific context.
5. **Stemming (Porter Stemmer):** Reduced remaining words to their base linguistic forms (e.g., *"running"*, *"runs"*, and *"ran"* all resolve to *"run"*), drastically simplifying the corpus vocabulary.

---

## 3. Feature Vectorization (TF-IDF)

* **Data Splitting:** Preprocessed text messages were divided into an **80% training set** and a **20% testing set**. We applied stratification to perfectly mirror the master spam/ham distribution across both subsets.
* **Vectorization:** We engineered text features using **TF-IDF (Term Frequency-Inverse Document Frequency)** with a `max_features=5000` limit. 
* **Core Logic:** TF-IDF quantifies the relative importance of words in a document context against the background dataset. This mathematical framework down-weights universally common vocabulary while spotlighting highly unique keywords.

---

## 4. Model Training & Comparison

We implemented and bench-tested four distinct classifiers within our constructed TF-IDF feature space:

| Machine Learning Algorithm | Evaluation Accuracy | Area Under Curve (AUC) |
| :--- | :---: | :---: |
| **Naive Bayes** | 96.32% | 0.0971 |
| **Logistic Regression** | 96.50% | 0.9810 |
| **Random Forest** | 97.40% | 0.9840 |
| **Support Vector Machine (SVM)** | **98.21%** | **0.9830** |

### Project Winner
The **Support Vector Machine (SVM)** high-dimensional margin classifier achieved the highest overall evaluation accuracy of **98.21%**, solidifying its spot as our champion production model.

---

## 5. Deployment & Prediction

* **Model Persistence:** We utilized the `pickle` library to serialize and save the trained SVM model alongside the fitted TF-IDF vectorizer configuration. These serialized assets can be instantly deployed into web environments or automation networks without requiring pipeline retraining.
* **Interactive Predictor:** A deployment-ready helper function—`predict_spam()`—was coded to streamline inference. It handles raw string ingestion, applies the upstream preprocessing stack, runs data through the saved TF-IDF vocabulary matrix, and exposes the categorical classification status ("Spam" or "Ham") paired with real-time confidence scores.
