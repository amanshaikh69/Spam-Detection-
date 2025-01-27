import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data = data.rename(columns={'v1': 'label', 'v2': 'message'})
    data = data[['label', 'message']]
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data


def preprocess_text(text, method="none"):
    # Lowercase text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize words
    words = text.split()
    # Apply preprocessing method
    if method == "stemming":
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words if word not in stopwords.words("english")]
    elif method == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
    elif method == "stopwords":
        words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)


def train_model(preprocessing="none", vectorizer="count", model_type="naive_bayes"):
    data = load_data()
    data['processed_message'] = data['message'].apply(lambda x: preprocess_text(x, method=preprocessing))
    
    X = data['processed_message']
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text
    if vectorizer == "count":
        vectorizer_obj = CountVectorizer()
    elif vectorizer == "tfidf":
        vectorizer_obj = TfidfVectorizer()
    X_train_vectorized = vectorizer_obj.fit_transform(X_train)
    X_test_vectorized = vectorizer_obj.transform(X_test)
    
    # Train model
    if model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC()
    
    model.fit(X_train_vectorized, y_train)
    return model, vectorizer_obj, X_test_vectorized, y_test


def classification_report_as_dataframe(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()


def main():
    st.set_page_config(page_title="Enhanced Spam Detection App", page_icon="📧", layout="wide")
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Menu", ["Home", "Predict", "Model Evaluation", "About"])

    if menu == "Home":
        st.title("📧 Enhanced Spam Detection App")
        st.markdown("""
        This app allows you to classify SMS messages as Spam or Ham, practice various NLP techniques, and benchmark different models.
        - **Preprocessing Options**: Stemming, Lemmatization, Stopword Removal
        - **Models**: Naive Bayes, Logistic Regression, Random Forest, SVM
        """)
        st.image("https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minimal-email-header.jpg", use_column_width=True)

    elif menu == "Predict":
        st.title("📩 Spam or Ham Prediction")
        preprocessing = st.selectbox("Select Preprocessing Technique", ["none", "stemming", "lemmatization", "stopwords"])
        vectorizer = st.selectbox("Select Vectorizer", ["count", "tfidf"])
        model_type = st.selectbox("Select Model", ["naive_bayes", "logistic_regression", "random_forest", "svm"])
        
        user_message = st.text_area("Type your SMS message here:")
        if st.button("Classify"):
            if not user_message.strip():
                st.warning("⚠️ Please enter a valid message!")
            else:
                with st.spinner("Classifying..."):
                    model, vectorizer_obj, _, _ = train_model(preprocessing, vectorizer, model_type)
                    user_message_vectorized = vectorizer_obj.transform([preprocess_text(user_message, preprocessing)])
                    prediction = model.predict(user_message_vectorized)[0]
                    label = "Spam" if prediction == 1 else "Ham"
                st.success(f"✅ The message is classified as: **{label}**")

    elif menu == "Model Evaluation":
        st.title("📊 Model Evaluation")
        preprocessing = st.selectbox("Select Preprocessing Technique for Evaluation", ["none", "stemming", "lemmatization", "stopwords"])
        vectorizer = st.selectbox("Select Vectorizer for Evaluation", ["count", "tfidf"])
        model_type = st.selectbox("Select Model for Evaluation", ["naive_bayes", "logistic_regression", "random_forest", "svm"])
        
        model, vectorizer_obj, X_test, y_test = train_model(preprocessing, vectorizer, model_type)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.2f}")
        
        st.subheader("Classification Report")
        report_df = classification_report_as_dataframe(y_test, y_pred)
        st.dataframe(report_df.style.background_gradient(cmap="Blues"))
        
        st.subheader("Confusion Matrix")
        confusion = confusion_matrix(y_test, y_pred)
        st.table(pd.DataFrame(confusion, index=["Actual Ham", "Actual Spam"], columns=["Predicted Ham", "Predicted Spam"]))

    elif menu == "About":
        st.title("ℹ️ About the App")
        st.markdown("""
        This app demonstrates text classification with various preprocessing techniques and models.
        - Project By = Aman Shaikh
        - Email ID = amannshaikh0208@gmail.com
        """)


if __name__ == "__main__":
    main()

