import streamlit as st
st.set_page_config(page_title="Fake News Detector", layout="wide")

import re
import nltk
import requests
import pickle
from nltk.corpus import stopwords
from streamlit_autorefresh import st_autorefresh

nltk.download('stopwords')

# ⏱️ Cached model & vectorizer loading
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
vectorizer = load_vectorizer()

# 🧼 Text cleaning
def clean_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [word for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# 📡 NewsAPI fetch
API_KEY = "ac985774b8bb448fbd720422fd53b8fc"
def fetch_news():
    url = f"https://newsapi.org/v2/everything?q=india&language=en&pageSize=5&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])
    else:
        return []

# 🖼️ Streamlit setup
st.title("📰 Fake News Detection System")

# 🎨 Custom CSS
st.markdown("""
    <style>
    html, body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f4f4f9;
    }
    .stTextArea textarea {
        background-color: #f1f1f1 !important;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 1rem;
        color: #333;
    }
    .news-card {
        background-color: #ffffff;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .headline {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: grey;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 Sidebar
with st.sidebar:
    st.title("🔍 About")
    st.info("This app detects **fake news** using a machine learning model.")
    st.markdown("Developed by *Neeraj Chandel and Khushi under the guidance of Dr. Sunita Soni (HOD CSE)* 👨‍💻")
    st.markdown("---")
    st.write("🌐 Uses NewsAPI for live headlines.")

st.markdown("Enter a news article or headline below to check if it's fake or real.")

# 📝 User input
user_input = st.text_area("📝 Enter News Text", height=100)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        preprocessed_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input])
        prediction = model.predict(vectorized_input)
        prob = model.predict_proba(vectorized_input).max()

        if prediction[0] == 1:
            st.error(f"🚨 This is **Fake News**! ")
        else:
            st.success(f"✅ This is **Real News**. ")

st.markdown("---")

# 🔄 Real-time News Stream
st.header("🌐 Real-Time News Detection")

refresh_interval = st.slider("⏱ Refresh Interval (seconds)", 0, 60, 30)
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="auto-refresh")

articles = fetch_news()
if not articles:
    st.error("⚠️ Could not fetch news. Please check API key or try again.")
else:
    for article in articles:
        title = article.get("title", "No title")
        st.markdown(f"""
            <div class="news-card">
                <div class="headline">📰 {title}</div>
        """, unsafe_allow_html=True)

        preprocessed = clean_text(title)
        vectorized = vectorizer.transform([preprocessed])
        pred = model.predict(vectorized)
        prob = model.predict_proba(vectorized).max()


        label = "Fake" if pred[0] == 1 else "Real"
        emoji = "🚨" if label == "Fake" else "✅"
        st.write(f"**Prediction:** {emoji} {label} | ")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("___")
