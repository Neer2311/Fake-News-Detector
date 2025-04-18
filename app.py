import streamlit as st
import re
import nltk
import requests
import pickle
from nltk.corpus import stopwords
from streamlit_autorefresh import st_autorefresh

nltk.download('stopwords')

# Page config
st.set_page_config(page_title="📰 Fake News Detector", layout="wide")


# Load model and vectorizer
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

# Clean input
def clean_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [word for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Fetch news
API_KEY = "ac985774b8bb448fbd720422fd53b8fc"
def fetch_news():
    url = f"https://newsapi.org/v2/everything?q=india&language=en&pageSize=5&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body {
        font-family: 'Inter', sans-serif;
        background-color: #f4f4f9;
    }

    .stTextArea textarea {
        background-color: #fff !important;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 1rem;
        color: #333;
    }

    .news-card {
        background-color: #ffffffcc;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }

    .headline {
        font-size: 1.1rem;
        font-weight: 600;
        color: #202020;
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: grey;
        margin-top: 2rem;
    }

    /* 🔘 Button Animation */
    .stButton>button {
        background-color: #0078FF;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #005ec2;
        transform: scale(1.05);
        box-shadow: 0 6px 14px rgba(0,0,0,0.2);
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- 📌 Sidebar ----------
with st.sidebar:
    st.title("🔍 About")
    st.info("This app detects **fake news** using a machine learning model.")
    st.markdown("Developed by *Neeraj Chandel and Khushi* under the guidance of **Dr. Sunita Soni (HOD CSE)** 👨‍💻")
    st.markdown("---")
    st.write("🌐 Powered by NewsAPI.org")
    st.markdown("🛠️ Built with Python, Streamlit, Scikit-learn")

# ---------- 🧠 Title & Input ----------
st.markdown("## 📰 Fake News Detection System")
st.markdown("Enter a **headline or article snippet** below to check if it's **Fake** or **Real**.")

user_input = st.text_area("✍️ Enter News Text", height=120)

if st.button("🔎 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        preprocessed_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input])
        prediction = model.predict(vectorized_input)
        prob = model.predict_proba(vectorized_input).max()

        if prediction[0] == 1:
            st.error(f"🚨 This appears to be **Fake News**! ")
        else:
            st.success(f"✅ This appears to be **Real News**. ")

st.markdown("---")
st.markdown("## 🌐 Real-Time News Headlines")

# ---------- 🔄 Real-Time News Stream ----------
refresh_interval = st.slider("⏱ Refresh Interval (seconds)", 0, 60, 30)
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="auto-refresh")

articles = fetch_news()
if not articles:
    st.error("⚠️ Could not fetch news. Please check your API key or try again later.")
else:
    for article in articles:
        title = article.get("title", "No title available")
        st.markdown(f"""<div class="news-card"><div class="headline">🗞 {title}</div>""", unsafe_allow_html=True)

        preprocessed = clean_text(title)
        vectorized = vectorizer.transform([preprocessed])
        pred = model.predict(vectorized)
        prob = model.predict_proba(vectorized).max()

        label = "Fake" if pred[0] == 1 else "Real"
        emoji = "🚨" if label == "Fake" else "✅"
        st.write(f"**Prediction:** {emoji} {label} | ")
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("___")
