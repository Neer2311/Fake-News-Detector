import streamlit as st
import re
import nltk
import requests
import pickle
import time
from urllib.parse import quote_plus
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="📰 Fake News Detector", layout="wide")


@st.cache_resource
def download_nltk_resources():
    try:
     
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
            return True
        except LookupError:
        
            nltk.download('punkt', download_dir='./.nltk_data')
            nltk.data.path.append('./.nltk_data')
            return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {str(e)}")
        return False


resources_available = download_nltk_resources()


def safe_sent_tokenize(text):
    if resources_available:
        try:
            return sent_tokenize(text)
        except LookupError:
            
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    else:
        
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

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


def clean_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [word for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

class GoogleFactChecker:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.cache = {}  # Cache to store previous results
        self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
    
    def extract_key_claims(self, article_text):
        """Extract the most important claims from an article for fact checking"""
        # Split text into sentences using the safe tokenizer
        sentences = safe_sent_tokenize(article_text)
        
     
        key_sentences = []
        claim_indicators = ["claim", "said", "says", "according to", "stated", "reported", 
                           "confirmed", "announced", "revealed", "alleged", "suggests"]
        
        # Always include the first 2 sentences (often contain the main claim)
        key_sentences.extend(sentences[:min(2, len(sentences))])
        
    
        for sentence in sentences[2:]:
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                key_sentences.append(sentence)
        
       
        if len(key_sentences) > 3:
           
            key_sentences = key_sentences[:3]
        elif len(key_sentences) == 0 and sentences:
           
            key_sentences = [sentences[0]]
        
        return key_sentences
    
    def clean_query(self, query):
        """Clean and prepare a query for the fact check API"""
    
        query = re.sub(r'[^\w\s.,!?"\']', ' ', query)
        
        query = query.strip()
        
     
        if len(query) > 150:
            
            words = query.split()
            query = ' '.join(words[:20])  # Keep first ~20 words
        
        return query
    
    def check_cache(self, query):
        """Check if we have cached results for this query"""
        if query in self.cache:
            timestamp, results = self.cache[query]
            if time.time() - timestamp < self.cache_expiry:
                return results
        return None
    
    def fact_check_claim(self, claim):
        """Check a single claim using Google Fact Check API"""
   
        cleaned_claim = self.clean_query(claim)
        cached_result = self.check_cache(cleaned_claim)
        if cached_result:
            return cached_result
        
        
        url = f"{self.base_url}?query={quote_plus(cleaned_claim)}&key={self.api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'claims' in data and data['claims']:
                    results = []
                    for claim_data in data['claims'][:3]:  # Limit to top 3 results per claim
                        review = claim_data.get("claimReview", [{}])[0]
                        result = {
                            "claim": claim_data.get("text", "No claim text"),
                            "claimant": claim_data.get("claimant", "Unknown"),
                            "rating": review.get("textualRating", "Not Rated"),
                            "publisher": review.get("publisher", {}).get("name", "Unknown"),
                            "url": review.get("url", ""),
                            "date": review.get("reviewDate", ""),
                            "original_query": claim
                        }
                        results.append(result)
                    
         
                    self.cache[cleaned_claim] = (time.time(), results)
                    return results
                else:
                  
                    self.cache[cleaned_claim] = (time.time(), [])
                    return []
            else:
              
                st.warning(f"Fact check API returned status code {response.status_code}")
                return []
                
        except Exception as e:
            st.warning(f"Error during fact check: {str(e)}")
            return []
    
    def fact_check_article(self, article_text):
        """Extract claims from article and check them"""
       
        key_claims = self.extract_key_claims(article_text)
        
       
        all_results = []
        for claim in key_claims:
            results = self.fact_check_claim(claim)
            if results: 
                all_results.extend(results)
        
        return all_results
    
    def get_credibility_score(self, fact_check_results):
        """Calculate a credibility score based on fact check results"""
        if not fact_check_results:
            return None  # No fact checks found
        
        
        rating_impacts = {
            "true": 1.0,
            "mostly true": 0.8,
            "half true": 0.5,
            "mixed": 0.5,
            "mostly false": 0.2,
            "false": 0.0,
            "pants on fire": -0.5, 
        }
        
        
        total_score = 0
        count = 0
        
        for result in fact_check_results:
            rating = result.get("rating", "").lower()
            
            # Check for match with our categories
            score_impact = None
            for category, impact in rating_impacts.items():
                if category in rating:
                    score_impact = impact
                    break
            
            # If no match found but we have a rating, use neutral score
            if score_impact is None and rating:
                score_impact = 0.5
                
            # Only count if we determined a score
            if score_impact is not None:
                total_score += score_impact
                count += 1
        
        # Return normalized score if we have any ratings
        if count > 0:
            return total_score / count
        else:
            return None

# Initialize the fact checker
@st.cache_resource
def initialize_fact_checker():
    return GoogleFactChecker(api_key="AIzaSyASFDHY0-f_2c3DgC2zqzV0OeEh43dLd0Q")

fact_checker = initialize_fact_checker()

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
        background-color: #f7f9fc;
        color: #1f2937;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 1rem;
        color: #111827;
    }

    .news-card {
        background-color: #ffffff;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }

    .headline {
        font-size: 1.125rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 2rem;
    }

    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #2563eb;
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        cursor: pointer;
    }

    .fact-check-result {
        background-color: #f9fafb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
    }

    .fact-check-true {
        border-left: 4px solid #10b981;
    }

    .fact-check-false {
        border-left: 4px solid #ef4444;
    }

    .fact-check-mixed {
        border-left: 4px solid #f59e0b;
    }

    .credibility-meter {
        height: 8px;
        width: 100%;
        background-color: #e5e7eb;
        border-radius: 4px;
        margin: 8px 0;
    }

    .credibility-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- 📌 Sidebar ----------
with st.sidebar:
    st.title("🔍 About")
    st.info("This app detects **fake news** using a machine learning model and Google's Fact Check API.")
    st.markdown("Developed by *Neeraj Chandel and Khushi* under the guidance of **Dr. Sunita Soni (HOD CSE)** 👨‍💻")
    st.markdown("---")
    st.write("🌐 Powered by NewsAPI.org and Google Fact Check Tools")
    st.markdown("🛠️ Built with Python, Streamlit, Scikit-learn")

# ---------- 🧠 Title & Input ----------
st.markdown("## 📰 Fake News Detection System")
st.markdown("Enter a **headline or article snippet** below to check if it's **Fake** or **Real**.")

user_input = st.text_area("✍️ Enter News Text", height=120)

if st.button("🔎 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Model prediction
        preprocessed_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([preprocessed_input])
        prediction = model.predict(vectorized_input)
        prob = model.predict_proba(vectorized_input).max()
        
        # Fact checking
        with st.spinner("Checking facts..."):
            fact_check_results = fact_checker.fact_check_article(user_input)
            credibility_score = fact_checker.get_credibility_score(fact_check_results)
        
        # Display initial model results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 AI Model Analysis")
            if prediction[0] == 1:
                st.error(f"🚨 AI classifies as **Fake News**")
                st.markdown(f"Model confidence: **{prob*100:.1f}%**")
            else:
                st.success(f"✅ AI classifies as **Real News**")
                st.markdown(f"Model confidence: **{prob*100:.1f}%**")
        
        with col2:
            st.subheader("🔍 Fact Check Analysis")
            if credibility_score is not None:
                # Display credibility meter
                st.markdown("<div class='credibility-meter'><div class='credibility-fill' style='width:{}%;'></div></div>".format(
                    credibility_score * 100), unsafe_allow_html=True)
                
                if credibility_score >= 0.7:
                    st.success(f"✅ Fact checks suggest **LIKELY TRUE**")
                elif credibility_score >= 0.4:
                    st.warning(f"⚠️ Fact checks suggest **MIXED RELIABILITY**")
                else:
                    st.error(f"❌ Fact checks suggest **LIKELY FALSE**")
                st.markdown(f"Credibility score: **{credibility_score*100:.1f}%**")
            else:
                st.info("⚠️ No fact-check data available")
        
        # Combined assessment
        st.markdown("---")
        st.subheader("📊 Final Assessment")
        
        # Determine final verdict (combining model and fact checks)
        if credibility_score is not None:
            # We have fact checks - give them more weight
            if credibility_score < 0.3:
                st.error("🚨 **VERDICT: FALSE** - Fact checks indicate this is likely false information")
            elif credibility_score > 0.7:
                st.success("✅ **VERDICT: TRUE** - Fact checks indicate this is likely accurate information")
            else:
                st.warning("⚠️ **VERDICT: MIXED** - This content contains a mix of true and false claims")
        else:
            # No fact checks - rely on model
            if prediction[0] == 1:
                st.error("🚨 **VERDICT: LIKELY FALSE** - Our AI model flags this as potential fake news")
            else:
                st.success("✅ **VERDICT: LIKELY TRUE** - No red flags detected by our AI model")
        
        # Display detailed fact check results
        st.markdown("---")
        st.subheader("Detailed Fact Check Results")
        
        if fact_check_results:
            for i, claim in enumerate(fact_check_results):
                # Determine which CSS class to use based on rating
                css_class = "fact-check-result"
                rating = claim['rating'].lower()
                if "true" in rating and "false" not in rating:
                    css_class += " fact-check-true"
                elif "false" in rating or "pants" in rating:
                    css_class += " fact-check-false"
                else:
                    css_class += " fact-check-mixed"
                
                st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                st.markdown(f"**Claim {i+1}:** {claim['claim']}")
                st.markdown(f"**Claimant:** {claim['claimant']}")
                st.markdown(f"**Rating:** {claim['rating']} by {claim['publisher']}")
                if claim['url']:
                    st.markdown(f"**Source:** [View full fact check]({claim['url']})")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No specific fact-checked claims found for this text. This could mean:")
            st.markdown("- The claims in this text haven't been fact-checked yet")
            st.markdown("- The text doesn't contain specific verifiable claims")
            st.markdown("- The claims were worded differently than in fact check databases")

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
        st.markdown(f"""<div class=\"news-card\"><div class=\"headline\">📾 {title}</div>""", unsafe_allow_html=True)

        # Use the enhanced fact checker for live news too
        with st.spinner("Analyzing..."):
            # Model prediction
            preprocessed = clean_text(title)
            vectorized = vectorizer.transform([preprocessed])
            pred = model.predict(vectorized)
            prob = model.predict_proba(vectorized).max()
            
            # Quick fact check (just the title)
            fact_results = fact_checker.fact_check_claim(title)
            fact_score = fact_checker.get_credibility_score(fact_results)
        
        # Show quick analysis
        col1, col2 = st.columns(2)
        
        with col1:
            label = "Fake" if pred[0] == 1 else "Real"
            emoji = "🚨" if label == "Fake" else "✅"
            st.write(f"**AI Model:** {emoji} {label} ({prob*100:.0f}% confidence)")
        
        with col2:
            if fact_score is not None:
                if fact_score >= 0.7:
                    st.write("**Fact Check:** ✅ Verified True")
                elif fact_score >= 0.4:
                    st.write("**Fact Check:** ⚠️ Mixed Claims")
                else:
                    st.write("**Fact Check:** ❌ Contains False Claims")
            else:
                st.write("**Fact Check:** ⚙️ No fact checks found")
                
        # Add link to full article
        if article.get("url"):
            st.markdown(f"[Read full article]({article.get('url')})")
            
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("___")
st.markdown("<div class='footer'>© 2025 Fake News Detection System</div>", unsafe_allow_html=True)
