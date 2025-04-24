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

#required NLTK resources
import os
nltk_path = os.path.join(os.path.expanduser("~"), "nltk_data")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_path)

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

# Clean input for model
def clean_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [word for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

def safe_sent_tokenize(text):
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        # Fallback: naive sentence splitting if punkt fails
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
# Google Fact Checker Class
class GoogleFactChecker:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.cache = {}  # Cache to store previous results
        self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
    
    def extract_key_claims(self, article_text):
        """Extract the most important claims from an article for fact checking"""
        # Split text into sentences
        sentences = safe_sent_tokenize(article_text)
        
        # If there are too many sentences, focus on:
        # 1. First few sentences (usually contain the main claim)
        # 2. Sentences with potential claims (containing specific keywords)
        key_sentences = []
        claim_indicators = ["claim", "said", "says", "according to", "stated", "reported", 
                           "confirmed", "announced", "revealed", "alleged", "suggests"]
        
        # Always include the first 2 sentences (often contain the main claim)
        key_sentences.extend(sentences[:min(2, len(sentences))])
        
        # Add sentences with claim indicators
        for sentence in sentences[2:]:
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                key_sentences.append(sentence)
        
        # If we have too many or too few sentences, adjust
        if len(key_sentences) > 3:
            # Prioritize sentences with claim indicators
            key_sentences = key_sentences[:3]
        elif len(key_sentences) == 0 and sentences:
            # If no key sentences identified, use the first sentence
            key_sentences = [sentences[0]]
        
        return key_sentences
    
    def clean_query(self, query):
        """Clean and prepare a query for the fact check API"""
        # Remove special characters but keep basic punctuation
        query = re.sub(r'[^\w\s.,!?"\']', ' ', query)
        
        # Trim whitespace and ensure proper length
        query = query.strip()
        
        # API has query length limitations - ensure it's not too long
        if len(query) > 150:
            # Truncate but try to keep complete sentences
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
        # First check cache
        cleaned_claim = self.clean_query(claim)
        cached_result = self.check_cache(cleaned_claim)
        if cached_result:
            return cached_result
        
        # Prepare the API request
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
                    
                    # Cache the results
                    self.cache[cleaned_claim] = (time.time(), results)
                    return results
                else:
                    # No claims found, but request was successful
                    self.cache[cleaned_claim] = (time.time(), [])
                    return []
            else:
                # Handle API error
                st.warning(f"Fact check API returned status code {response.status_code}")
                return []
                
        except Exception as e:
            st.warning(f"Error during fact check: {str(e)}")
            return []
    
    def fact_check_article(self, article_text):
        """Extract claims from article and check them"""
        # Extract key claims/sentences
        key_claims = self.extract_key_claims(article_text)
        
        # Check each claim
        all_results = []
        for claim in key_claims:
            results = self.fact_check_claim(claim)
            if results:  # Only add if we found fact checks
                all_results.extend(results)
        
        return all_results
    
    def get_credibility_score(self, fact_check_results):
        """Calculate a credibility score based on fact check results"""
        if not fact_check_results:
            return None  # No fact checks found
        
        # Define rating categories and their score impacts
        rating_impacts = {
            "true": 1.0,
            "mostly true": 0.8,
            "half true": 0.5,
            "mixed": 0.5,
            "mostly false": 0.2,
            "false": 0.0,
            "pants on fire": -0.5,  # Extra penalty for egregious falsehoods
        }
        
        # Calculate score based on ratings
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

# Fetch news with improved error handling and fallbacks
API_KEY = "ac985774b8bb448fbd720422fd53b8fc"  # Consider storing this in Streamlit secrets

@st.cache_data(ttl=300)  # Cache results for 5 minutes
def fetch_news():
    """Fetch news with improved error handling and fallbacks"""
    url = f"https://newsapi.org/v2/everything?q=india&language=en&pageSize=5&sortBy=publishedAt&apiKey={API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "articles" in data and len(data["articles"]) > 0:
                return data["articles"]
            else:
                st.warning("NewsAPI returned no articles. Using fallback news.")
                return get_fallback_news()
        else:
            st.warning(f"NewsAPI returned status code: {response.status_code}")
            # Only log detailed error in development
            if not st.secrets.get("production", False):
                st.error(f"Error details: {response.text}")
            return get_fallback_news()
    except Exception as e:
        st.warning(f"Error fetching news: {str(e)}")
        return get_fallback_news()

def get_fallback_news():
    """Provide fallback news when API fails"""
    return [
        {
            "title": "AI Technologies Reshaping Healthcare Industry with New Diagnostic Tools",
            "source": {"name": "Tech Today"},
            "url": "https://example.com/news1",
            "publishedAt": "2025-04-24T12:00:00Z",
            "content": "AI technologies continue to transform healthcare with new diagnostic tools..."
        },
        {
            "title": "Economic Indicators Show Strong Recovery in Manufacturing Sector",
            "source": {"name": "Business Daily"},
            "url": "https://example.com/news2",
            "publishedAt": "2025-04-24T10:30:00Z",
            "content": "Recent economic indicators suggest a strong recovery in the manufacturing sector..."
        },
        {
            "title": "Climate Change Initiatives Gain International Support at Global Summit",
            "source": {"name": "World Report"},
            "url": "https://example.com/news3",
            "publishedAt": "2025-04-23T22:15:00Z",
            "content": "Global leaders agreed to new climate initiatives during the recent summit..."
        },
        {
            "title": "New Education Policies Aim to Bridge Digital Divide Among Students",
            "source": {"name": "Education Weekly"},
            "url": "https://example.com/news4",
            "publishedAt": "2025-04-23T18:45:00Z",
            "content": "Government announces new policies to ensure all students have access to digital learning tools..."
        },
        {
            "title": "Space Exploration: Next Generation Telescope Reveals New Exoplanets",
            "source": {"name": "Science Today"},
            "url": "https://example.com/news5",
            "publishedAt": "2025-04-22T14:20:00Z",
            "content": "Astronomers discover potentially habitable exoplanets using the new orbital telescope..."
        }
    ]

# Create a function to attempt alternative news sources if primary fails
def try_alternative_news_source():
    """Try an alternative news source if NewsAPI fails"""
    # This is a placeholder - you might want to implement an actual alternative API
    # For now, we'll just return the fallback news
    return get_fallback_news()

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

    .api-status {
        font-size: 0.8rem;
        color: #6b7280;
        text-align: right;
        margin-bottom: 0.5rem;
    }

    .fallback-notice {
        padding: 0.5rem;
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        margin-bottom: 1rem;
        font-size: 0.9rem;
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
    
    # Add API status indicator in sidebar
    st.markdown("### API Status")
    
    # Check if NewsAPI is working
    try:
        test_url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey={API_KEY}"
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            st.success("✅ NewsAPI: Online")
        else:
            st.error("❌ NewsAPI: Offline (using fallback data)")
    except:
        st.error("❌ NewsAPI: Offline (using fallback data)")
    
    # Check if Google Fact Check API is working
    try:
        test_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query=climate&key={fact_checker.api_key}"
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            st.success("✅ Google Fact Check: Online")
        else:
            st.warning("⚠️ Google Fact Check: Issues detected")
    except:
        st.warning("⚠️ Google Fact Check: Issues detected")

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

# Try to fetch news with fallbacks
articles = fetch_news()

# Check if we're using fallback data
is_fallback = len([a for a in articles if a.get("url", "").startswith("https://example.com")]) > 0

if is_fallback:
    st.markdown("<div class='fallback-notice'>⚠️ Using sample news data. API connection unavailable.</div>", unsafe_allow_html=True)

# Display articles (whether from API or fallback)
for article in articles:
    title = article.get("title", "No Title")
    source = article.get("source", {}).get("name", "Unknown Source")
    url = article.get("url", "")
    published_at = article.get("publishedAt", "Unknown Time")

    st.markdown(f"""
        <div class="news-card">
            <div class="headline">{title}</div>
            <div><strong>Source:</strong> {source} | <strong>Published at:</strong> {published_at}</div>
            <div><a href="{url}" target="_blank">🔗 Read Full Article</a></div>
        </div>
    """, unsafe_allow_html=True)

    # Analyze headline with fact checker and model
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

# Footer
st.markdown("___")
st.markdown("<div class='footer'>© 2025 Fake News Detection System</div>", unsafe_allow_html=True)

