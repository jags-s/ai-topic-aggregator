import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import sources
import threading
import os
from dateutil import parser

# Set page configuration
st.set_page_config(
    page_title="AI Topic Aggregator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .article-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #FF4B4B;
    }
    .article-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .article-meta {
        font-size: 14px;
        color: #666;
        margin-bottom: 10px;
    }
    .article-summary {
        font-size: 15px;
        margin-bottom: 10px;
    }
    .trending-score {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 12px;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-text {
        display: inline-block;
    }
    .logo-image {
        width: 24px;
        height: 24px;
        margin-right: 8px;
        vertical-align: middle;
    }
    .filter-container {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for articles cache
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'is_fetching' not in st.session_state:
    st.session_state.is_fetching = False

def fetch_articles_background(days, keyword, source):
    """Fetch articles in the background"""
    st.session_state.is_fetching = True
    try:
        st.session_state.fetch_error = None
        # Add debug log
        st.session_state.debug_log = []
        st.session_state.debug_log.append(f"Fetching articles with: days={days}, keyword={keyword}, source={source}")
        
        # Get sources
        all_sources = sources.get_sources()
        st.session_state.debug_log.append(f"Found {len(all_sources)} sources to fetch from")
        
        try:
            articles = sources.fetch_all_articles(days=days, keyword_filter=keyword, source_filter=source)
            st.session_state.debug_log.append(f"Fetched {len(articles)} articles total")
            st.session_state.articles = articles
            st.session_state.last_update = datetime.now()
        except Exception as e:
            st.session_state.fetch_error = str(e)
            st.session_state.debug_log.append(f"Error fetching articles: {str(e)}")
            import traceback
            st.session_state.debug_log.append(traceback.format_exc())
    finally:
        st.session_state.is_fetching = False

def format_date(date_str):
    """Format a date string to a friendly format"""
    try:
        dt = parser.parse(date_str)
        return dt.strftime("%b %d, %Y")
    except:
        return date_str

# App header
st.title("🤖 AI Topic Aggregator")
st.markdown("Stay updated with the latest trending articles from top LLM providers")

# Sidebar with filters
st.sidebar.header("Filters")

# Time period filter
days_options = [7, 14, 30, 60, 90]
days_filter = st.sidebar.selectbox(
    "Time Period",
    options=days_options,
    index=2,  # Default to 30 days
    format_func=lambda x: f"Last {x} days"
)

# Source filter
all_sources = [s.name for s in sources.get_sources()]
source_filter = st.sidebar.selectbox(
    "Provider",
    options=["All"] + all_sources,
    index=0
)

# Keyword filter
keyword_filter = st.sidebar.text_input("Keyword Search")

# Topics of interest
topics = [
    "Artificial Intelligence (AI)",
    "Large Language Models (LLM)",
    "Agentic AI",
    "Model Context",
    "RAG (Retrieval Augmented Generation)",
    "LoRA (Low-Rank Adaptation)",
    "MoE (Mixture of Experts)",
    "AI Agents",
    "Fine-tuning",
    "Prompt Engineering"
]
selected_topics = st.sidebar.multiselect(
    "Topics of Interest",
    options=topics
)

# Refresh button
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    refresh = st.button("🔄 Refresh Articles")

# Show last update time
if st.session_state.last_update:
    st.sidebar.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Prepare for fetching
source_name = None if source_filter == "All" else source_filter
keyword = None if not keyword_filter else keyword_filter

# Add topic keywords to search if selected
if selected_topics:
    if keyword:
        keyword = f"{keyword} {' '.join(selected_topics)}"
    else:
        keyword = " ".join(selected_topics)

# Fetch articles when needed
if (not st.session_state.articles or 
    refresh or 
    not st.session_state.last_update or 
    (datetime.now() - st.session_state.last_update).total_seconds() > 3600):  # Auto-refresh after 1 hour
    
    if not st.session_state.is_fetching:
        with st.spinner("Fetching latest articles..."):
            # Start fetching in a thread to avoid blocking UI
            thread = threading.Thread(
                target=fetch_articles_background,
                args=(days_filter, keyword, source_name)
            )
            thread.start()
            # Wait for the thread to complete (simplified approach for Streamlit)
            while st.session_state.is_fetching:
                time.sleep(0.1)

# Display a message if fetching
if st.session_state.is_fetching:
    st.info("Fetching the latest articles, please wait...")

# Show debug info if available
if hasattr(st.session_state, 'debug_log') and st.session_state.debug_log:
    with st.expander("Debug Information"):
        for log in st.session_state.debug_log:
            st.text(log)

# Show error if available
if hasattr(st.session_state, 'fetch_error') and st.session_state.fetch_error:
    st.error(f"Error fetching articles: {st.session_state.fetch_error}")

# Display source test results if available
if hasattr(st.session_state, 'source_test_results'):
    st.subheader("Source Test Results")
    
    # Count total articles found
    total_articles = sum(result["count"] for result in st.session_state.source_test_results.values())
    
    # Create a list to collect articles from test results
    if 'collected_articles' not in st.session_state:
        st.session_state.collected_articles = []
    
    # Display results and gather articles
    for source_name, result in st.session_state.source_test_results.items():
        if result["success"] and result["count"] > 0:
            st.success(f"{source_name}: {result['message']}")
            # If we have successful articles but they're not in the main display,
            # add them to session state articles
            if 'test_articles' in result and result['test_articles']:
                st.session_state.collected_articles.extend(result['test_articles'])
        elif result["success"]:
            st.info(f"{source_name}: {result['message']}")
        else:
            st.error(f"{source_name}: {result['message']}")
    
    # If we found articles in testing but not in main display, use them
    if total_articles > 0 and (not st.session_state.articles or len(st.session_state.articles) == 0):
        st.info(f"Using {total_articles} articles found during testing")
        # Use any collected articles
        if st.session_state.collected_articles:
            st.session_state.articles = st.session_state.collected_articles

# Show articles if available
if st.session_state.articles and len(st.session_state.articles) > 0:
    # Show result count
    st.subheader(f"Top {len(st.session_state.articles)} Trending Articles")
    
    # Display articles
    for i, article in enumerate(st.session_state.articles):
        with st.container():
            st.markdown(
                f"""
                <div class="article-card">
                    <div class="header-container">
                        <div class="header-text">
                            <span class="trending-score">#{i+1}</span> 
                            <img src="{article.get('logo_url', '')}" class="logo-image" onerror="this.style.display='none'">
                            <span>{article.get('source', 'Unknown')}</span>
                        </div>
                        <div>
                            <span class="trending-score">Score: {int(article.get('trending_score', 0))}</span>
                        </div>
                    </div>
                    <div class="article-title">{article.get('title', 'No Title')}</div>
                    <div class="article-meta">Published: {format_date(article.get('published', ''))}</div>
                    <div class="article-summary">{article.get('summary', 'No summary available.')}</div>
                    <a href="{article.get('link', '#')}" target="_blank">Read More</a>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    if st.session_state.is_fetching:
        st.info("Fetching articles, please wait...")
    else:
        st.warning("No articles found. Try changing your filters or refresh.")
        
        # Add a button to test each source individually
        if st.button("Test Each Source"):
            st.session_state.source_test_results = {}
            st.session_state.collected_articles = []
            
            for source in sources.get_sources():
                try:
                    articles = source.fetch_articles(days=30)
                    
                    # Store the actual articles in the test results
                    st.session_state.source_test_results[source.name] = {
                        "success": True,
                        "count": len(articles),
                        "message": f"Successfully fetched {len(articles)} articles",
                        "test_articles": articles  # Store the actual articles
                    }
                    
                    # Also add to our collected articles
                    if articles:
                        st.session_state.collected_articles.extend(articles)
                except Exception as e:
                    st.session_state.source_test_results[source.name] = {
                        "success": False,
                        "count": 0,
                        "message": str(e)
                    }
            
            # If we collected any articles, add them to the main articles list
            if st.session_state.collected_articles:
                st.session_state.articles = st.session_state.collected_articles
                
            # Use st.rerun() instead of experimental_rerun
            st.rerun()
            
        # We don't need to display source test results here as they're already displayed above

# Footer
st.markdown("---")
st.markdown("© 2025 AI Topic Aggregator | Powered by Streamlit")
