import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import re
import pandas as pd
from dateutil import parser
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Download NLTK data for summarization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleSource:
    """Base class for article sources"""
    
    def __init__(self, name, base_url, logo_url=None):
        self.name = name
        self.base_url = base_url
        self.logo_url = logo_url
        
    def fetch_articles(self, days=30):
        """Fetch articles from the source"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_social_metrics(self, url):
        """Get social media metrics for an article"""
        # This is a placeholder for actual social media API integration
        # In a real implementation, you would use Twitter, LinkedIn, Reddit APIs
        # For now, we'll return random values for demonstration
        import random
        # Only generate random metrics if we have a valid URL
        if url and url.startswith(('http://', 'https://')):
            return {
                'twitter_shares': random.randint(5, 500),
                'linkedin_shares': random.randint(5, 300),
                'reddit_score': random.randint(1, 100)
            }
        else:
            # Return minimal values for invalid URLs
            return {
                'twitter_shares': 0,
                'linkedin_shares': 0,
                'reddit_score': 0
            }
    
    def calculate_trending_score(self, article):
        """Calculate a trending score for an article based on metrics"""
        metrics = self.get_social_metrics(article.get('link', ''))
        
        # Simple formula: sum of social shares with weights
        score = (
            metrics.get('twitter_shares', 0) * 1.0 + 
            metrics.get('linkedin_shares', 0) * 1.5 +
            metrics.get('reddit_score', 0) * 2.0
        )
        
        # Add recency bonus (newer articles get higher scores)
        days_old = (datetime.now() - parser.parse(article.get('published', datetime.now().isoformat()))).days
        recency_bonus = max(0, 30 - days_old) * 5  # 5 points per day of recency (up to 30 days)
        
        return score + recency_bonus
    
    def summarize_text(self, text, max_sentences=3):
        """Create a simple summary of article text"""
        if not text:
            return ""
            
        # Simple extractive summarization
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
            
        return " ".join(sentences[:max_sentences]) + "..."


class RSSSource(ArticleSource):
    """Article source that uses RSS feeds"""
    
    def __init__(self, name, rss_url, base_url, logo_url=None):
        super().__init__(name, base_url, logo_url)
        self.rss_url = rss_url
    
    def fetch_articles(self, days=30):
        """Fetch articles from RSS feed"""
        logger.info(f"Fetching articles from RSS: {self.name}")
        
        # Debug the feed content to understand what's happening
        debug_rss_feed(self.name, self.rss_url)
        
        try:
            # Try with timeout to avoid hanging
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/rss+xml, application/xml, text/xml, application/atom+xml, text/html, */*'
                }
                
                logger.info(f"{self.name}: Trying direct request to {self.rss_url}")
                response = requests.get(self.rss_url, headers=headers, timeout=20, verify=True)
                logger.info(f"{self.name}: RSS status code {response.status_code}")
                
                if response.status_code == 200:
                    # Try to parse response content
                    feed = feedparser.parse(response.content)
                    logger.info(f"{self.name}: Parsed feed with {len(feed.entries)} entries")
                else:
                    # Try direct feedparser parsing
                    logger.info(f"{self.name}: Status code not 200, trying feedparser directly")
                    feed = feedparser.parse(self.rss_url)
                    logger.info(f"{self.name}: Direct parsing found {len(feed.entries)} entries")
            except Exception as e:
                logger.error(f"Request error for {self.name}: {str(e)}")
                logger.info(f"{self.name}: Trying feedparser as fallback")
                feed = feedparser.parse(self.rss_url)
            
            logger.info(f"{self.name}: Found {len(feed.entries)} entries in feed")
            
            # Always use a naive cutoff date to avoid timezone comparison issues
            cutoff_date = datetime.now() - timedelta(days=days)
            logger.info(f"{self.name}: Using cutoff date {cutoff_date}")
            
            articles = []
            
            for entry in feed.entries:
                # Try to get the entry title and link first
                title = entry.get('title', 'No Title')
                link = entry.get('link', '')
                
                # Skip entries without title or link
                if title == 'No Title' or not link:
                    logger.warning(f"{self.name}: Skipping entry without title or link")
                    continue
                
                # Parse the publication date with multiple fallbacks
                try:
                    # Try multiple date fields
                    date_fields = ['published', 'updated', 'pubDate', 'created', 'date']
                    pub_date = None
                    
                    for field in date_fields:
                        if field in entry and entry[field]:
                            try:
                                pub_date = parser.parse(entry[field])
                                break
                            except:
                                continue
                    
                    # If we couldn't parse any date, use current time
                    if not pub_date:
                        pub_date = datetime.now()
                        
                    # Convert to naive datetime for consistent comparison
                    if pub_date.tzinfo:
                        # Convert to UTC then remove tzinfo
                        pub_date = pub_date.astimezone(timezone.utc).replace(tzinfo=None)
                except Exception as e:
                    logger.warning(f"{self.name}: Date parsing error: {str(e)}")
                    pub_date = datetime.now()
                
                # Skip articles older than cutoff date (both naive now)
                if pub_date < cutoff_date:
                    logger.info(f"{self.name}: Skipping article from {pub_date} - too old")
                    continue
                
                # Extract article content with multiple fallback options
                summary = ''
                
                # Try different content fields
                content_options = [
                    # Option 1: 'content' field (common in many feeds)
                    lambda: entry.get('content', [{}])[0].get('value', ''),
                    # Option 2: 'summary' field
                    lambda: entry.get('summary', ''),
                    # Option 3: 'description' field
                    lambda: entry.get('description', ''),
                    # Option 4: 'summary_detail' field
                    lambda: entry.get('summary_detail', {}).get('value', ''),
                    # Option 5: Just use the title if nothing else
                    lambda: title
                ]
                
                # Try each option until we get content
                for get_content in content_options:
                    try:
                        content = get_content()
                        if content:
                            # Parse HTML if present
                            soup = BeautifulSoup(content, 'html.parser')
                            text = soup.get_text()
                            if text.strip():
                                summary = self.summarize_text(text)
                                break
                    except Exception as e:
                        logger.warning(f"{self.name}: Error extracting content: {str(e)}")
                        continue
                
                # If we still don't have a summary, use the title
                if not summary:
                    summary = title
                
                article = {
                    'title': entry.get('title', 'No Title'),
                    'link': entry.get('link', ''),
                    'published': pub_date.isoformat(),
                    'summary': summary,
                    'source': self.name,
                    'logo_url': self.logo_url
                }
                
                # Add trending score
                article['trending_score'] = self.calculate_trending_score(article)
                
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed for {self.name}: {str(e)}")
            return []



class WebScraperSource(ArticleSource):
    """Article source that uses web scraping"""
    
    def __init__(self, name, blog_url, base_url, article_selector, title_selector, link_selector, date_selector=None, logo_url=None):
        super().__init__(name, base_url, logo_url)
        self.blog_url = blog_url
        self.article_selector = article_selector
        self.title_selector = title_selector
        self.link_selector = link_selector
        self.date_selector = date_selector
    
    def fetch_articles(self, days=30):
        """Fetch articles by scraping the blog page"""
        logger.info(f"Fetching articles via web scraping: {self.name}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            response = requests.get(self.blog_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            logger.info(f"{self.name}: Scraping with selector: {self.article_selector}")
            article_elements = soup.select(self.article_selector)
            logger.info(f"{self.name}: Found {len(article_elements)} article elements")
            
            cutoff_date = datetime.now() - timedelta(days=days)
            articles = []
            
            for element in article_elements:
                # Extract title
                title_element = element.select_one(self.title_selector)
                title = title_element.get_text().strip() if title_element else 'No Title'
                
                # Extract link
                link_element = element.select_one(self.link_selector)
                link = link_element.get('href') if link_element else ''
                
                # Make sure link is absolute
                if link and not link.startswith(('http://', 'https://')):
                    link = self.base_url.rstrip('/') + '/' + link.lstrip('/')
                
                # Extract date if selector is provided
                pub_date = datetime.now()
                if self.date_selector:
                    date_element = element.select_one(self.date_selector)
                    if date_element:
                        date_text = date_element.get_text().strip()
                        try:
                            pub_date = parser.parse(date_text, fuzzy=True)
                        except:
                            # Keep default current date if parsing fails
                            pass
                
                # Skip articles older than cutoff date
                if pub_date < cutoff_date:
                    continue
                
                # Try to get article content/summary
                summary = ""
                if link:
                    try:
                        article_response = requests.get(link, timeout=10)
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        # Try to find main content
                        content_selectors = ['article', 'main', '.content', '.post-content', '.entry-content']
                        for selector in content_selectors:
                            content = article_soup.select_one(selector)
                            if content:
                                text = content.get_text()
                                summary = self.summarize_text(text)
                                break
                    except:
                        # If we can't fetch article content, use the title as summary
                        summary = title
                
                article = {
                    'title': title,
                    'link': link,
                    'published': pub_date.isoformat(),
                    'summary': summary,
                    'source': self.name,
                    'logo_url': self.logo_url
                }
                
                # Add trending score
                article['trending_score'] = self.calculate_trending_score(article)
                
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping website for {self.name}: {str(e)}")
            return []


# Custom ArXiv source that limits results directly
class ArXivRSSSource(RSSSource):
    """A specialized RSS source for ArXiv that limits the number of articles"""
    
    def __init__(self, name, rss_url, base_url, logo_url=None, max_articles=20):
        super().__init__(name, rss_url, base_url, logo_url)
        self.max_articles = max_articles
    
    def fetch_articles(self, days=30):
        """Fetch articles with a direct limit on the number of results"""
        logger.info(f"Fetching articles from ArXiv with limit: {self.max_articles}")
        
        try:
            # Try with timeout to avoid hanging
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml, application/atom+xml, text/html, */*'
            }
            
            # Custom URL for ArXiv to limit results
            # Note: ArXiv has a max_results parameter that can be used to limit results
            limited_url = f"{self.rss_url}?max_results={self.max_articles}"
            logger.info(f"Using limited ArXiv URL: {limited_url}")
            
            response = requests.get(limited_url, headers=headers, timeout=20)
            logger.info(f"{self.name}: RSS status code {response.status_code}")
            
            feed = feedparser.parse(response.content)
            logger.info(f"{self.name}: Found {len(feed.entries)} entries in feed")
            
            # Proceed with normal processing
            cutoff_date = datetime.now() - timedelta(days=days)
            articles = []
            
            for entry in feed.entries[:self.max_articles]:  # Additional safeguard
                title = entry.get('title', 'No Title')
                link = entry.get('link', '')
                
                if title == 'No Title' or not link:
                    continue
                
                # Parse the publication date
                try:
                    date_fields = ['published', 'updated', 'pubDate', 'created', 'date']
                    pub_date = None
                    
                    for field in date_fields:
                        if field in entry and entry[field]:
                            try:
                                pub_date = parser.parse(entry[field])
                                break
                            except:
                                continue
                    
                    if not pub_date:
                        pub_date = datetime.now()
                        
                    if pub_date.tzinfo:
                        pub_date = pub_date.astimezone(timezone.utc).replace(tzinfo=None)
                except:
                    pub_date = datetime.now()
                
                # Skip articles older than cutoff date
                if pub_date < cutoff_date:
                    continue
                
                # Extract content
                summary = ''
                content_options = [
                    lambda: entry.get('content', [{}])[0].get('value', ''),
                    lambda: entry.get('summary', ''),
                    lambda: entry.get('description', ''),
                    lambda: entry.get('summary_detail', {}).get('value', ''),
                    lambda: title
                ]
                
                for get_content in content_options:
                    try:
                        content = get_content()
                        if content:
                            soup = BeautifulSoup(content, 'html.parser')
                            text = soup.get_text()
                            if text.strip():
                                summary = self.summarize_text(text)
                                break
                    except:
                        continue
                
                if not summary:
                    summary = title
                
                article = {
                    'title': title,
                    'link': link,
                    'published': pub_date.isoformat(),
                    'summary': summary,
                    'source': self.name,
                    'logo_url': self.logo_url
                }
                
                article['trending_score'] = self.calculate_trending_score(article)
                articles.append(article)
            
            logger.info(f"{self.name}: Returning {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching ArXiv RSS feed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

# Function to print the entire content of RSS feed for debugging
def debug_rss_feed(name, feed_url):
    """Debug function to examine an RSS feed"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        response = requests.get(feed_url, headers=headers, timeout=15)
        logger.info(f"Debug {name}: HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            logger.info(f"Debug {name}: Content-Type: {content_type}")
            
            # Try to parse with feedparser
            import feedparser
            feed = feedparser.parse(response.text)
            
            if hasattr(feed, 'entries') and len(feed.entries) > 0:
                logger.info(f"Debug {name}: Found {len(feed.entries)} entries in feed")
                for i, entry in enumerate(feed.entries[:3]):  # Print first 3 entries
                    logger.info(f"Debug {name}: Entry {i+1} title: {entry.get('title', 'No title')}")
                    logger.info(f"Debug {name}: Entry {i+1} link: {entry.get('link', 'No link')}")
                    logger.info(f"Debug {name}: Entry {i+1} published: {entry.get('published', 'No date')}")
            else:
                logger.error(f"Debug {name}: No entries found in feed")
                
                # Try to parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                logger.info(f"Debug {name}: HTML Snippet: {soup.get_text()[:500]}")
    except Exception as e:
        logger.error(f"Debug {name}: Error inspecting feed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Initialize sources for top LLM providers
def get_sources():
    """Get all configured article sources"""
    logger.info("Initializing article sources")
    sources = [
        # Anthropic - Custom implementation for more reliable scraping
        WebScraperSource(
            name="Anthropic",
            blog_url="https://www.anthropic.com/blog",
            base_url="https://www.anthropic.com",
            article_selector=".posts-list article, .blog-posts-list article, article, div[class*='blog-post']",
            title_selector="h3, h2, .title, a[class*='heading']",
            link_selector="a",
            logo_url="https://www.anthropic.com/favicon.ico"
        ),
        
        # OpenAI - Using a more direct approach with their blog
        WebScraperSource(
            name="OpenAI",
            blog_url="https://openai.com/blog",  # Direct blog URL
            base_url="https://openai.com",
            article_selector="article, .post, .post-card, .post-item, a[href*='/blog/'], a[href*='/research/']",
            title_selector="h1, h2, h3, .post-title, .post-card-title",
            link_selector="a",
            logo_url="https://openai.com/favicon.ico"
        ),
        
        # Meta AI - Direct URL with simpler selectors
        WebScraperSource(
            name="Meta AI",
            blog_url="https://ai.meta.com/blog/",
            base_url="https://ai.meta.com",
            article_selector="article, .article, .collection-items--item, .card, li, div[class*='blog']",
            title_selector="h1, h2, h3, .title, span",
            link_selector="a",
            logo_url="https://ai.meta.com/favicon.ico"
        ),
        
        # Cohere - Using their main blog with better selectors
        WebScraperSource(
            name="Cohere",
            blog_url="https://cohere.com/blog",  # Updated to main blog URL
            base_url="https://cohere.com",
            article_selector="article, .post, .blog-post, .card, div[class*='blog'], div[class*='post']",
            title_selector="h1, h2, h3, .title",
            link_selector="a",
            logo_url="https://cohere.com/favicon.ico"
        ),
        
        # Mistral AI - Updated blog URL and selectors
        WebScraperSource(
            name="Mistral AI",
            blog_url="https://mistral.ai/blog/",  # Changed from news to blog
            base_url="https://mistral.ai",
            article_selector=".blog-item, article, .post, div[class*='blog'], div[class*='card'], li, .news-item",
            title_selector="h1, h2, h3, h4, .title, .heading",
            link_selector="a",
            logo_url="https://mistral.ai/favicon.ico"
        ),
        
        # NVIDIA AI
        WebScraperSource(
            name="NVIDIA AI",
            blog_url="https://blogs.nvidia.com/blog/category/deep-learning/",
            base_url="https://blogs.nvidia.com",
            article_selector=".blog-item, article, .post",
            title_selector="h2, .post-title, .blog-item-title",
            link_selector="a",
            logo_url="https://www.nvidia.com/favicon.ico"
        ),
        
        # HuggingFace - Reliable RSS source
        RSSSource(
            name="Hugging Face",
            rss_url="https://huggingface.co/blog/feed.xml",
            base_url="https://huggingface.co",
            logo_url="https://huggingface.co/favicon.ico"
        ),
        
        # Google DeepMind - Using a more reliable URL with broader selectors
        WebScraperSource(
            name="Google DeepMind",
            blog_url="https://deepmind.google/discover/blog/",  # More specific blog URL
            base_url="https://deepmind.google",
            article_selector="article, div[class*='card'], div[class*='blog'], div[class*='post'], li, a[href*='/blog/']",
            title_selector="h1, h2, h3, h4, .title, span",
            link_selector="a",
            logo_url="https://deepmind.google/favicon.ico"
        ),
        
        # Google AI Blog - Direct URL approach with broader selectors
        WebScraperSource(
            name="Google AI",
            blog_url="https://blog.google/technology/ai/",  # More reliable Google AI blog
            base_url="https://blog.google",
            article_selector="article, .post, .article, div[class*='card'], div[class*='post'], div[class*='blog'], li, .feed-item",
            title_selector="h1, h2, h3, h4, .title, span", 
            link_selector="a",
            logo_url="https://blog.google/favicon.ico"
        ),
        
        # Microsoft Research
        RSSSource(
            name="Microsoft Research",
            rss_url="https://www.microsoft.com/en-us/research/feed/",
            base_url="https://www.microsoft.com/en-us/research",
            logo_url="https://www.microsoft.com/favicon.ico"
        ),
        
        # ArXiv AI Papers - Using custom source class with built-in limiting
        ArXivRSSSource(
            name="ArXiv AI",
            rss_url="https://arxiv.org/rss/cs.AI",
            base_url="https://arxiv.org",
            logo_url="https://static.arxiv.org/static/base/0.22.2/images/icons/favicon.ico",
            max_articles=20
        ),
        
        # MIT Technology Review AI
        RSSSource(
            name="MIT Tech Review",
            rss_url="https://www.technologyreview.com/topic/artificial-intelligence/feed",
            base_url="https://www.technologyreview.com",
            logo_url="https://wp-preprod.technologyreview.com/wp-content/uploads/2022/06/favicon.png"
        ),
        
        # Analytics Vidhya - Using a more reliable scraping approach
        WebScraperSource(
            name="Analytics Vidhya",
            blog_url="https://www.analyticsvidhya.com/blog-archive/",  # Archive page has all posts
            base_url="https://www.analyticsvidhya.com",
            article_selector=".article-box, .article-card, article, .recent-articles-list li",
            title_selector="h3, h2, .title, .entry-title",
            link_selector="a",
            logo_url="https://www.analyticsvidhya.com/wp-content/uploads/2015/02/av_logo.png"
        ),
        
        # KDnuggets - Data Science and AI
        RSSSource(
            name="KDnuggets",
            rss_url="https://www.kdnuggets.com/feed",
            base_url="https://www.kdnuggets.com",
            logo_url="https://www.kdnuggets.com/wp-content/uploads/kdnuggets-logo-300x300.jpg"
        ),
        
        # Towards Data Science - Alternative feed URL
        RSSSource(
            name="Towards Data Science",
            rss_url="https://towardsdatascience.com/feed",
            base_url="https://towardsdatascience.com",
            logo_url="https://miro.medium.com/1*emiGsBgJu2KHWyjluhKXQw.png"
        ),
        
        # Add more reliable ML sources
        RSSSource(
            name="Neptune AI Blog",
            rss_url="https://neptune.ai/blog/feed",
            base_url="https://neptune.ai",
            logo_url="https://neptune.ai/wp-content/themes/neptune/assets/images/icons/apple-touch-icon.png"
        ),
        
        RSSSource(
            name="Lex Fridman",
            rss_url="https://lexfridman.com/feed/",
            base_url="https://lexfridman.com",
            logo_url="https://lexfridman.com/wordpress/wp-content/uploads/2023/02/cropped-lex-logo-black-192x192.png"
        ),
        
        WebScraperSource(
            name="DeepLearning.AI",
            blog_url="https://www.deeplearning.ai/blog/",
            base_url="https://www.deeplearning.ai",
            article_selector=".post, article, .blog-post",
            title_selector="h2, h3, .post-title",
            link_selector="a",
            logo_url="https://www.deeplearning.ai/favicon.ico"
        )
    ]
    
    return sources


def fetch_all_articles(days=30, keyword_filter=None, source_filter=None):
    """Fetch articles from all sources with optional filtering"""
    logger.info(f"Fetching all articles with days={days}, keyword={keyword_filter}, source={source_filter}")
    sources = get_sources()
    
    # Filter sources if source_filter is provided
    if source_filter:
        sources = [s for s in sources if s.name.lower() == source_filter.lower()]
        logger.info(f"Filtered to {len(sources)} sources matching '{source_filter}'")
    
    all_articles = []
    for source in sources:
        try:
            logger.info(f"Fetching from {source.name}")
            articles = source.fetch_articles(days)
            
            # Limit to max 20 articles per source
            if len(articles) > 20:
                logger.info(f"Limiting {source.name} from {len(articles)} to 20 articles")
                articles = articles[:20]
                
            logger.info(f"Got {len(articles)} articles from {source.name}")
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error fetching from {source.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Filter by keyword if provided
    if keyword_filter:
        keyword_lower = keyword_filter.lower()
        all_articles = [
            article for article in all_articles 
            if keyword_lower in article['title'].lower() or keyword_lower in article['summary'].lower()
        ]
    
    # Sort by trending score (descending)
    all_articles.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
    
    # Take top 25 or all if less than 25
    top_articles = all_articles[:min(25, len(all_articles))]
    
    return top_articles
