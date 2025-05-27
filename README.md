# AI Topic Aggregator

A Streamlit application that fetches and displays trending articles from top LLM providers.

## Features

- Fetches articles from top 10 LLM providers
- Displays top 25 trending articles from the last 30 days
- Filters by provider or keyword
- Ranks articles based on social shares and engagement
- Clean UI showing title, summary, source, and link

## Supported Sources

- OpenAI
- Google DeepMind
- Amazon
- Microsoft
- Anthropic
- Meta AI (LLaMA)
- Hugging Face
- Nvidia
- Cohere
- Mistral AI

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the application:
```
streamlit run app.py
```

## Technical Implementation

- Uses RSS feeds and official blog APIs where available
- Web scraping as fallback for sources without RSS/API
- Simple article summarization
- Social media trend analysis for article ranking
