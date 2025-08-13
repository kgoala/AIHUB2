import feedparser

def get_news_from_sources():
    # Dictionary of news sources and their RSS URLs
    sources = {
        "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "Reuters": "http://feeds.reuters.com/reuters/topNews",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "NPR": "https://feeds.npr.org/1001/rss.xml"
    }
    
    all_articles = []
    
    for source, url in sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:  # Limit to 5 articles per source
            article = {
                'source': source,
                'title': entry.title,
                'link': entry.link,
                'published': getattr(entry, 'published', 'No date available')
            }
            all_articles.append(article)
    
    # Optional: sort articles by published date descending if dates exist
    # Commented out because some feeds have inconsistent date formats
    # all_articles.sort(key=lambda x: x['published'], reverse=True)
    
    return all_articles

def main():
    articles = get_news_from_sources()
    print("\n" + "="*80)
    print("📰 DAILY GLOBAL NEWS AGGREGATION")
    print("="*80 + "\n")
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. [{article['source']}] {article['title']}")
        print(f"   Published: {article['published']}")
        print(f"   Link: {article['link']}\n")

if __name__ == "__main__":
    main()
