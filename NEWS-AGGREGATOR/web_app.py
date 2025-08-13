from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_apscheduler import APScheduler
from flask_socketio import SocketIO
import feedparser, requests, hashlib, threading
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from dateutil import parser as date_parser

# Flask setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# AI models
nlp = spacy.load('en_core_web_sm')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

IST = timezone(timedelta(hours=5, minutes=30))

# Global state
cached = []
vectors = []
hashes = set()
last_update = datetime.now(IST)

# Expanded strictly 24h-only sources
SOURCES = {
  # APJ
  "CNN Asia":"http://rss.cnn.com/rss/edition_asia.rss",
  "BBC Asia":"http://feeds.bbci.co.uk/news/world/asia/rss.xml",
  "Japan Times":"https://www.japantimes.co.jp/feed/topstories/",
  "South China Morning Post":"https://www.scmp.com/rss/91/feed",
  "The Straits Times":"https://www.straitstimes.com/news/world/rss.xml",
  "Channel News Asia":"https://www.channelnewsasia.com/rssfeeds/8395986",
  "ABC News Australia":"https://www.abc.net.au/news/feed/51120/rss.xml",
  "Korea Herald":"http://www.koreaherald.com/common/rss_xml.php?ct=020000000",
  "Bangkok Post":"https://www.bangkokpost.com/rss/data/breaking.xml",
  "Jakarta Post":"https://www.thejakartapost.com/rss",
  # India
  "Times of India":"https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
  "The Hindu":"https://www.thehindu.com/news/national/feeder/default.rss",
  "NDTV":"https://feeds.feedburner.com/ndtvnews-top-stories",
  "India Today":"https://www.indiatoday.in/rss/home",
  "Business Standard":"https://www.business-standard.com/rss/home_page_top_stories.rss",
  "Hindustan Times":"https://www.hindustantimes.com/feeds/rss/top-news/rssfeed.xml",
  "News18":"https://www.news18.com/rss/india.xml",
  "The Wire":"https://thewire.in/feed",
  "Scroll":"https://scroll.in/feed",
  "DNA India":"https://www.dnaindia.com/feeds/india.xml",
  # EMEA
  "BBC World":"http://feeds.bbci.co.uk/news/world/rss.xml",
  "BBC Europe":"http://feeds.bbci.co.uk/news/world/europe/rss.xml",
  "Reuters Top News":"http://feeds.reuters.com/reuters/topNews",
  "Guardian World":"https://www.theguardian.com/world/rss",
  "Guardian UK":"https://www.theguardian.com/uk-news/rss",
  "CNN International":"http://rss.cnn.com/rss/edition_world.rss",
  "DW News":"https://rss.dw.com/rdf/rss-en-all",
  "France 24":"https://www.france24.com/en/rss",
  "Al Jazeera":"https://www.aljazeera.com/xml/rss/all.xml",
  "Euronews":"https://feeds.feedburner.com/euronews/en/home/",
  "Sky News":"http://feeds.skynews.com/feeds/rss/world.xml",
  "Independent":"https://www.independent.co.uk/rss",
  "Irish Times":"https://www.irishtimes.com/cmlink/news/rss.xml",
  "Deutsche Welle":"https://rss.dw.com/rdf/rss-en-all",
  "AfricaNews":"https://www.africanews.com/api/en/rss",
  "Middle East Eye":"https://www.middleeasteye.net/rss.xml",
  "Times of Israel":"https://www.timesofisrael.com/feed/",
  "Arab News":"https://www.arabnews.com/taxonomy/term/4926/feed",
}

REGION = {
  **{k:"APJ" for k in ["CNN Asia","BBC Asia","Japan Times","South China Morning Post","The Straits Times","Channel News Asia","ABC News Australia","Korea Herald","Bangkok Post","Jakarta Post"]},
  **{k:"India" for k in ["Times of India","The Hindu","NDTV","India Today","Business Standard","Hindustan Times","News18","The Wire","Scroll","DNA India"]},
  **{k:"EMEA" for k in ["BBC World","BBC Europe","Reuters Top News","Guardian World","Guardian UK","CNN International","DW News","France 24","Al Jazeera","Euronews","Sky News","Independent","Irish Times","Deutsche Welle","AfricaNews","Middle East Eye","Times of Israel","Arab News"]}
}

def parse_date(entry):
    for f in ('published','updated'):
        raw = entry.get(f)
        if raw:
            try:
                dt = date_parser.parse(raw)
                dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except: pass
    return None

def summarize(text,n=2):
    try:
        p=PlaintextParser.from_string(text,Tokenizer("english"))
        s=LsaSummarizer()(p.document,n)
        return " ".join(str(x) for x in s)
    except: return ""

def extract_kw(text):
    return [ent.text for ent in nlp(text).ents]

def fetch_and_cache():
    global cached,vectors,hashes,last_update
    now_utc=datetime.now(timezone.utc)
    new=[]
    for src,url in SOURCES.items():
        try:
            r=requests.get(url,timeout=10); r.raise_for_status()
            for e in feedparser.parse(r.content).entries:
                dt=parse_date(e)
                if not dt or (now_utc-dt).total_seconds()>86400: continue
                h=hashlib.md5((e.title+e.link).encode()).hexdigest()
                if h in hashes: continue
                hashes.add(h)
                txt=e.get('summary','') or e.get('description','') or ''
                new.append({
                  "region":REGION[src],"source":src,
                  "title":e.title,"link":e.link,
                  "published":dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                  "timestamp":dt.timestamp(),
                  "thumbnail":(e.get('media_thumbnail') or e.get('media_content') or [{}])[0].get('url',''),
                  "summary":summarize(txt),"keywords":extract_kw(e.title+" "+txt)
                })
        except: pass
    all_=sorted(new+cached,key=lambda x:x['timestamp'],reverse=True)[:100]
    cached=all_
    texts=[a['title']+a['summary'] for a in cached]
    vectors=embed_model.encode(texts,convert_to_tensor=False)
    last_update=datetime.now(IST)
    socketio.emit('update',{"count":len(new),"time":last_update.strftime("%Y-%m-%d %I:%M %p IST")})

scheduler.add_job(id='update_job',func=fetch_and_cache,trigger='interval',minutes=3)
fetch_and_cache()

@app.route('/')
def home():
    grouped=defaultdict(list)
    for a in cached: grouped[a['region']].append(a)
    for lst in grouped.values():
        for art in lst:
            dt=datetime.strptime(art['published'],"%Y-%m-%d %H:%M:%S UTC")
            art['published_ist']=dt.replace(tzinfo=timezone.utc).astimezone(IST).strftime("%Y-%m-%d %I:%M %p IST")
    return render_template('index.html',regions=grouped,query=None,last=last_update.strftime("%Y-%m-%d %I:%M %p IST"))

@app.route('/search')
def search():
    q=request.args.get('q','').strip()
    if not q: return redirect(url_for('home'))
    qv=vectors and embed_model.encode([q],convert_to_tensor=False)[0]
    sims=[(np.dot(qv,v)/(np.linalg.norm(qv)*np.linalg.norm(v)),a) for v,a in zip(vectors,cached)]
    tops=[a for _,a in sorted(sims,reverse=True)[:50]]
    grouped=defaultdict(list)
    for a in tops: grouped[a['region']].append(a)
    for lst in grouped.values():
        for art in lst:
            dt=datetime.strptime(art['published'],"%Y-%m-%d %H:%M:%S UTC")
            art['published_ist']=dt.replace(tzinfo=timezone.utc).astimezone(IST).strftime("%Y-%m-%d %I:%M %p IST")
    return render_template('index.html',regions=grouped,query=q,last=last_update.strftime("%Y-%m-%d %I:%M %p IST"))

@app.route('/refresh')
def refresh():
    threading.Thread(target=fetch_and_cache,daemon=True).start()
    return redirect(url_for('home'))

@app.route('/keywords')
def keywords():
    kws=set()
    for a in cached: kws.update(a['keywords'])
    return jsonify(sorted(kws))

if __name__=="__main__":
    socketio.run(app,debug=True,port=5000)
