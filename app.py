from flask import Flask, render_template, request
import requests
import traceback
from predict import predict_news

app = Flask(__name__)
NEWS_API_KEY = "53b5c5ab8829400c9f184ff6bf69afb7"


def research_news(text):
    """
    Research the news across the world using NewsAPI.
    Returns a list of matching reputable articles or an empty list.
    """
    # Take first 10-12 words for better search results
    search_query = " ".join(text.split()[:12])
    url = (
        f"https://newsapi.org/v2/everything?q={search_query}"
        f"&apiKey={NEWS_API_KEY}&language=en&sortBy=relevancy"
    )
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return [
                {
                    "title": a['title'],
                    "url": a['url'],
                    "source": a['source']['name']
                } for a in articles[:3] # Return top 3 reputable matches
            ]
    except Exception as e:
        print(f"Research API Error: {e}")
    return []


@app.route("/", methods=["GET", "POST"])
def home():
    """
    Main route for the web application. Handles news analysis requests.
    """
    result = None
    news_text = ""
    
    if request.method == "POST":
        # Get the input text from the HTML form
        news_text = request.form.get("news", "")
        
        if news_text.strip():
            try:
                # Get prediction using our trained ML model
                prediction = predict_news(news_text)

                # Format the prediction result for premium UI
                if isinstance(prediction, str) and prediction.upper() == "FAKE":
                    result = {
                        "status": "fake", 
                        "label": "PROBABLE FAKE",
                        "icon": "🚨",
                        "message": "Warning: This content shows strong indicators of misinformation."
                    }
                elif isinstance(prediction, str) and prediction.upper() == "REAL":
                    result = {
                        "status": "real", 
                        "label": "AUTHENTIC NEWS",
                        "icon": "✅",
                        "message": "This article aligns with typical journalistic standards of reliability."
                    }
                else:
                    # In case of partial errors or calibration
                    error_msg = str(prediction) if "ERROR" in str(prediction) else "The AI is currently calibrating Phase 5 features."
                    result = {
                        "status": "error",
                        "label": "MODEL CALIBRATING",
                        "icon": "⚙️",
                        "message": (
                            f"Technical Update: {error_msg}. "
                            "Please try again in 10 seconds."
                        )
                    }

                # Add Global Research Verification
                print(f"Researching news: {news_text[:50]}...")
                research_results = research_news(news_text)
                result["research"] = research_results
                
                if research_results:
                    result["research_status"] = "VERIFIED BY GLOBAL SOURCES"
                else:
                    result["research_status"] = "NO MATCHING GLOBAL REPORTS FOUND"

            except Exception as e:
                # Full debug for Phase 6 errors
                print(f"CRITICAL SYSTEM ERROR: {e}")
                traceback.print_exc() 
                result = {
                    "status": "error",
                    "label": "SYSTEM REFRESH",
                    "icon": "🔄",
                    "message": (
                        f"Internal trace: {str(e)}. We're optimizing the "
                        "detector precision. Please retry."
                    )
                }

    # Fetch Live News from NewsAPI
    news_url = (
        f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    )
    
    live_news = []
    try:
        response = requests.get(news_url, timeout=3)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            # Get top 5 articles with actual content
            for article in articles:
                title = article.get('title')
                desc = article.get('description')
                if title and desc and title != '[Removed]':
                    live_news.append({
                        "title": article['title'],
                        "description": article['description'],
                        "source": article.get('source', {}).get('name', 'Unknown')
                    })
                if len(live_news) >= 5:
                    break
        else:
             raise Exception(f"API Error {response.status_code}")
    except Exception as e:
        print(f"Failed to fetch live news: {e}")
        live_news = [
            {
                "title": "Global Tech Summit Announces New AI Regulations",
                "description": (
                    "Leaders from top tech companies gathered today to "
                    "discuss the future of artificial intelligence."
                ),
                "source": "TechCrunch"
            },
            {
                "title": "New Study Shows Coffee Might Actually Be Good For You",
                "description": (
                    "A comprehensive 10-year study reveals surprising "
                    "health benefits of daily coffee consumption."
                ),
                "source": "HealthLine"
            },
            {
                "title": "Local High School Team Wins Robotics Championship",
                "description": (
                    "The 'Tech Titans' took home the gold medal after an "
                    "intense final match against the defending champions."
                ),
                "source": "City Post"
            },
            {
                "title": "Scientists Discover New Species of Glowing Octopus",
                "description": (
                    "Marine biologists were stunned to find a completely "
                    "new bioluminescent species living at extreme depths."
                ),
                "source": "NatGeo"
            }
        ]

    # Render template with variables
    return render_template("index.html", result=result, news=news_text, live_news=live_news)

if __name__ == "__main__":
    print("Starting Flask web server...")
    app.run(debug=True)
