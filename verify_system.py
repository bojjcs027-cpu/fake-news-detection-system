import traceback
from app import research_news
from predict import predict_news


def test_system():
    """
    Automated test suite for Phase 6 Fake News Detection System.
    Validates prediction pipeline and global research API integration.
    """
    test_cases = [
        "Global Tech Summit Announces New AI Regulations",
        (
            "The US president said the air campaign could make negotiations a "
            "moot point if all potential leaders of Iran are killed and the "
            "Iranian military is destroyed."
        )
    ]
    
    for text in test_cases:
        print(f"\n--- Testing: {text[:50]}... ---")
        try:
            # 1. Test Prediction
            pred = predict_news(text)
            print(f"Prediction: {pred}")
            
            # 2. Test Research
            research = research_news(text)
            print(f"Research Hits: {len(research)}")
            for r in research:
                print(f" - {r['source']}: {r['title']}")
                
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    test_system()
