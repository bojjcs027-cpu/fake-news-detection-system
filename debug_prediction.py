from predict import predict_news


def debug():
    """
    Debug script to test individual news predictions.
    """
    test_text = (
        "The US president said the air campaign could make negotiations a "
        "moot point if all potential leaders of Iran are killed and the "
        "Iranian military is destroyed."
    )

    try:
        print(f"Testing prediction for: {test_text[:100]}...")
        result = predict_news(test_text)
        print(f"Prediction result: {result}")
    except Exception:
        import traceback
        print("Caught exception in predict_news:")
        traceback.print_exc()


if __name__ == "__main__":
    debug()
