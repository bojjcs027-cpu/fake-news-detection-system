# 🚀 Fake News Detection System

This is a Machine Learning based Web Application that detects whether a given news article is **REAL** or **FAKE**.

## 🛠️ Tech Stack
- **Python**: Core programming language
- **Flask**: Web framework for building the UI
- **Scikit-Learn**: Machine Learning library used for building the model
- **Pandas**: Used for data manipulation
- **HTML & CSS**: For building a beautiful, modern User Interface

## 🌟 Features
- Beautiful glass-morphic UI with vibrant gradients
- Custom trained `Logistic Regression` model using TF-IDF Vectorizer
- Fast and accurate Fake News prediction
- Easy to set up and run locally

## 🚀 How to Run Locally

### 1. Install Requirements
Open your terminal and run:
`pip install -r requirements.txt`

### 2. Run the Application
Start the Flask server:
`python app.py`

### 3. Open in Browser
Visit the local server address:
`http://127.0.0.1:5000`

## 🌍 How to get a Public URL

If you want to share your project with others, you have two main options:

### 1. Temporary Public URL (Quick Sharing)
You can use **Localtunnel** to expose your local server to the internet without any deployment setup.
- While the Flask app is running, open a **new terminal** and run:
  `npx localtunnel --port 5000`
- This will provide a URL like `https://shiny-goats-jump.loca.lt` that anyone can visit.

### 2. Permanent Public URL (Deployment)
For a permanent URL, you can deploy your project to a cloud platform:
- **Render** (Recommended): Connect your GitHub repository to [Render.com](https://render.com), choose **Web Service**, and use the following settings:
  - **Runtime**: `Python`
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `gunicorn app:app`
- I have already prepared the `Procfile` and updated `requirements.txt` to support this!

---

## 📢 LinkedIn Post Template

Feel free to use this template to showcase your project on LinkedIn:

---

**🚀 Excited to share my latest machine learning project!**

I just built a **Fake News Detection System** using Machine Learning and Python. In today's digital age, fact-checking is more important than ever, so I decided to tackle this problem head-on! 📰🔍

**🛠️ How it works:**
The application uses **Natural Language Processing (NLP)** to analyze the textual content of a news article. I trained a **Logistic Regression** model using **TfidfVectorizer** from `scikit-learn` to classify news as **REAL ✅** or **FAKE 🚨**. 

I also built a sleek, responsive web interface using **Flask**, **HTML**, and **Vanilla CSS** to make testing the model easy and intuitive for users.

**💡 Tech Stack Used:**
- Python 🐍
- Scikit-Learn (Machine Learning) 🤖
- Pandas (Data Processing) 📊
- Flask (Web Framework) 🌐
- HTML & Modern CSS for a clean UI ✨

You can check out the source code on my GitHub. I'm actively looking for feedback and suggestions for improvement! 👇

🔗 GitHub Link: [Insert your GitHub repo link here]

#MachineLearning #Python #DataScience #ArtificialIntelligence #NLP #WebDevelopment #Flask #TechProjects #OpenSource

---

## 💾 Dataset
This project includes a sample `news.csv` to ensure everything runs perfectly out of the box. For higher accuracy, you can replace the `news.csv` file with a larger dataset from Kaggle (like the *Fake and real news dataset*).
