# 🤖 NLPBot — AI Chatbot | IncoVision Internship Task 01

A Python AI chatbot available in both the terminal and a responsive browser interface. The browser experience features a friendly CSS-animated 3D mascot and live chat, while both interfaces share the same NLP intent matcher. It demonstrates tokenization, stopword removal, TF-IDF vectorization, and cosine similarity—without paid APIs or API keys. NLTK stopwords are downloaded automatically on first use when needed; a built-in stopword list is used if they cannot be fetched.

---

## 📋 Project Description

NLPBot is a rule-based chatbot that uses NLP techniques to understand user input and respond intelligently — without paid APIs or API keys. After NLTK stopwords are cached (or the built-in list is used), matching runs locally and offline. It covers 25+ intents including greetings, AI/NLP explanations, Python basics, jokes, and dynamic time/date responses.

---

## 📁 Project Structure

```
nlp-chatbot/
├── chatbot.py        # Main program and chat loop
├── app.py            # Local Flask server and chat API
├── preprocess.py     # Text cleaning and NLP preprocessing functions
├── intents.json      # Knowledge base with 25+ intents
├── templates/         # Browser chat interface
├── static/            # Responsive styling and chat interactions
├── test_chatbot.py   # Automated preprocessing and matching tests
├── chatbot_test.py   # Alternate test runner
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── .gitignore        # Python gitignore
```

---

## 🧠 How It Works

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│        Preprocessing            │
│  lowercase → remove punctuation │
│  → tokenize → remove stopwords  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│      TF-IDF Vectorization       │
│  Convert text to numeric vectors│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│      Cosine Similarity          │
│  Compare input vs all patterns  │
│  Pick highest scoring intent    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│      Response Selection         │
│  Random pick from intent's      │
│  responses list                 │
└─────────────────────────────────┘
    │
    ▼
  Bot Reply
```

---

## 📚 NLP Concepts Explained

### 1. Tokenization
Tokenization splits raw text into individual units called **tokens** (usually words).

```
Input:  "What is Natural Language Processing?"
Output: ["What", "is", "Natural", "Language", "Processing"]
```

This is the first step in any NLP pipeline — the computer needs words as discrete units before it can analyze them.

---

### 2. Stopword Removal
**Stopwords** are common words like *"the"*, *"is"*, *"and"*, *"a"* that appear frequently but carry little meaning.

```
Before: ["what", "is", "natural", "language", "processing"]
After:  ["natural", "language", "processing"]
```

Removing them reduces noise and helps the model focus on meaningful keywords.

---

### 3. TF-IDF (Term Frequency–Inverse Document Frequency)
TF-IDF converts text into numerical vectors that reflect how **important** each word is.

- **TF (Term Frequency)**: How often a word appears in a document.
- **IDF (Inverse Document Frequency)**: Penalizes words that appear in many documents (common words get lower scores).

```
TF-IDF(word) = TF(word) × IDF(word)
```

Words that are unique and frequent in a specific document get high TF-IDF scores — making them great identifiers.

---

### 4. Cosine Similarity
Cosine similarity measures the **angle** between two text vectors.

```
similarity = cos(θ) = (A · B) / (|A| × |B|)
```

- Score of **1.0** = identical meaning
- Score of **0.0** = completely different

The chatbot computes cosine similarity between the user's input vector and every stored pattern vector, then picks the intent with the highest score when it meets a confidence threshold. A fuzzy string comparison can recover simple typos; low-confidence inputs receive a polite fallback.

---

## ⚙️ Setup & Installation

### Step 1 — Clone or download the project
```bash
git clone https://github.com/YOUR_USERNAME/nlp-chatbot.git
cd nlp-chatbot
```

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the chatbot
```bash
python chatbot.py
```

### Browser interface
Install dependencies as above, then start the local web app:
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The page includes an animated 3D-style CSS mascot and a responsive chat interface backed by the same local intent matcher.

### Step 5 — Run the tests
```bash
python -m unittest -v
```

### Step 6 — Exit the chatbot
Type any of: `exit`, `quit`, `bye`, or press `Ctrl+C`

---

## 💬 Sample Conversation

```
  _   _ _     ____  ____        _
 | \ | | |   |  _ \| __ )  ___ | |_
 |  \| | |   | |_) |  _ \ / _ \| __|
 | |\  | |___|  __/| |_) | (_) | |_
 |_| \_|_____|_|   |____/ \___/ \__|

  🤖  NLPBot — AI Chatbot for NLP Internship
  Built with Python, NLTK, scikit-learn & colorama
  ──────────────────────────────────────────────────
  ✅  Loaded 25 intents from intents.json
  ✅  TF-IDF index built successfully
  ──────────────────────────────────────────────────

  Bot: Hello! I'm NLPBot. How can I help you today?

  You: hi there
  Bot: Hey! Great to see you. What's on your mind?

  You: what is nlp?
  Bot: Natural Language Processing (NLP) is a branch of AI that helps
       computers understand, interpret, and generate human language.

  You: tell me a joke
  Bot: Why do programmers prefer dark mode? Because light attracts bugs! 🐛

  You: what time is it?
  Bot: The current time is 02:45 PM.

  You: what is tfidf?
  Bot: TF-IDF stands for Term Frequency-Inverse Document Frequency. It
       measures how important a word is in a document relative to a
       collection of documents.

  You: bye
  Bot: Goodbye! It was great chatting with you. 👋

  Thanks for using NLPBot! #incodevision
```

---

## 📸 Screenshots

> _Add screenshots of your terminal here after running the chatbot._

| Welcome Banner | Sample Chat |
|---|---|
| _(screenshot)_ | _(screenshot)_ |

---

## 🚀 Push to GitHub

```bash
# 1. Initialize git repository
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "feat: Add NLPBot - Task 01 IncoVision AI Internship"

# 4. Create main branch
git branch -M main

# 5. Add remote origin (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/nlp-chatbot.git

# 6. Push to GitHub
git push -u origin main
```

---

## 🔮 Future Improvements

- [ ] **Deep Learning**: Replace TF-IDF with a trained neural network (LSTM or BERT)
- [ ] **Context Awareness**: Remember previous messages in the conversation
- [ ] **Voice Input/Output**: Add speech recognition and text-to-speech
- [ ] **Web Interface**: Build a Flask or FastAPI web frontend
- [ ] **Database**: Store conversation history in SQLite or MongoDB
- [ ] **Sentiment Analysis**: Detect user mood and adjust responses
- [ ] **Multi-language Support**: Add support for Arabic, French, Spanish
- [ ] **Live API Integration**: Connect to weather, news, or Wikipedia APIs
- [ ] **Entity Recognition**: Extract names, dates, and locations from input
- [ ] **Confidence Display**: Show similarity scores alongside responses

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| NLTK | Tokenization & stopwords |
| scikit-learn | TF-IDF vectorization & cosine similarity |
| colorama | Colored terminal output |
| NumPy | Numerical operations |
| difflib | Fuzzy string matching (built-in) |
| json | Intent storage |
| datetime | Dynamic time/date responses |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built as **Task 01** of the AI/NLP Internship at **IncoVision**.

#incodevision
