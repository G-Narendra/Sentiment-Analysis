# 💬 Twitter Sentiment Analysis
### Multi-Class Tweet Classification using Machine Learning

<p align="center">
<img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/NLP-TF--IDF-8E44AD?style=for-the-badge">
<img src="https://img.shields.io/badge/Best%20Accuracy-74.5%25-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Library-Pandas-150458?style=for-the-badge&logo=pandas">
</p>

---

## 🌟 Overview

This project explores the application of **Natural Language Processing (NLP)** and **Supervised Machine Learning** to categorize the emotional tone of Twitter data. By analyzing the linguistic patterns of tweets, the system classifies content into four distinct categories: **Positive, Neutral, Negative, or Irrelevant**.



---

## 📊 Dataset Overview

The analysis is performed on a comprehensive dataset consisting of labeled tweets mapped to specific topics (e.g., Facebook, Amazon, Microsoft).

* **Training Data:** `twitter_training.csv`
* **Validation Data:** `twitter_validation.csv`
* **Labels:**
    * **0:** Negative
    * **1:** Neutral
    * **2:** Positive
    * **3:** Irrelevant

---

## 🎯 Project Workflow

1.  **Data Cleaning & Preprocessing:** Handling null values, removing duplicates, and normalizing text (lowercasing, punctuation removal).
2.  **Feature Engineering:** Utilizing **TF-IDF Vectorization** to convert raw text into numerical feature matrices.
3.  **Model Training:** Evaluating four distinct classification architectures.
4.  **Performance Visualization:** Generating **Confusion Matrices** and accuracy comparison plots.
5.  **Selection:** Fine-tuning the best-performing model for final deployment.



---

## 🧠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **NLP** | TF-IDF Vectorization |

---

## 📁 Project Structure

```bash
Sentiment-Analysis/
├── src/
│   └── sentiment.py         # Main ML implementation script
├── twitter_training.csv     # Training dataset
├── twitter_validation.csv   # Validation dataset
├── reports/
│   ├── sentiment_report.txt # Detailed performance analysis
│   └── confusion_matrix.png # Visual evaluation
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```



## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/G-Narendra/Sentiment-Analysis.git](https://github.com/G-Narendra/Sentiment-Analysis.git)
cd Sentiment-Analysis

```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```

### 3️⃣ Run the Analysis

```bash
python sentiment.py

```

---

## 📉 Model Performance & Evaluation

We evaluated multiple algorithms to find the most robust classifier for short-form social media text:

| Model | Accuracy Score |
| --- | --- |
| **Logistic Regression** | **0.745 (74.5%)** |
| **Random Forest Classifier** | 0.730 (73.0%) |
| **Support Vector Machine (SVM)** | 0.720 (72.0%) |
| **Naive Bayes** | 0.715 (71.5%) |

### **🏆 Champion Model: Logistic Regression**

The **Logistic Regression** model achieved the highest accuracy, demonstrating superior capability in handling high-dimensional TF-IDF sparse matrices for sentiment tasks.

---

## 🚀 Future Roadmap

* [ ] **Deep Learning:** Implementing LSTMs or GRUs to capture sequential dependencies in text.
* [ ] **Transformer Models:** Integrating **BERT** or **RoBERTa** for context-aware embeddings.
* [ ] **Real-time Pipeline:** Connecting to the Twitter (X) API for live sentiment tracking.

---

## Engineering Decisions & Challenges Solved

| Challenge | Decision | Why |
|---|---|---|
| Text data is high-dimensional and sparse | TF-IDF vectorization with max features limit | Bag-of-words produces 10K+ features from a small dataset — TF-IDF with max features reduces dimensionality while preserving signal |
| Negative sentiment is harder to detect than positive | Class-weighted model + precision/recall reporting | "Not good" vs "good" — negation flips sentiment; class weighting ensures the model learns to detect both |
| Simple vs complex model trade-off | Compare Naive Bayes, Logistic Regression, SVM | For small datasets, simple models often outperform complex ones — comparison justifies the choice |
| Model needs to work on unseen text | Hold-out test set evaluation with classification report | A model that performs well on training but fails on new text is useless — test set simulates real deployment |

## 👨‍💻 Author

**Narendra (G‑Narendra)** AI | ML | Python | Full Stack | GenAI Enthusiast

📧 [Email Me](mailto:narendragandikota2540@gmail.com) | 💼 [LinkedIn](https://linkedin.com/in/g-narendra/) | 👨‍💻 [GitHub](https://github.com/G-Narendra)

---

<p align="center">⭐ If you find this project helpful, please give it a star! 🚀</p>
