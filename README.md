# **💬 Twitter Sentiment Analysis using Machine Learning**

*A machine learning-based approach to classify tweet sentiments.*

## 🌟 **Overview**
This project develops and evaluates **machine learning models** for **Twitter sentiment analysis**, categorizing tweets as **Positive, Neutral, Negative, or Irrelevant**. The dataset contains labeled tweets for training and validation.

## 📊 **Dataset Overview**
- **Training Dataset:** `twitter_training.csv`
- **Validation Dataset:** `twitter_validation.csv`
- **Features:**
  - **ID**: Unique identifier for each tweet
  - **Topic**: The subject of the tweet (e.g., Facebook, Amazon)
  - **Sentiment**: Labeled as **Positive (2), Neutral (1), Negative (0), or Irrelevant (3)**
  - **Tweet Text**: The actual content of the tweet

## 🎯 **Project Workflow**
✅ **Data Cleaning & Preprocessing** – Handling missing values, duplicate removal, and text normalization.  
✅ **Feature Engineering** – Encoding sentiment labels and vectorizing text with **TF-IDF**.  
✅ **Model Training & Evaluation** – Comparing multiple classification models.  
✅ **Performance Visualization** – Confusion matrices and accuracy plots.  
✅ **Best Model Selection** – Identifying the most accurate sentiment classifier.  

## 🛠️ **Tech Stack**
🔹 **Programming Language:** Python  
🔹 **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
🔹 **Model Type:** Classification (Logistic Regression, SVM, Random Forest, Naive Bayes)  
🔹 **Development Environment:** Jupyter Notebook / Python Script  

## 📂 **Project Structure**
```
Sentiment-Analysis/
├── sentiment.py               # Python script with model implementation
├── twitter_training.csv       # Training dataset
├── twitter_validation.csv     # Validation dataset
├── sentiment_intro.txt        # Dataset overview
├── sentiment_report.txt       # Detailed project report
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project documentation
```

## 🚀 **Installation & Setup**
1️⃣ **Clone the Repository**  
```sh
git clone https://github.com/G-Narendra/Sentiment-Analysis.git
cd Sentiment-Analysis
```
2️⃣ **Install Dependencies**  
```sh
pip install -r requirements.txt
```
3️⃣ **Run the Sentiment Analysis Model**  
```sh
python sentiment.py
```

## 📉 **Model Performance & Evaluation**
Four classification models were trained and evaluated:

| Model                 | Accuracy |
|----------------------|----------|
| **Logistic Regression** | **0.745** |
| **Support Vector Machine (SVM)** | 0.720 |
| **Random Forest Classifier** | 0.730 |
| **Naive Bayes** | 0.715 |

### **Best Performing Model: Logistic Regression**
The **Logistic Regression** model outperformed others with **the highest accuracy (74.5%)**, making it the most effective for predicting tweet sentiment.

## 📊 **Evaluation Metrics & Visualization**
- **Classification Reports** – Precision, recall, and F1-score analysis.
- **Confusion Matrices** – Visualizing model performance across sentiment classes.

## 💡 **Conclusion**
This project successfully developed **machine learning models** for **Twitter sentiment analysis**. The **Logistic Regression model** achieved the highest accuracy. Future improvements could involve:
- **Using deep learning models (LSTMs, Transformers) for better text understanding**.
- **Applying advanced NLP techniques like Word2Vec and BERT embeddings**.

## 🤝 **Contributions**
💡 Open to improvements! Feel free to:
1. Fork the repo  
2. Create a new branch (`feature-branch`)  
3. Make changes & submit a PR  



## 📩 **Connect with Me**
📧 **Email:** [narendragandikota2540@gmail.com](mailto:narendragandikota2540@gmail.com)  
🌐 **Portfolio:** [G-Narendra Portfolio](https://g-narendra-portfolio.vercel.app/)  
💼 **LinkedIn:** [G-Narendra](https://linkedin.com/in/g-narendra/)  
👨‍💻 **GitHub:** [G-Narendra](https://github.com/G-Narendra)  

⭐ **If you find this project useful, drop a star!** 🚀

