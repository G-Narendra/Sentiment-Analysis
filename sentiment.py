import pandas as pd

# Load the datasets
training_data_path = 'twitter_training.csv'
validation_data_path = 'twitter_validation.csv'

# Read the CSV files
training_data = pd.read_csv(training_data_path, header=None)
validation_data = pd.read_csv(validation_data_path, header=None)

# Display the first few rows of each dataset
print(training_data.head())
print(validation_data.head())


import string

# Remove duplicates and handle missing values
training_data_cleaned = training_data.drop_duplicates().dropna()
validation_data_cleaned = validation_data.drop_duplicates().dropna()

# Standardize text: lowercase and remove punctuation
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

training_data_cleaned[3] = training_data_cleaned[3].apply(clean_text)
validation_data_cleaned[3] = validation_data_cleaned[3].apply(clean_text)

# Check the cleaned data
print(training_data_cleaned.head())
print(validation_data_cleaned.head())


# Define the label encoding
label_mapping = {
    'Positive': 2,
    'Neutral': 1,
    'Negative': 0,
    'Irrelevant': 3
}

# Encode the sentiment labels
training_data_cleaned['label'] = training_data_cleaned[2].map(label_mapping)
validation_data_cleaned['label'] = validation_data_cleaned[2].map(label_mapping)

# Split the data into features (tweets) and labels (sentiments)
X_train = training_data_cleaned[3]
y_train = training_data_cleaned['label']
X_val = validation_data_cleaned[3]
y_val = validation_data_cleaned['label']

# Check the data splits
print(X_train.head(), y_train.head())
print(X_val.head(), y_val.head())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Preprocessing: TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Create pipelines for different models
pipelines = {
    'Logistic Regression': Pipeline([
        ('tfidf', tfidf),
        ('scaler', StandardScaler(with_mean=False)),  # Scaling the data
        ('clf', LogisticRegression(max_iter=500, solver='lbfgs'))
    ]),
    'SVM': Pipeline([
        ('tfidf', tfidf),
        ('clf', SVC())
    ]),
    'Random Forest': Pipeline([
        ('tfidf', tfidf),
        ('clf', RandomForestClassifier())
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', tfidf),
        ('clf', MultinomialNB())
    ]),
}

# Train and evaluate each model
for model_name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_val, y_val)
    print(f'{model_name} Accuracy: {accuracy}')


from sklearn.metrics import classification_report

# Generate evaluation reports for each model
for model_name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_val)
    print(f'--- {model_name} ---')
    print(classification_report(y_val, y_pred, target_names=label_mapping.keys()))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Plot confusion matrices for each model
for model_name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
