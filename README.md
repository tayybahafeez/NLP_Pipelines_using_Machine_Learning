# NLP Text Classification Pipeline
This project implements a complete Natural Language Processing (NLP) pipeline for text classification, including preprocessing, TF-IDF vectorization, and multiple machine learning models.
## Clone the repository
git clone https://github.com/tayybahafeez/NLP_Pipelines_using_Machine_Learning.git
cd NLP_Pipelines_using_Machine_Learning

## Install required packages
pip install -r requirements.txt

## Required packages:
```bash
nltk>=3.6.0
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0
```
### Basic Usage

from nlp_pipeline import NLPPipeline

### Prepare your dataset
```bash
texts = ["Your first text", "Your second text", ...]
labels = [0, 1, ...]  # Your labels

# Initialize and run pipeline
pipeline = NLPPipeline(max_features=5000)
pipeline.prepare_data(texts, labels)
results = pipeline.train_and_evaluate()

# Make predictions on new texts
new_texts = ["Some new text to classify"]
predictions = pipeline.predict_new(new_texts)
```
### Customizing the Pipeline
1. Adding New Models
Modify the NLPPipeline class to include additional models:
```bash
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class NLPPipeline:
    def __init__(self, max_features=5000):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Add or modify models here
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'svm': LinearSVC(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100),
            'xgboost': XGBClassifier()
        }
```
2. Modifying Text Preprocessing
Customize the TextPreprocessor class to change preprocessing steps:
```bash
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words
        self.stop_words.update(['custom', 'words'])
    
    def preprocess(self, text):
        # Add custom preprocessing steps
        # For example, remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emojis
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Continue with standard preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words]
        return ' '.join(tokens)
```
3. Customizing TF-IDF Features
Modify the vectorizer parameters in the NLPPipeline class:
```bash
self.vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(1, 3),  # Include up to trigrams
    min_df=2,  # Minimum document frequency
    max_df=0.95,  # Maximum document frequency
    stop_words='english'
)
```
4. Using Different Data Formats
Example with CSV file:
```bash
import pandas as pd

# Load data from CSV
df = pd.read_csv('your_data.csv')

# Initialize pipeline
pipeline = NLPPipeline(max_features=5000)

# Prepare data
pipeline.prepare_data(
    texts=df['text_column'].tolist(),
    labels=df['label_column'].tolist()
)
```
### Model Training Parameters
You can modify the parameters of each model:
```bash
self.models = {
    'naive_bayes': MultinomialNB(alpha=0.1),
    'logistic_regression': LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced'
    ),
    'svm': LinearSVC(
        C=1.0,
        max_iter=1000,
        class_weight='balanced'
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5
    )
}
```
### Cross-Validation Example
Add cross-validation to the pipeline:
```bash
from sklearn.model_selection import cross_val_score

def cross_validate_models(self, cv=5):
    results = {}
    for name, model in self.models.items():
        scores = cross_val_score(
            model, 
            self.X_train, 
            self.y_train, 
            cv=cv, 
            scoring='f1'
        )
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
    return results
```
### Error Handling
Add error handling to your implementation:
```bash
try:
    pipeline = NLPPipeline(max_features=5000)
    pipeline.prepare_data(texts, labels)
    results = pipeline.train_and_evaluate()
except ValueError as e:
    print(f"Error in data preparation: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
#### License
This project is licensed under the MIT License - see the LICENSE file for details







