__path__ = r"fake_job_postings.csv"

import re, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(__path__)

drop_cols = [
    'job_id','location','department','salary_range',
    'employment_type','required_experience',
    'required_education','industry','function'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

TEXT_COLS = ['title','company_profile','description','requirements','benefits']

df[TEXT_COLS] = df[TEXT_COLS].fillna('').astype(str)
df['text'] = df[TEXT_COLS].agg(' '.join, axis=1).str.strip()
df['fraudulent'] = df['fraudulent'].astype(int)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

x = df['clean_text']
y = df['fraudulent']

x_tr, x_te, y_tr, y_te = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words='english',
        min_df=2
    )),
    ('clf', LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    ))
])

pipeline.fit(x_tr, y_tr)

y_pred = pipeline.predict(x_te)

print("Classification Report:")
print(classification_report(y_te, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_te, y_pred))

SAVE_PIPE = "fakejob_pipeline.joblib"

joblib.dump(pipeline, SAVE_PIPE)
print("Pipeline saved at:", SAVE_PIPE)
