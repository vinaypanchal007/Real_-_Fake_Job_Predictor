# Real_-_Fake_Job_Predictor
A machine learning project to detect fake job postings using NLP and Logistic Regression.
The main aim of this project is to learn how NLP, machine learning models, and basic deployment work together in a real-world scenario.


## Dataset Used
The dataset is taken from Kaggle: **Real or Fake Job Posting Prediction**.
- Labels:
  - 0 → Real Job
  - 1 → Fake Job
- The dataset contains only real and fake labels.
- There is no separate "unsure" label in the dataset.


## How the Model Works
1. Job details like title, description, requirements, etc. are combined.
2. Text is cleaned by removing URLs, special characters, and extra spaces.
3. TF-IDF is used to convert text into numerical features.
4. Logistic Regression is used to predict whether a job is fake or real.
5. The model outputs a probability instead of only a label.


## Decision Logic
Based on the fake probability:
- Fake probability ≤ 0.40 → Real Job.
- Fake probability between 0.40 and 0.60 → Unsure.
- Fake probability ≥ 0.60 → Fake Job.
- The "Unsure" category is added to handle borderline cases and reduce wrong predictions.


## Web Application
A simple Streamlit app is created where users can:
- Enter job details.
- See the fake probability.
- Get the result as Real, Fake, or Unsure.


## Limitations
- The dataset has only 2 labels 0 and 1 (real and fake respectively).
- Subtle scams may not always be detected.
