import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os

# Load the dataset
file_path = "data/geospatial_intent_dataset.csv"
df = pd.read_csv(file_path)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['output'], test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Evaluate performance
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save classification report to CSV
report_path = "data/classification_report.csv"
report_df.to_csv(report_path)

# Create and save confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=pipeline.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
conf_matrix_path = "data/confusion_matrix.png"
plt.savefig(conf_matrix_path, bbox_inches="tight")

# Save the trained model
model_path = "data/geospatial_intent_classifier.pkl"
joblib.dump(pipeline, model_path)

report_path, conf_matrix_path, model_path
