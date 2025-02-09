import os
import pandas as pd
import pickle
import logging
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Start of inferencing process")

# Defining directories
# For Docker Use
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Defining file paths (we are only interested at predcitions of the best model)
# That is why we are downloading Logistic Regression model that was trained on lemmatized, tf-idf vectorized dataset
# As long as test set, in which reviews were also lemmatized and vectorized with tf-idf technique
test_features_path = os.path.join(BASE_DIR, "data", "processed", "test_lemma_tfidf.csv")
model_path = os.path.join(BASE_DIR, "outputs", "models", "Logistic_Regression_lemma_tfidf.pkl")
raw_test_path = os.path.join(BASE_DIR, "data", "raw", "test.csv")
output_predictions_path = os.path.join(BASE_DIR, "outputs", "predictions", "predictions.csv")

# Loading test dataset (lemmatized; TF-IDF vectorization)
logging.info(f"Loading test features from {test_features_path}...")
test_data = pd.read_csv(test_features_path)

# Loading pretrained model
logging.info(f"Loading trained model from {model_path}...")
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Predicting using pretrained model
logging.info("Making predictions...")
X_test = test_data.drop(columns=["sentiment"])
y_test = test_data["sentiment"]
test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
logging.info(f"Accuracy of model on test set {accuracy}...")

# Loading raw test dataset, as we need unlemmatized and unvectorized reviews
logging.info(f"Loading raw test data from {raw_test_path}...")
raw_test_data = pd.read_csv(raw_test_path).drop(columns=["sentiment"])
raw_test_data = raw_test_data.drop_duplicates()

# Merging raw reviews and predictions of trained model
raw_test_data["prediction"] = test_predictions

# Saving inference
os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
raw_test_data.to_csv(output_predictions_path, index=False)

logging.info(f"Predictions saved to {output_predictions_path}")
logging.info("Inference step is completed!")