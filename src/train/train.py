import os
import subprocess
import logging
import pandas as pd
import pickle
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

# For Docker use
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# For Docker use  
model_save_path = os.path.join(BASE_DIR, "models")

#model_save_path = os.path.join(BASE_DIR, "outputs", "models")

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=["sentiment"])
    y_train = train_data["sentiment"]

    X_test = test_data.drop(columns=["sentiment"])
    y_test = test_data["sentiment"]

    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def save_model(model, model_name, save_path):
    with open(os.path.join(save_path, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(model, f)

def models_building(train_data_path, test_data_path, identification):
    X_train, y_train, X_test, y_test = load_data(train_data_path, test_data_path)

    # XGBoost classifier was commented, because it was taking too long for this model to train,
    # around 20 minutes for each (and we have 4 of them). If you want you can uncomment this model and use it training processalso;
    # It was mentioned also in README;
    # XGBoost Classifiers were trained in notebook, you can also check there them;
    models = {
        "Naive_Bayes": ComplementNB(),
        "Logistic_Regression": LogisticRegression(max_iter=1000, C=2.15, penalty="l2", solver="liblinear"),
        "Random_Forest": RandomForestClassifier(n_estimators=500, max_depth=50, random_state=1),
        # "XGBoost": XGBClassifier(objective="binary:logistic", max_depth=10, learning_rate=0.05, n_estimators=500)
    }

    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        logging.info(f"Training model: {name}")
        trained_model, accuracy = train_model(model, X_train, y_train, X_test, y_test)
        logging.info(f"{name} - Test Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

        name = name + "_" + identification
        save_model(trained_model, name, model_save_path)
    logging.info(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")


logging.info("Launching training scrpits...")

preprocess_script = os.path.join(BASE_DIR, "src", "train", "preprocess.py")
feature_engineering_script = os.path.join(BASE_DIR, "src", "train", "feature_engineering.py")

# Launching preprocessing script
logging.info("Launching data preprocessing script...")
subprocess.run(["python", "/app/train/preprocess.py"], check=True)

# Launching feature engineering script
logging.info("Launching feature engineering script...")
subprocess.run(["python", "/app/train/data_engineering.py"], check=True)

os.makedirs(model_save_path, exist_ok=True)

logging.info("\n-----------------------------------Training models (STEMMING;COUNT VECTORIZING)---------------------")
#Count stem models building
train_data_path = os.path.join(BASE_DIR, "data", "processed", "train_stem_count.csv")
test_data_path = os.path.join(BASE_DIR, "data", "processed", "test_stem_count.csv")
identification = "stem_count"
models_building(train_data_path, test_data_path, identification)

logging.info("\n-----------------------------------Training models (LEMMATIZATION;COUNT VECTORIZING)---------------------")
#Count lemma models building
train_data_path = os.path.join(BASE_DIR, "data", "processed", "train_lemma_count.csv")
test_data_path = os.path.join(BASE_DIR, "data", "processed", "test_lemma_count.csv")
identification = "lemma_count"
models_building(train_data_path, test_data_path, identification)

logging.info("\n-----------------------------------Training models (STEMMING;TF-IDF VECTORIZING)---------------------")
#TF-IDF stem models building
train_data_path = os.path.join(BASE_DIR, "data", "processed", "train_stem_tfidf.csv")
test_data_path = os.path.join(BASE_DIR, "data", "processed", "test_stem_tfidf.csv")
identification = "stem_tfidf"
models_building(train_data_path, test_data_path, identification)

logging.info("\n-----------------------------------Training models (LEMMATIZATION;TF-IDF VECTORIZING)---------------------")
#TF-IDF lemma models building
train_data_path = os.path.join(BASE_DIR, "data", "processed", "train_lemma_tfidf.csv")
test_data_path = os.path.join(BASE_DIR, "data", "processed", "test_lemma_tfidf.csv")
identification = "lemma_tfidf"
models_building(train_data_path, test_data_path, identification)

logging.info("End of models training!")
logging.info("Models and prepared data was saved in container!")
logging.info("Transfer models and prepared data to your machine!")