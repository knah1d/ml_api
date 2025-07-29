# services/model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    # Initialize LabelEncoder
    le = LabelEncoder()

    # Apply LabelEncoder to each column with non-numerical data
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column].astype(str))

    return df

def train_model(file_path: str):
    try:
        logger.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info("File read successfully")

        # Preprocess the data
        df = preprocess_data(df)

        # Assuming the last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Save the model
        joblib.dump(model, 'models/model.pkl')
        logger.info("Model trained and saved successfully")

        return accuracy
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
