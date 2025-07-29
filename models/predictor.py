import joblib

def predict(data: list):
    model = joblib.load('models/model.pkl')

    predictions = model.predict(data)
    return predictions.tolist()