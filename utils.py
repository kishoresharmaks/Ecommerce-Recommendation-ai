import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None

def preprocess_data(data):
    data = data.fillna('')
    return data

def evaluate_recommendations(true_labels, predicted_labels):
    k = min(len(predicted_labels), 5)
    relevant = sum(1 for pred in predicted_labels[:k] if pred in true_labels)
    precision = relevant / k if k > 0 else 0
    return precision

if __name__ == "__main__":
    data = load_data('sample.csv')
    if data is not None:
        processed_data = preprocess_data(data)
        print("Data Loaded and Preprocessed:", processed_data.head())