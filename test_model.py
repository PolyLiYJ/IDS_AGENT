import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
    
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
def load_data(file_path):
    """Load test data from CSV file and preprocess."""
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Connection Type']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def preprocess_data(df):
    """Preprocess the input data using the saved selector and scaler."""
    selector = joblib.load('models/selector.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    X_selected = selector.transform(df)
    X_scaled = scaler.transform(X_selected)
    
    return X_scaled

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix using matplotlib."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.show()
    plt.savefig('sine_wave_plot.png', dpi=600, bbox_inches='tight')


def classify_and_plot(file_path, model_name):
    """Load model, classify data, and plot confusion matrix."""
    df = load_data(file_path)
    df = df[df['Label'] != 'ARP Spoofing']

    # Extract true labels and drop them from the dataset
    y_true = df['Label'].values
    df.drop(columns=['Label'], inplace=True)
    
    # Preprocess the data
    X = preprocess_data(df)
    
    # Load the model
    model = joblib.load(model_name)
    
    # Predict the labels
    y_pred = model.predict(X)
    label_encoder = joblib.load('models/label_encoder.joblib')

    # # Assuming label_encoder is your fitted LabelEncoder and y_pred is your predicted labels
    # unique_classes = set(label_encoder.classes_)
    # print(unique_classes)

    # # Filter y_pred to only contain labels seen during training
    # y_pred_filtered = [label for label in y_pred if label in unique_classes]

    # # Now transform y_pred_filtered
    # y_pred = label_encoder.transform(y_pred_filtered)
        
    y_pred = label_encoder.inverse_transform(y_pred)
    labels = label_encoder.classes_
    
    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels)


def vote(file_path, model_names):
    """Load models, classify data by voting, report the classification report, and plot confusion matrix."""
    
    # Load and preprocess the data
    df = load_data(file_path)
    df = df[df['Label'] != 'ARP Spoofing']

    # Extract true labels and drop them from the dataset
    y_true = df['Label'].values
    df.drop(columns=['Label'], inplace=True)
    
    # Preprocess the data
    X = preprocess_data(df)
    
    # Initialize an array to store predictions from each model
    all_predictions = []
    
    # Load each model and predict
    for model_name in model_names:
        print(model_name)
        model = joblib.load(model_name)
        y_pred = model.predict(X)
        all_predictions.append(y_pred)
    
    # Convert the list of predictions into a numpy array for easy manipulation
    all_predictions = np.array(all_predictions)
    print("perform voting:")
    # Perform majority voting
    y_pred_voted = mode(all_predictions, axis=0)[0].flatten()
    
    # Load the label encoder to decode labels
    label_encoder = joblib.load('models/label_encoder.joblib')
    y_pred_voted = label_encoder.inverse_transform(y_pred_voted)
    labels = label_encoder.classes_
    
    # Print classification report
    print(classification_report(y_true, y_pred_voted, target_names=labels))
    
    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred_voted, labels)


if __name__ == "__main__":
    test_file_path = 'dataset/ACIIoT/test_set_small.csv'
    model_names = ['models/Logistic Regression.joblib', 'models/Decision Tree.joblib', 'models/K-Nearest Neighbors.joblib']
    
    vote(test_file_path, model_names)
