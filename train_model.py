import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.metrics import classification_report
# Load the dataset

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

df = load_data('dataset/ACIIoT/train_set_small.csv')

# Encode categorical columns
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
joblib.dump(label_encoder, "models/label_encoder.joblib")
#df['Connection Type'] = label_encoder.fit_transform(df['Connection Type'])

# Split features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Feature selection: select the top 15 features
selector = SelectKBest(score_func=f_classif, k=15)
X_new = selector.fit_transform(X, y)
selected_features = selector.get_support(indices=True)
print(selected_features)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)


# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the selector and scaler
joblib.dump(selector, 'models/selector.joblib')
joblib.dump(scaler, 'models/scaler.joblib')


# List of models to train
models = [
    ('Random Forest', RandomForestClassifier()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Support Vector Classifier', SVC()),
    ('Decision Tree', DecisionTreeClassifier())
]

# Train and evaluate models
accuracies = []
weighted_f1_scores = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    f1_scores = f1_score(y_test, y_pred, average=None)
    class_counts = y_test.value_counts().sort_index()
    weighted_f1 = (class_counts * f1_scores).sum() / class_counts.sum()
    weighted_f1_scores.append(weighted_f1)
    
    # Save the model
    joblib.dump(model, f'models/{name}.joblib')
    print(name)
    print("accuracy:", accuracy)
    print("weighted f1:",weighted_f1)
     # Print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix for {name}:\n{cm}\n')
    
    

# Compute average accuracy and weighted F1 score
avg_accuracy = sum(accuracies) / len(accuracies)
avg_weighted_f1 = sum(weighted_f1_scores) / len(weighted_f1_scores)

print(f'Average Accuracy: {avg_accuracy:.4f}')
print(f'Average Weighted F1 Score: {avg_weighted_f1:.4f}')


df = load_data("dataset/ACIIoT/test_set.csv")
df = df[df['Label'] != 'ARP Spoofing']

# Extract true labels and drop them from the dataset
y_true = df['Label'].values
df.drop(columns=['Label'], inplace=True)

# Preprocess the data
X = preprocess_data(df)
for model_name, model in models:
    # Load the model
    print(model_name)
    model = joblib.load("models/"+model_name+".joblib")

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
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    print(f'Confusion Matrix for {name}:\n{cm}\n')
    print(classification_report(y_true, y_pred, target_names = labels))
    accuracy = accuracy_score(y_true, y_pred)
    f1_scores = f1_score(y_true, y_pred, average="macro")

    print("accuracy:", accuracy)
    print("weighted f1:",weighted_f1)


