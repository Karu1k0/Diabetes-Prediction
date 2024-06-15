import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

# Assuming the target variable is named 'Diabetes_binary' in your dataset
#data = data.drop(columns=['Income'])

# Loại bỏ dữ liệu lặp
data_cleaned = data.drop_duplicates()

X = data_cleaned.drop('Diabetes_binary', axis=1)
y = data_cleaned['Diabetes_binary']

# Split the data into training and testing sets with an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of the XGBoost model: {accuracy * 100:.2f}%')
