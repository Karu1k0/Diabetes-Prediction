import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

duplicate_rows = data.duplicated()

# Loại bỏ dữ liệu lặp
data_cleaned = data.drop_duplicates()


#data = data.drop(columns=['Income'])
X = data_cleaned.drop('Diabetes_binary', axis=1)
y = data_cleaned['Diabetes_binary']

# Split the data into training and testing sets with an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of the Decision Tree model: {accuracy * 100:.2f}%')
