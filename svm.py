import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

# Assuming the target variable is named 'Diabetes_binary' in your dataset
# and all other columns are features
data = data.drop(columns=['Income'])

# Loại bỏ dữ liệu lặp
data_cleaned = data.drop_duplicates()

X = data_cleaned.drop('Diabetes_binary', axis=1)
y = data_cleaned['Diabetes_binary']

# Split the data into training and testing sets with an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng mẫu nhỏ hơn của dữ liệu
X_train_sample = X_train.sample(n=10000, random_state=42)
y_train_sample = y_train[X_train_sample.index]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_sample_scaled = scaler.fit_transform(X_train_sample)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Thiết lập GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}
grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=2, n_jobs=-1)

# Huấn luyện mô hình với GridSearchCV
grid.fit(X_train_sample_scaled, y_train_sample)

# In ra tham số tốt nhất
print(grid.best_params_)

# Dự đoán và tính toán độ chính xác
y_train_pred = grid.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = grid.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Accuracy on the training set: {train_accuracy * 100:.2f}%')
print(f'Accuracy on the test set: {test_accuracy * 100:.2f}%')
