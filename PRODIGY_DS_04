# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# Read the dataset from Google Drive
df = pd.read_csv('/content/drive/MyDrive/bank.csv')

# Display the first few rows of the dataset
df.head()

# Display information about the dataset, including data types and missing values
df.info()

# Display descriptive statistics of the dataset
df.describe()

# Check for missing values in the dataset
df.isnull().sum()

# Check for duplicated rows in the dataset
df.duplicated()

# Import necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Separate features (X) and target variable (y)
X = df.drop('deposit', axis=1)
y = df['deposit']

# Apply label encoding to categorical columns
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
