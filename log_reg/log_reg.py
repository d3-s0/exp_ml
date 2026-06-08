import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Read data
df = pd.read_csv('log_reg/iris/iris.csv')

# Features and target
X = df.iloc[:,0:4]
y = df['Species']

# Split into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make prediction
prediction = model.predict(X_test)
print(metrics.accuracy_score(prediction, y_test))