
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.info())
print(train_data.describe())
print(train_data.isnull().sum())
sns.countplot(data=train_data, x='Survived')
plt.title('Survival Count')
plt.show()

sns.countplot(data=train_data, x='Survived', hue='Sex')
plt.title('Survival by Gender')
plt.show()

sns.countplot(data=train_data, x='Survived', hue='Pclass')
plt.title('Survival by Class')
plt.show()

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Classification Report:\n", classification_report(y_train, y_train_pred))

predictions = model.predict(X_test)

submission = pd.read_csv('gender_submission.csv')
submission['Survived'] = predictions
output_path = 'C:/Users/joshi/Documents/titanic_predictions2.csv'

try:
    submission.to_csv(output_path, index=False)
    print("Predictions saved successfully to:", output_path)
except Exception as e:
    print("Error saving predictions:", str(e))

plt.show()

