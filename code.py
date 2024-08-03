import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn import neighbors, preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
# Fix random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
# Loading Alzheimer Dataset
df = pd.read_csv("alzheimer-data.csv")

# Handle missing values using SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df.iloc[:, :-1])  # Impute missing values for all columns except the last 'target' column

# Get the target variable
y = df['target']

# Fit the LDA model
lda = LinearDiscriminantAnalysis(n_components=2)
lda_X = lda.fit(X_imputed, y).transform(X_imputed)

# LDA cluster plot for Alzheimer dataset
plt.scatter(lda_X[y == 0, 0], lda_X[y == 0, 1], s=100, c='green', label='Target 0 (Non-Alzheimer)')
plt.scatter(lda_X[y == 1, 0], lda_X[y == 1, 1], s=100, c='red', label='Target 1 (Alzheimer)')
plt.scatter(lda_X[y == 2, 0], lda_X[y == 2, 1], s=100, c='yellow', label='Target 2 (Mild Dermented)')  # Adding green class
plt.title('LDA plot for Alzheimer Dataset')
plt.legend()
plt.show()

# Data augmentation using SMOTE
smote = SMOTE(random_state=random_seed)
X_augmented, y_augmented = smote.fit_resample(X_imputed, y)

# Calculate sizes of original and augmented datasets
original_size = len(y)
augmented_size = len(y_augmented)
print("Original dataset size:", original_size)
print("Augmented dataset size:", augmented_size)

# Assigning X_augmented and y_augmented for KNN classification
X = X_augmented
y = y_augmented

# Splitting the augmented data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

# Scaling the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling using KNN classifier with cross-validation
knn = neighbors.KNeighborsClassifier(n_neighbors=7)
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)  # 5-fold cross-validation
avg_accuracy = cv_scores.mean()

# Fitting the model
knn.fit(X_train_scaled, y_train)

# Predicting on test set
y_pred = knn.predict(X_test_scaled)

# Display the Output
print('Average Accuracy Score (Cross-Validation):', avg_accuracy)
print('Individual CV Scores:', cv_scores)
print('Accuracy Score on Test Set:', accuracy_score(y_test, y_pred))
print('Confusion matrix \n', confusion_matrix(y_test, y_pred))
print('Classification Report \n', classification_report(y_test, y_pred))

# Initialize lists to store results
n_neighbors_range = range(1, 20)
train_accuracy = []
test_accuracy = []

# Train and test KNN classifier for different numbers of neighbors
for n_neighbors in n_neighbors_range:
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)
    train_accuracy.append(accuracy_score(y_train, y_train_pred))
    test_accuracy.append(accuracy_score(y_test, y_test_pred))

# Plotting the performance graph
plt.plot(n_neighbors_range, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(n_neighbors_range, test_accuracy, label='Test Accuracy', marker='o')
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(min(n_neighbors_range), max(n_neighbors_range)+1, 1.0))  # Set x-axis ticks for each integer value
plt.legend()
plt.grid(True)
plt.show()
