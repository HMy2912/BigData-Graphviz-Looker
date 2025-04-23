# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "../creditcard.csv/creditcard.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']

# %%
# Standardize the 'Amount' feature manually
def standardize_feature(feature):
    mean = np.mean(feature)
    std = np.std(feature)
    standardized = (feature - mean) / std
    return standardized, mean, std

# Apply standardization
X['Amount'], amount_mean, amount_std = standardize_feature(X['Amount'].values)

# To apply this standardization to new data later:
# new_data = (new_data - amount_mean) / amount_std
# Standardize the 'Amount' feature
# scaler = StandardScaler()
# X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# %%
# Split into train and test sets (stratified to maintain class ratio)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42)

def stratified_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get indices for each class
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # Shuffle indices
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    # Calculate split points
    split_0 = int(len(class_0_indices) * (1 - test_size))
    split_1 = int(len(class_1_indices) * (1 - test_size))
    
    # Split indices
    train_indices = np.concatenate([
        class_0_indices[:split_0],
        class_1_indices[:split_1]
    ])
    test_indices = np.concatenate([
        class_0_indices[split_0:],
        class_1_indices[split_1:]
    ])
    
    # Shuffle the combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Return splits
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

# Perform the stratified split
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=42)

# %%
# Convert to numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# %%
# Add intercept term (column of 1s)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# %%
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    """Compute the logistic regression cost function"""
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y @ np.log(h)) - ((1 - y) @ np.log(1 - h))
    return cost / m

def gradient_descent(X, y, theta, alpha, num_iters):
    """Perform gradient descent to learn theta"""
    m = len(y)
    cost_history = []
    
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def predict(X, theta, threshold=0.5):
    """Make predictions using learned theta"""
    return (sigmoid(X @ theta) >= threshold).astype(int)

# %%
# Calculate class weights (important for imbalanced data)
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
total = neg + pos

# The weighting here means we'll count each positive example 100x more
# to compensate for the class imbalance
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

# Create sample weights
sample_weights = np.ones(y_train.shape)
sample_weights[y_train == 0] = weight_for_0
sample_weights[y_train == 1] = weight_for_1

# %%
# Initialize parameters
theta = np.zeros(X_train.shape[1])
alpha = 0.01  # learning rate
iterations = 1000

# %%
# Modified gradient descent function to handle weights
def gradient_descent(X, y, theta, alpha, num_iters, sample_weights=None):
    """Perform gradient descent to learn theta with optional sample weights"""
    m = len(y)
    cost_history = []
    
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        if sample_weights is not None:
            gradient = (X.T @ (sample_weights * (h - y))) / m
        else:
            gradient = X.T @ (h - y) / m
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history


# %%
# Train with weighted gradient descent
theta, cost_history = gradient_descent(
    X_train, 
    y_train, 
    theta, 
    alpha, 
    iterations,
    sample_weights=sample_weights  # Modified gradient descent to include weights
)

# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predict on test set
y_pred = predict(X_test, theta)

# Since data is imbalanced, don't use accuracy
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:", roc_auc_score(y_test, sigmoid(X_test @ theta)))

# %%
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_scores = sigmoid(X_test @ theta)

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Plot precision-recall curve
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="center left")
plt.title("Precision-Recall Tradeoff")
plt.show()

# Choose optimal threshold (example: maximize F1-score)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")

# Make predictions with optimal threshold
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)

print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred_optimal))

# %%
# Get feature importance (absolute value of coefficients)
feature_importance = pd.DataFrame({
    'Feature': ['Intercept'] + list(data.drop(['Time', 'Class'], axis=1).columns),
    'Coefficient': theta
})

# Sort by absolute value
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# %%
