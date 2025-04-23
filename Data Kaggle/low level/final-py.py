# %%
import numpy as np
import pandas as pd
from math import exp, log

# Load the dataset
file_path = "../creditcard.csv/creditcard.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']

# %%
# Add intercept term (column of 1s)
X = np.c_[np.ones(X.shape[0]), X]

# %%
# Split into train/test sets (80/20 split)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

# %%
# Normalize features (except intercept column)
for i in range(1, X_train.shape[1]):
    mean = X_train[:, i].mean()
    std = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - mean) / std
    X_test[:, i] = (X_test[:, i] - mean) / std

# %%
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.class_weight = class_weight
        self.weights = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, h, y, sample_weights=None):
        if sample_weights is None:
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        else:
            return ((-y * np.log(h) - (1 - y) * np.log(1 - h)) * sample_weights).mean()
    
    def fit(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1])
        
        # Calculate class weights if specified
        if self.class_weight == 'balanced':
            n_samples = len(y)
            n_classes = 2
            class_counts = np.bincount(y)
            self.class_weight_ = n_samples / (n_classes * class_counts)
            sample_weights = np.where(y == 1, self.class_weight_[1], self.class_weight_[0])
        else:
            sample_weights = None
        
        # Gradient descent
        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights)
            h = self._sigmoid(z)
            
            # Calculate gradient with class weights if specified
            if sample_weights is not None:
                gradient = np.dot(X.T, (h - y) * sample_weights) / len(y)
            else:
                gradient = np.dot(X.T, (h - y)) / len(y)
                
            self.weights -= self.learning_rate * gradient
            
            # Optional: Print loss every 100 iterations
            # if _ % 100 == 0:
            #     print(f'Loss at iteration {_}: {self._loss(h, y, sample_weights)}')
    
    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights))
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# %%
def smote(X, y, k=5, ratio=1.0):
    """
    SMOTE implementation for binary classification
    
    Parameters:
    X - feature matrix
    y - target vector
    k - number of nearest neighbors to consider
    ratio - sampling ratio (1.0 means equal classes)
    
    Returns:
    X_resampled, y_resampled
    """
    # Separate minority and majority classes
    minority_class = 1
    majority_class = 0
    X_min = X[y == minority_class]
    X_maj = X[y == majority_class]
    
    n_minority = X_min.shape[0]
    n_majority = X_maj.shape[0]
    n_features = X.shape[1]
    
    # Calculate how many synthetic samples to create
    n_synthetic = int((ratio * n_majority) - n_minority)
    
    # Find k nearest neighbors for each minority sample
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 to exclude self
    knn.fit(X_min)
    distances, indices = knn.kneighbors(X_min)
    
    # Generate synthetic samples
    synthetic_samples = np.zeros((n_synthetic, n_features))
    for i in range(n_synthetic):
        # Randomly select a minority sample
        idx = np.random.randint(0, n_minority)
        # Randomly select one of its k nearest neighbors
        neighbor_idx = np.random.choice(indices[idx, 1:])  # exclude self
        # Generate synthetic sample
        diff = X_min[neighbor_idx] - X_min[idx]
        synthetic_samples[i] = X_min[idx] + np.random.random() * diff
    
    # Combine original minority with synthetic samples
    X_resampled = np.vstack((X, synthetic_samples))
    y_resampled = np.hstack((y, np.ones(n_synthetic)))
    
    return X_resampled, y_resampled

# %%
def downsample(X, y, ratio=1.0):
    """Random undersampling for binary classification"""
    minority_class = 1
    majority_class = 0

    # Convert X and y to Pandas objects and reset index
    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)

    # Create boolean masks
    minority_mask = y_series == minority_class
    majority_mask = y_series == majority_class

    # Select minority and majority samples
    X_min, y_min = X_df[minority_mask], y_series[minority_mask]
    X_maj, y_maj = X_df[majority_mask], y_series[majority_mask]

    # Number of minority samples
    n_minority = len(X_min)

    # Number of majority samples to keep
    n_majority_down = int(n_minority / ratio)

    # Randomly sample majority class
    np.random.seed(42)
    X_maj_down = X_maj.sample(n=n_majority_down, random_state=42)
    y_maj_down = y_maj.loc[X_maj_down.index]

    # Combine undersampled majority with full minority class
    X_resampled = pd.concat([X_min, X_maj_down]).reset_index(drop=True)
    y_resampled = pd.concat([y_min, y_maj_down]).reset_index(drop=True)

    return X_resampled.to_numpy(), y_resampled.to_numpy()  # Convert back to NumPy arrays


# %%
# Option 1: Without SMOTE (using class weights)
model = LogisticRegression(learning_rate=0.1, n_iterations=1000, class_weight='balanced')
model.fit(X_train, y_train)

# %%
# Option 2: With SMOTE
X_train_smote, y_train_smote = smote(X_train, y_train, ratio=0.5)  # 50% minority class
model_smote = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model_smote.fit(X_train_smote, y_train_smote)

# %%
# Option 3: With downsampling
X_train_down, y_train_down = downsample(X_train, y_train, ratio=1.0)  # 1:1 ratio
model_down = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model_down.fit(X_train_down, y_train_down)

# %%
# Evaluation functions
def evaluate(y_true, y_pred):
    from collections import defaultdict
    metrics = defaultdict(float)
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics['confusion_matrix'] = np.array([[tn, fp], [fn, tp]])
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    return metrics

# %%
# Evaluate both models
y_pred = model.predict(X_test)
y_pred_smote = model_smote.predict(X_test)
y_pred_down = model_down.predict(X_test)

# %%

print("Without SMOTE:")
metrics = evaluate(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print("Confusion Matrix:")
print(metrics['confusion_matrix'])

print("\nWith SMOTE:")
metrics_smote = evaluate(y_test, y_pred_smote)
print(f"Accuracy: {metrics_smote['accuracy']:.4f}")
print(f"Precision: {metrics_smote['precision']:.4f}")
print(f"Recall: {metrics_smote['recall']:.4f}")
print(f"F1 Score: {metrics_smote['f1']:.4f}")
print("Confusion Matrix:")
print(metrics_smote['confusion_matrix'])

print("\nWith Downsampling:")
metrics_down = evaluate(y_test, y_pred_down)
print(f"Accuracy: {metrics_down['accuracy']:.4f}")
print(f"Precision: {metrics_down['precision']:.4f}")
print(f"Recall: {metrics_down['recall']:.4f}")
print(f"F1 Score: {metrics_down['f1']:.4f}")
print("Confusion Matrix:")
print(metrics_down['confusion_matrix'])

# %%
def find_best_threshold(model, X, y, thresholds=np.arange(0.1, 0.5, 0.01)):
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (model.predict_proba(X) >= threshold).astype(int)
        metrics = evaluate(y, y_pred)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
    
    return best_threshold

best_threshold = find_best_threshold(model, X_train, y_train)
print(f"Best threshold: {best_threshold:.2f}")
y_pred_tuned = (model.predict_proba(X_test) >= best_threshold).astype(int)

# %%
