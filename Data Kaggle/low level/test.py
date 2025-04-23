from pyspark import SparkContext
import math

sc = SparkContext.getOrCreate()

# 1. Load and parse the data
# Assuming we're using the credit card fraud dataset
lines = sc.textFile("Classification with Logistic Regression/creditcard.csv/creditcard.csv")

indexed_rdd = lines.zipWithIndex()

header = indexed_rdd.filter(lambda x: x[1] == 0).map(lambda x: x[0]).collect()
if header:
    header = header[0]
    data = indexed_rdd.filter(lambda x: x[1] > 0).map(lambda x: x[0])

# header = lines.first()
# data = lines.filter(lambda line: line != header)

# Parse each line: split by comma and convert each element to float
# Last column is the label (Class: 0 or 1), all previous columns are features
parsed_data = data.map(
    lambda line: [float(x.strip('"')) for x in line.split(",")]
)

# Create an RDD where each record is a tuple: (features as list, label)
rdd_data = parsed_data.map(lambda cols: (cols[:-1], cols[-1]))

# Cache the data as we'll use it multiple times
rdd_data.cache()

# 2. Initialize parameters
num_features = len(rdd_data.first()[0])
initial_weights = [0.0] * num_features
learning_rate = 0.01
num_iterations = 100
regularization_param = 0.1  # L2 regularization parameter

# 3. Define helper functions
def sigmoid(z):
    """Sigmoid function"""
    return 1.0 / (1.0 + math.exp(-z))

def predict(features, weights):
    """Compute prediction using sigmoid function"""
    z = sum(w * f for w, f in zip(weights, features))
    return sigmoid(z)

def compute_gradient(point, weights):
    """Compute gradient for a single data point"""
    features, label = point
    prediction = predict(features, weights)
    error = prediction - label
    gradient = [error * f for f in features]
    return gradient

def compute_loss(point, weights):
    """Compute logistic loss for a single data point"""
    features, label = point
    prediction = predict(features, weights)
    # Avoid log(0)
    epsilon = 1e-15
    prediction = max(min(prediction, 1 - epsilon), epsilon)
    return -label * math.log(prediction) - (1 - label) * math.log(1 - prediction)

# 4. Implement gradient descent
current_weights = initial_weights.copy()

for i in range(num_iterations):
    # Compute gradients for all points and average them
    gradients = rdd_data.map(
        lambda point: compute_gradient(point, current_weights)
    ).reduce(
        lambda a, b: [x + y for x, y in zip(a, b)]
    )
    
    # Add regularization (L2 penalty)
    regularization = [regularization_param * w for w in current_weights]
    
    # Average gradients and add regularization
    num_points = rdd_data.count()
    gradients = [(g / num_points) + r for g, r in zip(gradients, regularization)]
    
    # Update weights
    current_weights = [
        w - learning_rate * g for w, g in zip(current_weights, gradients)
    ]
    
    # Compute and print loss every 10 iterations
    if i % 10 == 0:
        total_loss = rdd_data.map(
            lambda point: compute_loss(point, current_weights)
        ).sum()
        avg_loss = total_loss / rdd_data.count()
        print(f"Iteration {i}: Average Loss = {avg_loss}")

# Final weights
print("Final weights:", current_weights)

# 5. Make predictions and evaluate
# Function to classify based on threshold (default 0.5)
def classify(features, weights, threshold=0.5):
    return 1 if predict(features, weights) >= threshold else 0

# Calculate accuracy
predictions = rdd_data.map(
    lambda point: (classify(point[0], current_weights), point[1])
)
correct = predictions.filter(lambda pred: pred[0] == pred[1]).count()
total = rdd_data.count()
accuracy = correct / total

print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, F1-score
true_positives = predictions.filter(lambda pred: pred[0] == 1 and pred[1] == 1).count()
false_positives = predictions.filter(lambda pred: pred[0] == 1 and pred[1] == 0).count()
false_negatives = predictions.filter(lambda pred: pred[0] == 0 and pred[1] == 1).count()

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")