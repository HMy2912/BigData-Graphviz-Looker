from pyspark import SparkContext
import math
import time

sc = SparkContext.getOrCreate()

# 1. Improved Data Loading and Parsing
def load_and_parse_data(sc, filepath):
    """Load and parse CSV data, handling headers and malformed records"""
    try:
        # More robust header handling
        lines = sc.textFile(filepath)
        # header = lines.first()

        indexed_rdd = lines.zipWithIndex()

        header = indexed_rdd.filter(lambda x: x[1] == 0).map(lambda x: x[0]).collect()
        if header:
            header = header[0]
            data = indexed_rdd.filter(lambda x: x[1] > 0).map(lambda x: x[0])
        
        # Skip header and parse data
        data = lines.filter(lambda line: line != header).map(
            lambda line: [float(x.strip('"')) if x.strip('"').isdigit() else 0.0 
            for x in line.split(",")]
        )
        
        # Create feature-label pairs, handle empty lines
        rdd_data = data.filter(lambda cols: len(cols) > 1).map(
            lambda cols: (cols[:-1], cols[-1])
        )
        
        # Cache as we'll reuse this RDD
        rdd_data.cache()
        
        # Count features for verification
        num_features = len(rdd_data.first()[0]) if not rdd_data.isEmpty() else 0
        print(f"Loaded dataset with {rdd_data.count()} records and {num_features} features")
        
        return rdd_data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return sc.emptyRDD()

# Load data
file_path = "Classification with Logistic Regression/creditcard.csv/creditcard.csv"
rdd_data = load_and_parse_data(sc, file_path)

if rdd_data.isEmpty():
    print("No data loaded, exiting...")
    sc.stop()
    exit()

# 2. Enhanced Initialization with Feature Scaling
def feature_scaling(rdd):
    """Scale features to zero mean and unit variance"""
    # Calculate stats for each feature
    feature_stats = rdd.map(lambda x: x[0]).zipWithIndex().flatMap(
        lambda x: [(i, (val, val**2, 1)) for i, val in enumerate(x[0])]
    ).reduceByKey(
        lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    ).collectAsMap()
    
    # Calculate mean and std for each feature
    scaling_params = {}
    for i, (sum_x, sum_x2, count) in feature_stats.items():
        mean = sum_x / count
        std = math.sqrt((sum_x2 / count) - mean**2)
        scaling_params[i] = (mean, std if std != 0 else 1.0)
    
    # Scale features
    scaled_rdd = rdd.map(
        lambda x: (
            [(val - scaling_params[i][0])/scaling_params[i][1] for i, val in enumerate(x[0])],
            x[1]
        )
    )
    return scaled_rdd

# Scale features (important for gradient descent)
scaled_data = feature_scaling(rdd_data)

# Initialize parameters
num_features = len(scaled_data.first()[0])
initial_weights = [0.0] * num_features
learning_rate = 0.1  # Can be larger with scaled features
num_iterations = 50
regularization_param = 0.01  # L2 regularization

# 3. Enhanced Helper Functions
def sigmoid(z):
    """Numerically stable sigmoid function"""
    # Prevent overflow
    z = max(min(z, 20), -20)
    return 1.0 / (1.0 + math.exp(-z))

def predict(features, weights):
    """Compute prediction with bias term"""
    # Add bias term (1.0) to features
    extended_features = features + [1.0]
    extended_weights = weights + [0.0]  # Bias term weight
    z = sum(w * f for w, f in zip(extended_weights, extended_features))
    return sigmoid(z)

def compute_gradient(point, weights):
    """Compute gradient with regularization"""
    features, label = point
    prediction = predict(features, weights)
    error = prediction - label
    
    # Gradient for features
    gradient = [error * f for f in features]
    
    # Add regularization (excluding bias term)
    gradient = [g + regularization_param * w for g, w in zip(gradient, weights)]
    
    # Gradient for bias term (always 1.0)
    bias_gradient = error
    
    return gradient + [bias_gradient]

def compute_loss(point, weights):
    """Compute regularized logistic loss"""
    features, label = point
    prediction = predict(features, weights)
    
    # Avoid log(0)
    epsilon = 1e-15
    prediction = max(min(prediction, 1 - epsilon), epsilon)
    
    # Log loss
    loss = -label * math.log(prediction) - (1 - label) * math.log(1 - prediction)
    
    # L2 regularization
    reg_loss = 0.5 * regularization_param * sum(w**2 for w in weights)
    
    return loss + reg_loss

# 4. Enhanced Training with Early Stopping
def train_logistic_regression(rdd, initial_weights, learning_rate, max_iter):
    """Train logistic regression with early stopping"""
    weights = initial_weights.copy()
    best_weights = initial_weights.copy()
    best_loss = float('inf')
    no_improvement_count = 0
    
    for i in range(max_iter):
        start_time = time.time()
        
        # Compute gradients (map-reduce)
        gradients = rdd.map(
            lambda point: compute_gradient(point, weights)
        ).reduce(
            lambda a, b: [x + y for x, y in zip(a, b)]
        )
        
        # Average gradients
        num_points = rdd.count()
        gradients = [g / num_points for g in gradients]
        
        # Update weights
        weights = [w - learning_rate * g for w, g in zip(weights, gradients)]
        
        # Compute loss
        total_loss = rdd.map(
            lambda point: compute_loss(point, weights)
        ).sum()
        avg_loss = total_loss / num_points
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = weights.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= 5:
                print(f"Early stopping at iteration {i}")
                break
        
        # Print progress
        iteration_time = time.time() - start_time
        print(f"Iteration {i}: Loss = {avg_loss:.6f}, Time = {iteration_time:.2f}s")
    
    return best_weights

# Train model
print("\nStarting training...")
final_weights = train_logistic_regression(
    scaled_data, 
    initial_weights + [0.0],  # Add bias term
    learning_rate, 
    num_iterations
)
print("\nTraining completed.")

# 5. Comprehensive Evaluation
def evaluate_model(rdd, weights, threshold=0.5):
    """Evaluate model with multiple metrics"""
    # Make predictions
    predictions = rdd.map(
        lambda point: (
            predict(point[0], weights),  # Probability
            point[1]  # Actual label
        )
    ).cache()
    
    # Calculate metrics at different thresholds
    results = {}
    for threshold in [0.3, 0.5, 0.7]:
        # Classify based on threshold
        classified = predictions.map(
            lambda x: (1 if x[0] >= threshold else 0, x[1])
        )
        
        # Calculate confusion matrix
        true_pos = classified.filter(lambda x: x[0] == 1 and x[1] == 1).count()
        false_pos = classified.filter(lambda x: x[0] == 1 and x[1] == 0).count()
        true_neg = classified.filter(lambda x: x[0] == 0 and x[1] == 0).count()
        false_neg = classified.filter(lambda x: x[0] == 0 and x[1] == 1).count()
        
        # Calculate metrics
        accuracy = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[threshold] = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'true_pos': true_pos,
                'false_pos': false_pos,
                'true_neg': true_neg,
                'false_neg': false_neg
            }
        }
    
    return results

# Evaluate model
print("\nEvaluating model...")
eval_results = evaluate_model(scaled_data, final_weights)

for threshold, metrics in eval_results.items():
    print(f"\nMetrics at threshold {threshold}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(f"True Positives: {metrics['confusion_matrix']['true_pos']}")
    print(f"False Positives: {metrics['confusion_matrix']['false_pos']}")
    print(f"True Negatives: {metrics['confusion_matrix']['true_neg']}")
    print(f"False Negatives: {metrics['confusion_matrix']['false_neg']}")

# Save final weights
print("\nFinal weights (including bias term):")
print(final_weights)