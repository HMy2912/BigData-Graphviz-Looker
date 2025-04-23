from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark import SparkContext

# Initialize Spark
spark = SparkSession.builder.appName("LowLevelLogisticRegression").getOrCreate()
sc = spark.sparkContext

# Load dataset as RDD
raw_rdd = sc.textFile("creditcard.csv")

# Extract header and filter it out
header = raw_rdd.first()
data_rdd = raw_rdd.filter(lambda row: row != header)

# Parse CSV lines into (features, label)
def parse_line(line):
    values = line.split(",")
    label = float(values[-1])  # Last column is 'Class'
    features = [float(x) for x in values[1:-1]]  # Exclude 'Time' and 'Class'
    return (features, label)

parsed_rdd = data_rdd.map(parse_line)

# Initialize weights to zeros
num_features = len(parsed_rdd.first()[0])  # Get number of features
weights = [0.0] * num_features
learning_rate = 0.01
iterations = 100

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + pow(2.718, -z))

# Define prediction function
def predict(features, weights):
    z = sum(w * f for w, f in zip(weights, features))
    return sigmoid(z)

# Perform gradient descent
for _ in range(iterations):
    # Compute gradients
    gradients = parsed_rdd.map(lambda x: 
        [(predict(x[0], weights) - x[1]) * f for f in x[0]]
    ).reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    
    # Update weights
    weights = [w - learning_rate * grad for w, grad in zip(weights, gradients)]

# Evaluate the model
predictions = parsed_rdd.map(lambda x: (1 if predict(x[0], weights) > 0.5 else 0, x[1]))

# Compute accuracy
accuracy = predictions.filter(lambda x: x[0] == x[1]).count() / float(parsed_rdd.count())

# Print results
print(f"Final Weights: {weights[:5]} ...")  # Print first 5 weights
print(f"Model Accuracy: {accuracy:.4f}")

# Stop Spark
spark.stop()
