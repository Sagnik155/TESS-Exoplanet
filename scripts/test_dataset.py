from training.dataset import DatasetLoader
import numpy as np

loader = DatasetLoader("data/processed")

X_train, X_val, X_test, y_train, y_val, y_test = loader.split()

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

print()

print("Train positives :", np.sum(y_train))
print("Train negatives :", len(y_train) - np.sum(y_train))

print()

print("Validation positives :", np.sum(y_val))
print("Validation negatives :", len(y_val) - np.sum(y_val))

print()

print("Test positives :", np.sum(y_test))
print("Test negatives :", len(y_test) - np.sum(y_test))