from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def show_samples(label, count=5):
    samples = df[df["label"] == label].iloc[:count, 1:].values
    for i, img in enumerate(samples):
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.title(f"Label {label} - Sample {i}")
        plt.show()

df = pd.read_csv("data/sign_mnist_alpha_digits_train.csv")
label_counts = Counter(df["label"])
for label in sorted(label_counts):
    print(f"Label {label}: {label_counts[label]} samples")

# Example: check why 0 and 1 might be messed up
show_samples(29)  # Digit 0
show_samples(30)  # Digit 0
show_samples(31)  # Digit 0
show_samples(32)  # Digit 0
show_samples(33)  # Digit 0
show_samples(34)  # Digit 0
show_samples(35)  # Digit 0






