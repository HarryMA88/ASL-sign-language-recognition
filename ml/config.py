# This file contains the configuration settings for the training process of a machine learning model.
# It includes hyperparameters such as default batch size, number of epochs, learning rate, and other settings.
# Used for consistency across the training process.
TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,
    "device": "cuda",
    "num_classes": 36,
    "image_size": 28,
    "normalize_mean": 0.5,
    "normalize_std": 0.5,
}
