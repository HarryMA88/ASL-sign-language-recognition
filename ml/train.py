import torch
from model_data.dataset import get_loaders
from models.lebron2_3.lebron import Lebron23
from trainer.trainer import train_model

def main(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/sign_mnist_alpha_digits_train.csv" #update path here
    train_loader, val_loader = get_loaders(csv_path, batch_size=64, val_split=0.2)

    model = Lebron23(num_classes=36)

    config = {
        "name": "lebron2_3", 
        "lr": 1e-3,
        "epochs": epochs,
        "save_path": "saved_models"
    }

    train_model(model, train_loader, val_loader, device, config)

if __name__ == "__main__":
    main(30)
