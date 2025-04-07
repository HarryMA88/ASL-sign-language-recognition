from models.resnet.model import build_model as build_resnet
from models.alexnet.model import build_model as build_alexnet

MODEL_REGISTRY = {
    "resnet": build_resnet,
    "alexnet": build_alexnet,
}

def get_model(name: str, num_classes: int):
    return MODEL_REGISTRY[name](num_classes)
