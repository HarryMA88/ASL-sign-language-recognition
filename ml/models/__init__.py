from .resnet import get_model as resnet
from .alexnet import get_model as alexnet
from .lebron import get_model as lebron

# Registry mapping model names to their constructor functions
# Used to select models during training.
model_registry = {
    "resnet": resnet,
    "alexnet": alexnet,
    "lebron": lebron
}
