from .mnist_loader import load_mnist
from .titanic_loader import load_titanic
from .wine_loader import load_wine_dataset
from .adult_loader import load_adult_income
from .compas_loader import load_compas
from .cifar10_loader import load_cifar10
from .breast_cancer_loader import load_breast_cancer_dataset as load_breast_cancer
from .vehicle_loader import load_vehicle_dataset

__all__ = [
    'load_mnist',
    'load_titanic',
    'load_wine_dataset',
    'load_adult_income',
    'load_compas',
    'load_cifar10',
    'load_breast_cancer',
    'load_vehicle_dataset'
]
