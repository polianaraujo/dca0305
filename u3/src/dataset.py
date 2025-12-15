# Script para carregar o SVHN (DataLoader)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_svhn_loaders(batch_size=64, root='./data', num_workers=2):
    """
    Retorna DataLoaders de Treino e Teste para o SVHN.
    """
    # Transformações: Converter para Tensor e Normalizar
    # Média e Std aproximados do SVHN
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    # Download e Carregamento do Dataset de Treino
    train_dataset = datasets.SVHN(
        root=root, 
        split='train', 
        download=True, 
        transform=transform
    )

    # Download e Carregamento do Dataset de Teste
    test_dataset = datasets.SVHN(
        root=root, 
        split='test', 
        download=True, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, test_loader