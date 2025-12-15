# Funções para gerar os plots específicos (EWMA, Gradients)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_losses(train_losses, val_losses, title="Curva de Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gradients_distribution(raw, ewma, adapted, epoch_idx):
    """
    Plota histogramas comparando gradientes crus vs adaptados.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Raw
    sns.histplot(raw, bins=50, kde=True, ax=ax[0], color='gray', stat='density')
    ax[0].set_title(f'Gradientes Crus (Epoch {epoch_idx})')
    
    # Plot EWMA
    sns.histplot(ewma, bins=50, kde=True, ax=ax[1], color='blue', stat='density')
    ax[1].set_title('EWMA (Momentum)')
    
    # Plot Adapted
    sns.histplot(adapted, bins=50, kde=True, ax=ax[2], color='green', stat='density')
    ax[2].set_title('Gradientes Adaptados (Adam)')
    
    plt.tight_layout()
    plt.show()

def plot_lr_history(lrs, title="Learning Rate Decay"):
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title(title)
    plt.xlabel('Steps/Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()