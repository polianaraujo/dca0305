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
    
def plot_gradients_distribution_with_sma(raw, sma, ewma, adapted):
    """
    Plota 4 histogramas: Raw, SMA (Simulado), EWMA (Real) e Adapted (Real).
    """
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    
    # Configurações visuais comuns
    kwargs = {'bins': 50, 'kde': True, 'stat': 'density', 'alpha': 0.6, 'edgecolor': None}

    # 1. Gradientes Crus
    sns.histplot(raw, ax=ax[0], color='gray', **kwargs)
    ax[0].set_title("1. Gradientes Crus (Raw)")
    ax[0].set_xlabel("Magnitude")
    
    # 2. Média Móvel Simples (SMA) - Simulada
    sns.histplot(sma, ax=ax[1], color='orange', **kwargs)
    ax[1].set_title("2. Média Móvel Simples (SMA-10)")
    ax[1].set_xlabel("Magnitude Média")
    # Força a mesma escala do EWMA para comparação justa
    ax[1].set_xlim(ax[2].get_xlim()) 
    
    # 3. EWMA (Momentum do Adam) - Real
    sns.histplot(ewma, ax=ax[2], color='blue', **kwargs)
    ax[2].set_title("3. EWMA (Momentum Real)")
    ax[2].set_xlabel("Magnitude Suavizada")
    
    # 4. Gradientes Adaptados (Adam) - Real
    sns.histplot(adapted, ax=ax[3], color='green', **kwargs)
    ax[3].set_title("4. Gradientes Adaptados (Final Step)")
    ax[3].set_xlabel("Magnitude Normalizada")
    
    plt.tight_layout()
    plt.show()