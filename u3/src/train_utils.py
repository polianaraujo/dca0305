# Funções de treino que retornam losses e gradients

import torch
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Loop de treino padrão."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Loop de validação."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

def capture_optimizer_internals(model, optimizer, layer_name='conv1'):
    """
    Retorna (gradiente_cru, ewma, gradiente_adaptado) para uma camada específica.
    Útil para Adam.
    """
    target_param = None
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            target_param = param
            break
            
    if target_param is None or target_param.grad is None:
        return None, None, None

    # 1. Gradiente Cru
    raw_grad = target_param.grad.data.cpu().numpy().flatten()
    
    # Acessa estado interno (só existe após o primeiro step)
    state = optimizer.state[target_param]
    if len(state) == 0:
        return raw_grad, np.zeros_like(raw_grad), np.zeros_like(raw_grad)
        
    # 2. EWMA (exp_avg no Adam)
    exp_avg = state['exp_avg'].cpu().numpy().flatten()
    
    # 3. Adaptado (exp_avg / (sqrt(exp_avg_sq) + eps))
    exp_avg_sq = state['exp_avg_sq'].cpu().numpy().flatten()
    denom = np.sqrt(exp_avg_sq) + optimizer.param_groups[0]['eps']
    adapted_grad = exp_avg / denom
    
    return raw_grad, exp_avg, adapted_grad