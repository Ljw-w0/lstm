import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return losses

def eval_model(model, test_loader, criterion, device):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0.0
    losses = []

    progress_bar = tqdm.tqdm(test_loader, desc='Evaluating', unit='batch')

    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

        val_loss = total_loss / len(test_loader)
        losses.append(val_loss)

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    return predictions, actuals, losses