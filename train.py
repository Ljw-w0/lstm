import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import mean_squared_error

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
            epoch_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses

def eval_model(model, test_loader, criterion, device, scaler=None):
    model.eval()
    predictions = []
    actuals = []

    progress_bar = tqdm.tqdm(test_loader, desc='Evaluating', unit='batch')

    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)    

            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        actuals = scaler.inverse_transform(actuals)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f'RMSE: {rmse:.4f}')

    return predictions, actuals