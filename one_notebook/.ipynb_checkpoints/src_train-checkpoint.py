
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from src.utils import save_history, plot_metrics
from src.config import LEARNING_RATE, WEIGHT_DECAY, LR_STEP_SIZE, LR_GAMMA, NUM_EPOCHS, DEVICE, BEST_MODEL_PATH, EARLY_STOPPING_PATIENCE

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    model.to(DEVICE)
    history = {'train_loss': [], 'val_loss': []}
    early_stopping = EarlyStopping()
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break
    save_history(history)
    plot_metrics(history)
    return model
