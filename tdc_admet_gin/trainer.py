import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from molfeat.trans.pretrained import PretrainedDGLTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, auc, precision_recall_curve
from tdc.benchmark_group import admet_group
import numpy as np
import torch.nn as nn
import wandb
import random

class SMILESFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        node_features = torch.tensor(feature, dtype=torch.float)
        return node_features, torch.tensor(label, dtype=torch.float)

class FineTuningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.4):
        super(FineTuningModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, node_feats):
        x = self.mlp(node_feats)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class BalancedBCELoss(nn.Module):
    def __init__(self):
        super(BalancedBCELoss, self).__init__()

    def forward(self, inputs, targets):
        positive_count = (targets == 1).sum().float()
        negative_count = (targets == 0).sum().float()
        
        pos_weight = negative_count / (positive_count + 1e-6) 

        weight_tensor = torch.where(targets == 1, pos_weight, torch.tensor(1.0, device=targets.device))

        loss = F.binary_cross_entropy(inputs, targets, weight=weight_tensor, reduction='mean')
        return loss

def train():
    wandb.init()
    config = wandb.config

    group = admet_group(path='data/')
    benchmark = group.get('cyp2c9_veith')
    train_val, test = benchmark['train_val'], benchmark['test']
    train_data, valid_data = train_test_split(
            train_val, test_size=0.2, random_state=7, stratify=train_val['Y']
        )
    test_data = test

    train_smiles = train_data['Drug'].tolist()
    valid_smiles = valid_data['Drug'].tolist()
    test_smiles = test_data['Drug'].tolist()

    transformer = PretrainedDGLTransformer(kind=config.transformer_kind, dtype=float)

    train_features = transformer(train_smiles)
    valid_features = transformer(valid_smiles)
    test_features = transformer(test_smiles)

    print("Train Features:", train_features.shape)
    print("Valid Features:", valid_features.shape)
    print("Test Features :", test_features.shape)

    train_dataset = SMILESFeatureDataset(train_features, train_data['Y'].tolist())
    valid_dataset = SMILESFeatureDataset(valid_features, valid_data['Y'].tolist())
    test_dataset = SMILESFeatureDataset(test_features, test_data['Y'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    input_dim = train_features.shape[1]
    model = FineTuningModel(input_dim=input_dim, 
                            hidden_dim=config.hidden_dim, 
                            num_layers=config.num_layers)

    if config.loss == "BCE":
        criterion = nn.BCELoss()
    elif config.loss == "focal":
        criterion = FocalLoss()
    elif config.loss == "balanced_bce":
        criterion = BalancedBCELoss()
    else:
        criterion = nn.BCELoss()

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_epochs = 200
    patience = 10  
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for batch in train_loader:
            node_features, labels = batch
            node_features, labels = node_features.to(device), labels.to(device)

            outputs = model(node_features).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            preds = (outputs.detach().cpu().numpy() > 0.5).astype(int)
            all_train_preds.extend(preds)
            all_train_targets.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(all_train_targets, all_train_preds)

        model.eval()
        total_val_loss = 0
        y_true_val = []
        y_pred_val = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in valid_loader:
                node_features, labels = batch
                node_features, labels = node_features.to(device), labels.to(device)

                outputs = model(node_features).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                val_loss = criterion(outputs, labels)

                total_val_loss += val_loss.item()
                y_true_val.extend(labels.cpu().numpy())

                preds = (outputs.detach().cpu().numpy() > 0.5).astype(int)
                all_val_preds.extend(preds)
                all_val_targets.extend(labels.cpu().numpy())

                if outputs.dim() == 0:
                    y_pred_val.append(outputs.cpu().numpy())
                else:
                    y_pred_val.extend(outputs.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / len(valid_loader)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        ap_score = average_precision_score(y_true_val, y_pred_val)
        precision, recall, _ = precision_recall_curve(y_true_val, y_pred_val)
        auprc = auc(recall, precision)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Acc: {train_acc:.4f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val AUPRC: {auprc:.4f}")

        wandb.log({
            'epoch': epoch+1,
            'train_acc': train_acc,
            'train_loss': avg_train_loss,
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'val_ap': ap_score,
            'val_auprc': auprc
        })

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_path = f".ckpt/best_model_cyp2c9_veith_{config.transformer_kind}_{config.hidden_dim}_{config.num_layers}_{config.optimizer}_{config.lr}_{config.loss}.pth"
            torch.save(model.state_dict(), best_model_path)
            print("  >>> Best model saved (val_loss improved).")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} with best val_loss: {best_loss:.4f}")
            break

    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_true_test = []
    y_pred_test = []

    with torch.no_grad():
        print("Testing with the best model...")
        for batch in test_loader:
            node_features, labels = batch
            node_features, labels = node_features.to(device), labels.to(device)

            outputs = model(node_features).squeeze()
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(outputs.cpu().numpy())

    test_preds = (np.array(y_pred_test) > 0.5).astype(int)
    test_acc = accuracy_score(y_true_test, test_preds)

    ap_score_test = average_precision_score(y_true_test, y_pred_test)
    precision_test, recall_test, _ = precision_recall_curve(y_true_test, y_pred_test)
    auprc_test = auc(recall_test, precision_test)

    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Average Precision Score: {ap_score_test:.4f}")
    print(f"Test AUPRC Score: {auprc_test:.4f}")

    wandb.log({
        'test_acc': test_acc,
        'test_ap': ap_score_test,
        'test_auprc': auprc_test
    })

    wandb.finish()

if __name__ == "__main__":
    train()