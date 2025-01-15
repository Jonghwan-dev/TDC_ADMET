import torch
from torch.utils.data import Dataset, DataLoader
from molfeat.trans.pretrained import PretrainedDGLTransformer
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
from tdc.single_pred import ADME
from tdc.single_pred import Tox
import numpy as np
from dgllife.model import load_pretrained
import torch.nn as nn

# 데이터 로드
split = 'scaffold'
data = ADME(name='cyp2c9_veith')
# data = Tox(name='herg')
split_data = data.get_split(method=split)

train_data, valid_data, test_data = split_data['train'], split_data['valid'], split_data['test']

# 'Drug' 컬럼에 SMILES 데이터를 가져오기
train_smiles = train_data['Drug'].tolist()
valid_smiles = valid_data['Drug'].tolist()
test_smiles = test_data['Drug'].tolist()

# SMILES 데이터가 모델의 입력 형식에 맞게 변환되도록 transformer 설정
transformer = PretrainedDGLTransformer(kind='gin_supervised_contextpred', dtype=float)

# SMILES 데이터를 변환하여 features로 저장
train_features = transformer(train_smiles)
valid_features = transformer(valid_smiles)
test_features = transformer(test_smiles)

# 확인을 위한 출력
print("Train Features:", train_features.shape)
print("Valid Features:", valid_features.shape)
print("Test Features :", test_features.shape)

# Custom Dataset Class
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

# Dataset 및 DataLoader 생성
train_dataset = SMILESFeatureDataset(train_features, train_data['Y'].tolist())
valid_dataset = SMILESFeatureDataset(valid_features, valid_data['Y'].tolist())
test_dataset = SMILESFeatureDataset(test_features, test_data['Y'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Fine-Tuning 모델 정의
class FineTuningModel(nn.Module):
    def __init__(self, pretrained_model, input_dim, hidden_dim=256, num_layers=3, dropout=0.4):
        super(FineTuningModel, self).__init__()
        self.pretrained = pretrained_model
        for param in self.pretrained.parameters():
            param.requires_grad = False  # Pretrained 모델 freeze

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # 이진 분류를 위한 Sigmoid 출력

        self.mlp = nn.Sequential(*layers)

    def forward(self, node_feats):
        # embeddings = self.pretrained(node_feats)  # 필요 시 활용
        x = self.mlp(node_feats)
        return x

# Pretrained 모델 로드
pretrained_model = load_pretrained('gin_supervised_contextpred')

# Fine-tuning 모델
input_dim = train_features.shape[1]
model = FineTuningModel(pretrained_model=pretrained_model, input_dim=input_dim)

# 손실 함수 및 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# >>> 스케줄러 추가 (ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

device = "cuda"
model.to(device)

# 학습 루프 정의
num_epochs = 200
patience = 10  # EarlyStopping patience
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    # -----------------
    # 1) Training step
    # -----------------
    for batch in train_loader:
        node_features, labels = batch
        node_features, labels = node_features.to(device), labels.to(device)

        outputs = model(node_features).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # -----------------
    # 2) Validation step
    # -----------------
    model.eval()
    total_val_loss = 0
    y_true_val = []
    y_pred_val = []

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

            if outputs.dim() == 0:
                y_pred_val.append(outputs.cpu().numpy())
            else:
                y_pred_val.extend(outputs.cpu().numpy().tolist())

    avg_val_loss = total_val_loss / len(valid_loader)

    # 메트릭 (AP, AUPRC) 계산
    ap_score = average_precision_score(y_true_val, y_pred_val)
    precision, recall, _ = precision_recall_curve(y_true_val, y_pred_val)
    auprc = auc(recall, precision)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val AP: {ap_score:.4f} | "
          f"Val AUPRC: {auprc:.4f}")

    # >>> 스케줄러 스텝 (val_loss 이용)
    scheduler.step(avg_val_loss)

    # 3) Early Stopping 기준: Validation Loss 감소 여부
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model_cyp2c9_veith.pth")
        print("  >>> Best model saved (val_loss improved).")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} with best val_loss: {best_loss:.4f}")
        break

# -----------------------------------------------------
# Test with the best model
model.load_state_dict(torch.load("best_model_cyp2c9_veith.pth"))
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

# Test 지표 계산
ap_score_test = average_precision_score(y_true_test, y_pred_test)
precision_test, recall_test, _ = precision_recall_curve(y_true_test, y_pred_test)
auprc_test = auc(recall_test, precision_test)

print(f"Test Average Precision Score: {ap_score_test:.4f}")
print(f"Test AUPRC Score: {auprc_test:.4f}")