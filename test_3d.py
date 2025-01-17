import logging as log
import os
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import SchNet
import torch.nn.functional as F
from tdc.single_pred import ADME
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def setup_logging():
    log.basicConfig(level=log.INFO)
    log.getLogger("torch_geometric").setLevel(log.WARNING)

def _attempt_chirality_flip(mol):
    """
    Attempt to fix impossible stereochemistry by flipping chiral centers one-by-one.
    Returns a hydrogen-added RDKit Mol after trying different chirality flips.
    """
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    if not chiral_centers:
        return Chem.AddHs(mol)

    for (atom_idx, chirality) in chiral_centers:
        mol_copy = Chem.Mol(mol)
        atom = mol_copy.GetAtomWithIdx(atom_idx)
        atom.InvertChirality()
        molH = Chem.AddHs(mol_copy)

        try:
            if AllChem.EmbedMolecule(molH, maxAttempts=10) != -1:
                return molH
        except Exception as e:
            continue

    return Chem.AddHs(mol)

def generate_3D_coordinates(df, cache_file='molecule_coordinates_cache.csv'):
    """
    Generate 3D/2D coordinates for molecules with caching.
    If coordinates are already cached, reuse them without generating new data.
    """
    if os.path.exists(cache_file):
        log.info(f"Cache file {cache_file} found. Loading cached coordinates...")
        cached_df = pd.read_csv(cache_file)
        cached_df = cached_df.dropna(subset=['pos', 'z'])  # 유효한 데이터만 사용
        molecules = []
        for _, row in cached_df.iterrows():
            pos = torch.tensor(eval(row['pos']), dtype=torch.float)
            z = torch.tensor(eval(row['z']), dtype=torch.long)
            y = torch.tensor([row['y']], dtype=torch.float)
            molecules.append(Data(pos=pos, z=z, y=y))
        return molecules, df  # Cache 사용 시 원본 df를 그대로 반환

    # If no cache, generate coordinates
    log.info(f"Cache file {cache_file} not found. Generating new coordinates...")
    molecules = []
    failed_smiles = []
    new_data = []

    for smiles in df['Drug']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                # Try generating 3D coordinates
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if result == -1:  # 3D embedding failed
                    log.warning(f"3D embedding failed for SMILES: {smiles}, attempting chirality flip.")
                    mol = _attempt_chirality_flip(mol)  # Attempt chirality flip
                    result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Retry 3D embedding
                if result == -1:  # Still failed
                    log.warning(f"3D embedding still failed for SMILES: {smiles}, falling back to 2D coordinates.")
                    AllChem.Compute2DCoords(mol)  # Fallback to 2D coordinates

                conf = mol.GetConformer()
                coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                y = df.loc[df['Drug'] == smiles, 'Y'].values[0]
                
                molecules.append(Data(
                    pos=torch.tensor(coords, dtype=torch.float),
                    z=torch.tensor(atomic_numbers, dtype=torch.long),
                    y=torch.tensor([y], dtype=torch.float)
                ))
                new_data.append({'Drug': smiles, 'pos': coords.tolist(), 'z': atomic_numbers, 'y': y})
            except Exception as e:
                log.error(f"Failed to process SMILES: {smiles}. Error: {str(e)}")
                failed_smiles.append(smiles)
        else:
            log.warning(f"Invalid SMILES: {smiles}")
            failed_smiles.append(smiles)

    # Save new data to cache
    if new_data:
        new_df = pd.DataFrame(new_data)
        new_df.to_csv(cache_file, index=False)
        log.info(f"New coordinates saved to {cache_file}")

    valid_df = df[~df['Drug'].isin(failed_smiles)]
    return molecules, valid_df

def get_data():
    split = 'scaffold'
    data = ADME(name='CYP2C9_Veith')
    split_data = data.get_split(method=split)

    train_data, valid_data, test_data = split_data['train'], split_data['valid'], split_data['test']

    return train_data, valid_data, test_data

def collate_fn_3D(batch):
    batch = [b for b in batch if b is not None]  # None 데이터 필터링
    coords = [item[0] for item in batch]  # 각 분자의 좌표
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)  # 라벨
    return coords, labels

def train_schnet(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.z, data.pos, data.batch).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_schnet(model, loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    probabilities = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.z, data.pos, data.batch).squeeze()
            loss = F.binary_cross_entropy_with_logits(output, data.y)
            total_loss += loss.item()
            probabilities.append(torch.sigmoid(output).cpu().numpy())
            predictions.append((torch.sigmoid(output).cpu().numpy() > 0.5).astype(int))
            targets.append(data.y.cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    accuracy = accuracy_score(targets, predictions)

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    auprc = auc(recall, precision)

    return total_loss / len(loader), accuracy, auprc

def load_pretrained_schnet(model, checkpoint_path):
    """
    사전 학습된 가중치를 SchNet 모델에 로드하는 함수.

    Args:
        model (torch.nn.Module): SchNet 모델
        checkpoint_path (str): 사전 학습된 모델 가중치 파일 경로 (.pt)
    
    Returns:
        model (torch.nn.Module): 가중치가 로드된 SchNet 모델
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load('./schnet_20M.pt', map_location=torch.device('cpu'))  # CPU에서 로드
        if 'state_dict' in checkpoint:  # 체크포인트에 'state_dict'가 포함된 경우
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint  # 직접 state_dict로 사용
        model.load_state_dict(state_dict, strict=False)  # strict=False로 누락된 키 무시
        log.info(f"Loaded pretrained model weights from {checkpoint_path}")
    else:
        log.error(f"Checkpoint path {checkpoint_path} does not exist.")
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
    return model


def main():
    setup_logging()

    # Load data
    train_df, valid_df, test_df = get_data()

    # DataFrame에서 3D 좌표 데이터 생성
    train_data_list, train_df = generate_3D_coordinates(train_df)
    valid_data_list, valid_df = generate_3D_coordinates(valid_df)
    test_data_list, test_df = generate_3D_coordinates(test_df)

    # Set parameters
    project_name = "SchNet_Pretrained"
    output_path = './SchNet_Pretrained_Output'
    model_name = 'schnet_model'
    model_folder = os.path.join(output_path, model_name)
    os.makedirs(model_folder, exist_ok=True)

    epochs = 200
    batch_size = 256
    patience = 10
    learning_rate = 1e-4
    manual_seed = 112

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoaders
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_3D)
    valid_loader = DataLoader(valid_data_list, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_3D)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_3D)

    # Initialize SchNet model
    model = SchNet(hidden_channels=128, num_filters=128, num_interactions=3, cutoff=10.0, num_gaussians=50).to(device)

    # Load pretrained weights
    pretrained_path = './schnet_20M.pt'  # 사전 학습된 모델 경로
    model = load_pretrained_schnet(model, pretrained_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_schnet(model, train_loader, optimizer, device)
        valid_loss, valid_accuracy, valid_auprc = evaluate_schnet(model, valid_loader, device)

        log.info(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid AUPRC: {valid_auprc:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info("Early stopping triggered.")
            break

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pth')))

    # Evaluate on test set
    test_loss, test_accuracy, test_auprc = evaluate_schnet(model, test_loader, device)

    log.info(f"Test Loss: {test_loss:.4f}")
    log.info(f"Test Accuracy: {test_accuracy:.4f}")
    log.info(f"Test AUPRC: {test_auprc:.4f}")
    log.info(f"Valid Loss: {valid_loss:.4f}")
    log.info(f"Valid Accuracy: {valid_accuracy:.4f}")
    log.info(f"Valid AUPRC: {valid_auprc:.4f}")

if __name__ == '__main__':
    main()