"""
unimodal/unimol3d.py
"""
import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from unimol_tools import UniMolRepr

# unimol_tools 라이브러리의 INFO 로그 억제 - log 심함;;
logging.getLogger("unimol_tools").setLevel(logging.WARNING)

class UniMol_3D(nn.Module):

    def __init__(self, 
                 transformer_model='unimolv1', 
                 model_size='84m', 
                 num_classes=2,
                 hidden_dim=256,  
                 dropout_rate=0.4,
                 max_atomic_len=None):   
        super(UniMol_3D, self).__init__()

        self.unimol = UniMolRepr(
            data_type='molecule', 
            remove_hs=False,
            model_name=transformer_model, 
            model_size=model_size
        )

        self.cls_embedding_dim = 512
        self.atomic_embedding_dim = 512

        self.max_atomic_len = max_atomic_len

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.atomic_embedding_dim, num_heads=4, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_combined = nn.Sequential(
            nn.Linear(self.cls_embedding_dim + self.atomic_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, smiles_list):
        reprs = self.unimol.get_repr(smiles_list, return_atomic_reprs=True)
        cls_embeddings = torch.tensor(reprs['cls_repr'], dtype=torch.float32)

        atomic_emb_list = [torch.tensor(atom_repr, dtype=torch.float32) for atom_repr in reprs['atomic_reprs']]
        atomic_embeddings = pad_sequence(atomic_emb_list, batch_first=True)

        device = self.mlp_combined[0].weight.device
        cls_embeddings = cls_embeddings.to(device)
        atomic_embeddings = atomic_embeddings.to(device)

        if self.max_atomic_len:
            if atomic_embeddings.size(1) < self.max_atomic_len:
                padding = torch.zeros(
                    atomic_embeddings.size(0),
                    self.max_atomic_len - atomic_embeddings.size(1),
                    self.atomic_embedding_dim,
                    device=device
                )
                atomic_embeddings = torch.cat((atomic_embeddings, padding), dim=1)
            elif atomic_embeddings.size(1) > self.max_atomic_len:
                atomic_embeddings = atomic_embeddings[:, :self.max_atomic_len, :]

        attn_output, _ = self.attention_layer(atomic_embeddings, atomic_embeddings, atomic_embeddings)
        atomic_summary = attn_output.mean(dim=1)  

        combined_embeddings = torch.cat((cls_embeddings, atomic_summary), dim=1)

        x = self.dropout(combined_embeddings)
        logits = self.mlp_combined(x)

        return logits