import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, vector_size=1024, M=512, L=256, dropout=0.6, n_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(vector_size, M),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_net = nn.Sequential(
            nn.Linear(M, L),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(L, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(M, n_classes)
        )

    def forward(self, bag):
        # bag: [K, D] or [B, K, D]
        if bag.dim() == 2:
            bag = bag.unsqueeze(0)  # -> [1, K, D]

        B, K, D = bag.shape

        H = self.feature_extractor(bag.reshape(B * K, D)).reshape(B, K, -1)  # [B, K, M]
        S = self.attention_net(H.reshape(B * K, -1)).reshape(B, K, 1)        # [B, K, 1]
        A = torch.softmax(S.transpose(1, 2), dim=-1)                          # [B, 1, K]

        Z = torch.bmm(A, H)                                                   # [B, 1, M]
        logits = self.classifier(Z.squeeze(1))                                # [B, n_classes]
        return logits, A