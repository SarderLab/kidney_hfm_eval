import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    ABMIL Regression Version (Breen-style, Single Branch)

    Feature extractor: Linear(vector_size → M) + ReLU
    Attention:         Linear(M → L) + Tanh + Linear(L → 1)
    Pooling:           [1, M]
    Regression head:   Linear(M → 1)
    """

    def __init__(self, vector_size=1024, M=512, L=256, dropout=0.25):
        super().__init__()

        self.M = M
        self.L = L
        self.B = 1   # <<< SINGLE ATTENTION BRANCH (as you required)

        # ---- Feature extractor ----
        self.feature_extractor = nn.Sequential(
            nn.Linear(vector_size, M),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- Attention network ----
        self.attention_net = nn.Sequential(
            nn.Linear(M, L),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(L, 1)     # <<< SINGLE BRANCH ATTENTION
        )

        # ---- Regression head ----
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(M, 1)     # <<< REGRESSION OUTPUT (1-dim)
        )

    def forward(self, bag):
        """
        bag: [K, vector_size]
        returns:
            y_pred: [1,1]
            A: [1, K]
        """

        if bag.dim() == 3:
            bag = bag.squeeze(0)

        # Feature extraction
        H = self.feature_extractor(bag)       # [K, M]

        # Attention scores
        A = self.attention_net(H)             # [K, 1]
        A = A.transpose(1, 0)                 # [1, K]
        A = F.softmax(A, dim=1)               # weight over K instances

        # MIL pooling
        Z = torch.mm(A, H)                    # [1, M]

        # Regression output
        y_pred = self.regressor(Z)            # [1, 1]

        return y_pred, A