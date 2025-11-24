import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Attention(nn.Module):
#     def __init__(self, vector_size=1024, M=512, L=256, attention_branches=1, dropout=0.5):
#         super(Attention, self).__init__()
#         self.M = M
#         self.L = L
#         self.ATTENTION_BRANCHES = attention_branches

#         self.feature_extractor = nn.Sequential(
#             nn.Linear(vector_size, self.M),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         self.attention = nn.Sequential(
#             nn.Linear(self.M, self.L),  # matrix V
#             nn.Tanh(),
#             nn.Dropout(dropout),
#             nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
#         )

#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(self.M * self.ATTENTION_BRANCHES, 2),
#             # nn.Sigmoid()
#         )

    
#     def forward(self, bag):
#         """
#         Forward pass for the Attention model.
#         Args:
#             bag (torch.Tensor): Input bag tensor of shape [K, vector_size].
#         Returns:
#             Y_prob (torch.Tensor): Predicted probabilities.
#             Y_hat (torch.Tensor): Predicted labels (0 or 1).
#             A (torch.Tensor): Attention weights.
#         """
#         # Ensure bag is 2D (remove batch dimension if present)
#         if bag.dim() == 3:
#             bag = bag.squeeze(0)  # Shape becomes [K, vector_size]

#         # Feature extraction
#         H = self.feature_extractor(bag)  # Shape [K, M]

#         # Attention weights
#         A = self.attention(H)  # Shape [K, ATTENTION_BRANCHES]
#         A = A.transpose(1, 0)  # Shape [ATTENTION_BRANCHES, K]
#         A = F.softmax(A, dim=1)  # Softmax over K (instances in the bag)

#         # Weighted feature aggregation
#         Z = torch.mm(A, H)  # Shape [ATTENTION_BRANCHES, M]
#         # Z = torch.matmul(A, H)  # Shape [ATTENTION_BRANCHES, M]
#         # Z = Z.reshape(1, -1)

#         # Classification
#         logits = self.classifier(Z)  # Shape [ATTENTION_BRANCHES, 1]
#         return logits, A                  # or return logits, A, H if you like
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    ABMIL following Breen et al. architecture:

    Feature extractor: Linear(vector_size → M) + ReLU
    Attention: Linear(M → L) + Tanh + Linear(L → 1)
    Classifier: Linear(M → 2)
    """

    def __init__(self, vector_size=1024, M=512, L=256, dropout=0.6):
        super().__init__()

        # ---- Feature extractor  ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(vector_size, M),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- Attention network ----
        self.attention_net = nn.Sequential(
            nn.Linear(M, L),     # L = hidden
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(L, 1)      # single-branch attention
        )

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(M, 2)      # binary classifier
        )


    def forward(self, bag):
        """
        bag: [K, vector_size]
        """
        if bag.dim() == 3:
            bag = bag.squeeze(0)

        # Feature extraction
        H = self.feature_extractor(bag)     # [K, M]

        # Attention
        A = self.attention_net(H)           # [K, 1]
        A = A.transpose(1, 0)               # [1, K]
        A = F.softmax(A, dim=1)             # softmax over instances

        # MIL Pooling
        Z = torch.mm(A, H)                  # [1, M]

        # Classification
        logits = self.classifier(Z)         # [1, 2]

        return logits, A
