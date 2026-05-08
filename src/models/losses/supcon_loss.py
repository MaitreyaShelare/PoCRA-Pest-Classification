# """
# Supervised Contrastive Learning loss.
# Used jointly with ArcFace in Phase 1.
# Paper: https://arxiv.org/abs/2004.11362
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SupConLoss(nn.Module):
#     """
#     Supervised Contrastive Loss.
#     Treats samples from the same class as positives.
#     Maximizes agreement between different views of same class.
#     """
    
#     def __init__(self, temperature: float = 0.07, contrast_mode: str = "all") -> None:
#         """
#         Initialize SupCon loss.
        
#         Args:
#             temperature: Temperature for softmax (lower = sharper)
#             contrast_mode: "one" (single positive per sample) or "all" (all same class)
#         """
#         super().__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
    
#     def forward(
#         self,
#         features: torch.Tensor,
#         labels: torch.Tensor = None,
#         mask: torch.Tensor = None,
#     ) -> torch.Tensor:
#         """
#         Compute SupCon loss.
        
#         Args:
#             features: Features of shape (batch_size, feat_dim)
#                      or (batch_size, n_views, feat_dim) if multiple views
#             labels: Labels of shape (batch_size,)
#             mask: Mask of shape (batch_size, batch_size) for custom positive pairs
        
#         Returns:
#             Scalar loss
#         """
#         # device = (torch.device("cuda") if features.is_cuda else torch.device("cpu"))
#         device = features.device
        
#         # Handle 2D and 3D inputs
#         # if len(features.shape) < 3:
#         #     raise ValueError("Features should be (batch_size, n_views, feat_dim) or use external views")
        
#         # Expect single-view features: (B, D)
#         if len(features.shape) != 2:
#             raise ValueError(
#                 "Features should have shape (batch_size, feat_dim)"
#     )
#         batch_size = features.shape[0]
        
#         # Normalize features
#         features = features.float()
#         features = F.normalize(features, dim=-1)
        
#         # Compute similarity matrix
#         # similarity_matrix = torch.matmul(features, features.T)  # (B, B)

#         # features = features.squeeze(1)  # (B, 768)
#         similarity_matrix = torch.matmul(features, features.T)
        
#         # Diagonal mask: discard self-similarity
#         logits_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
#         # similarity_matrix.masked_fill_(logits_mask, -9e15)
#         similarity_matrix = similarity_matrix.masked_fill(
#             logits_mask,
#             -1e4,
#         )
        
#         # Create label mask: same label = positive
#         if labels is not None:
#             labels = labels.contiguous().view(-1, 1)  # (B, 1)
#             label_mask = torch.eq(labels, labels.T).float()  # (B, B)
#             label_mask.masked_fill_(logits_mask, 0)
#         elif mask is not None:
#             label_mask = mask.float()
#         else:
#             raise ValueError("Either labels or mask must be provided")
        
#         # Compute positives and negatives
#         pos_mask = label_mask
#         pos = (similarity_matrix * pos_mask).sum(dim=1, keepdim=True)
        
#         # Compute log probability
#         logits = similarity_matrix / self.temperature
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach()
        
#         exp_logits = torch.exp(logits)
#         log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
#         # Compute loss
#         loss = -(log_prob * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-12)
#         loss = loss.mean()
        
#         return loss

"""
Supervised Contrastive Learning loss.
Used jointly with ArcFace in Phase 1.
Paper: https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Treats samples from the same class as positives.
    Maximizes agreement between same-class embeddings.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
    ) -> None:
        """
        Initialize SupCon loss.

        Args:
            temperature: Softmax temperature
            contrast_mode: Unused currently
        """

        super().__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features:
                Tensor of shape (B, D)

            labels:
                Tensor of shape (B,)

            mask:
                Optional custom positive mask

        Returns:
            Scalar loss
        """

        device = features.device

        # Expect (B, D)
        if len(features.shape) != 2:
            raise ValueError(
                "Features should have shape "
                "(batch_size, feat_dim)"
            )

        batch_size = features.shape[0]

        # IMPORTANT:
        # Force FP32 for numerical stability
        features = features.float()

        # Normalize embeddings
        features = F.normalize(
            features,
            dim=1,
        )

        # Cosine similarity matrix
        logits = torch.matmul(
            features,
            features.T,
        )

        # Temperature scaling
        logits = logits / self.temperature

        # Numerical stabilization
        logits_max, _ = torch.max(
            logits,
            dim=1,
            keepdim=True,
        )

        logits = logits - logits_max.detach()

        # Mask self similarity
        logits_mask = torch.ones_like(logits)

        logits_mask.fill_diagonal_(0)

        # Positive mask
        if labels is not None:

            labels = labels.contiguous().view(-1, 1)

            pos_mask = torch.eq(
                labels,
                labels.T,
            ).float().to(device)

        elif mask is not None:

            pos_mask = mask.float().to(device)

        else:
            raise ValueError(
                "Either labels or mask must be provided"
            )

        # Remove self-comparisons
        pos_mask = pos_mask * logits_mask

        # exp(logits)
        exp_logits = (
            torch.exp(logits) * logits_mask
        )

        # log_prob
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True)
            + 1e-12
        )

        # Number of positives per sample
        pos_per_sample = pos_mask.sum(dim=1)

        # Avoid division by zero
        pos_per_sample = torch.clamp(
            pos_per_sample,
            min=1.0,
        )

        # Mean log probability over positives
        mean_log_prob_pos = (
            (pos_mask * log_prob).sum(dim=1)
            / pos_per_sample
        )

        # Final loss
        loss = -mean_log_prob_pos.mean()

        # Safety against NaNs/Infs
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.zeros(
                [],
                device=device,
                requires_grad=True,
            )

        return loss