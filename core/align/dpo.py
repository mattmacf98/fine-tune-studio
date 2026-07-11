import torch


def dpo_loss(policy_logps, ref_logps, beta) -> torch.Tensor:
    """Implemented in W9."""