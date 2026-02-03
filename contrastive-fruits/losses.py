import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5, eps=1e-8):
    """Normalized temperature-scaled cross entropy loss (NT-Xent) for one pair of batches.
    z1, z2: tensors of shape (B, D) already normalized.
    Returns the scalar loss.
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D

    # cosine similarity matrix
    sim = torch.matmul(z, z.t())  # 2B x 2B

    # for numerical stability
    sim = sim / temperature

    # mask out self-similarities
    mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)).float()

    # numerator: positive pairs are (i, i+B) and (i+B, i)
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)])

    # denominator: sum over all except self
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1)

    loss = -torch.log(torch.exp(positives) / (denom + eps))
    loss = loss.mean()
    return loss


def combined_counterfactual_loss(z1, z2, z_cf, alpha=1.0, temperature=0.5):
    """Compute contrastive loss between z1,z2 and z1,z_cf and return weighted sum.
    z_cf is the counterfactual projection for the same batch (same order as z1).
    """
    l_pos = nt_xent_loss(z1, z2, temperature=temperature)
    l_cf = nt_xent_loss(z1, z_cf, temperature=temperature)
    return l_pos + alpha * l_cf
