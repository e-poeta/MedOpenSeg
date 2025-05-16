import torch
import torch.nn.functional as F

def compute_prototype_pull_loss(embeddings, labels, memory_bank, alpha):
    """
    Compute a "pull" loss that draws each voxel's embedding toward its class prototype.

    Args:
        embeddings: [B, F, D, H, W] final-layer embeddings for each voxel (batch dimension B).
        labels:     [B, D, H, W] integer segmentation labels, e.g. 0=background, 1=class1, ...
        memory_bank: memory bank storing {class_id: prototype (shape [F])}.
        alpha: the dynamic scaling factor for the pull loss this epoch.

    Returns:
        pull_loss: a scalar Tensor for the "pull" portion of the objective (already scaled by current_alpha).
    """
    device = embeddings.device
    B, F, D, H, W = embeddings.shape
    pull_loss_val = torch.zeros((), device=device)

    for b_idx in range(B):
        unique_labels = labels[b_idx].unique()
        for cls_val in unique_labels:
            cval = cls_val.item()
            if cval < 0:
                continue  # skip invalid or unknown labels if any
            if cval not in memory_bank.prototypes:
                continue
            mask = (labels[b_idx] == cval)  # shape [D,H,W], bool
            if mask.sum() == 0:
                continue

            proto = memory_bank.prototypes[cval]  # shape [F]
            emb_class = embeddings[b_idx][:, mask]  # shape [F, #voxels_in_class]

            diff = emb_class - proto.unsqueeze(-1)  # shape [F, #voxels]
            dist_sq = (diff ** 2).sum(dim=0)         # shape [#voxels]
            class_loss = dist_sq.mean()

            pull_loss_val += class_loss

    pull_loss_val = alpha * pull_loss_val
    return pull_loss_val

def get_warmup_alpha(epoch, warmup_epochs=10, alpha_final=0.05, alpha_start=0.001):
    """
    Return an alpha that linearly ramps from alpha_start in epoch=1
    up to alpha_final by epoch=warmup_epochs, then stays constant.
    """
    if epoch >= warmup_epochs:
        return alpha_final
    # linear scale
    progress = (epoch - 1) / (warmup_epochs - 1)  # from 0 to 1 across warmup epochs
    current_alpha = alpha_start + progress * (alpha_final - alpha_start)
    return current_alpha


def contrastive_loss(embeddings, labels, tau=0.1):
    """
    Memory-efficient contrastive loss to push unknowns away from known classes.

    Args:
        embeddings (Tensor): Feature embeddings of shape [B, F, D, H, W].
        labels (Tensor): Corresponding labels of shape [B, D, H, W].
        tau (float): Temperature scaling factor.

    Returns:
        Tensor: Contrastive loss value.
    """
    B, F_dim, D, H, W = embeddings.shape
    embeddings = embeddings.view(B, F_dim, -1)  # Flatten spatial dimensions [B, F, D*H*W]
    labels = labels.view(B, -1)  # Flatten labels [B, D*H*W]

    loss = 0.0
    for b in range(B):
        unique_labels = labels[b].unique()
        
        for lbl in unique_labels:
            mask = (labels[b] == lbl)  # Get binary mask for the class
            if mask.sum() == 0:
                continue  # Skip if no voxels belong to this label
            
            # Extract positive embeddings efficiently
            pos_emb = embeddings[b][:, mask]  # [F, num_pos_voxels]

            # Compute cosine similarity **efficiently** using batched dot product
            emb_norm = F.normalize(embeddings[b], dim=0)  # Normalize all embeddings
            pos_norm = F.normalize(pos_emb, dim=0)  # Normalize positive embeddings
            dist = torch.matmul(pos_norm.T, emb_norm)  # [num_pos_voxels, D*H*W]

            # Contrastive loss using numerically stable `logsumexp`
            pos_loss = torch.logsumexp(dist / tau, dim=1).mean()  # Positive pairs
            neg_loss = torch.logsumexp((1 - dist) / tau, dim=1).mean()  # Negative pairs
            
            loss += -(pos_loss - neg_loss)  # Minimize positive loss, maximize negative separation

    return loss / B  # Normalize by batch size



def compute_dce_and_vl_loss(embeddings, labels, prototypes, device='cuda',lambda_vl=0.01, ignore_index=999):
    """
    Compute the hybrid loss (DCE + VL) for 3D segmentation when prototypes are given as a dict.
    
    Args:
        embeddings: Tensor of shape [B, d, D, H, W] (3D embeddings).
        labels:     Tensor of shape [B, D, H, W] with integer class labels.
        prototypes_dict: Dictionary mapping class labels (e.g., 3.0, 2.0, etc.) to a 1D prototype tensor of shape [d].
        lambda_vl:  Weight factor for the variance (attractive) loss.
        ignore_index: Label to ignore (e.g., -1), and also pixels whose label is not in prototypes_dict.
        
    Returns:
        hybrid_loss: Combined loss (DCE loss + lambda_vl * VL loss).
        dce_loss:    Discriminative cross entropy loss.
        vl_loss:     Variance loss.
    """
    device = embeddings.device
    B, d, D, H, W = embeddings.shape
    N = D * H * W  # total number of voxels per sample

    # Flatten embeddings and labels:
    embeddings_flat = embeddings.view(B, d, N)  
    embeddings_flat = F.normalize(embeddings_flat, p=2, dim=1)# [B, d, N]
    labels_flat = labels.view(B, N)                    # [B, N]

    # Build a sorted list of prototypes and a mapping from label -> index.
    sorted_keys = sorted(prototypes.keys(), key=lambda x: float(x))
    proto_list = [F.normalize(prototypes[k].to(device), p=2, dim=0) for k in sorted_keys]
    prototypes_tensor = torch.stack(proto_list, dim=0)   # [num_classes, d]
    label_to_idx = {k: i for i, k in enumerate(sorted_keys)}
    num_classes = prototypes_tensor.shape[0]

    # Remap ground-truth labels to prototype indices.
    target_indices = torch.full_like(labels_flat, fill_value=ignore_index, dtype=torch.long)
    for label_val, idx in label_to_idx.items():
        mask = (labels_flat == label_val)
        target_indices[mask] = idx

    # Create a valid mask (ignore pixels with ignore_index).
    valid_mask = (target_indices != ignore_index).float()  # [B, N]

    # Instead of expanding to [B, num_classes, d, N], use the dot-product trick:
    # Let X be embeddings_flat [B, d, N] and P be prototypes_tensor [num_classes, d].
    # Then, for each voxel: ||x - m||^2 = ||x||^2 + ||m||^2 - 2 * x^T m.
    norm_x = torch.sum(embeddings_flat ** 2, dim=1)         # [B, N]
    norm_p = torch.sum(prototypes_tensor ** 2, dim=1)         # [num_classes]
    # Compute the cross term using einsum: result shape [B, num_classes, N]
    cross = torch.einsum("bdn,cd->bcn", embeddings_flat, prototypes_tensor)
    # Compute squared distances: shape [B, num_classes, N]
    dist_sq = norm_x.unsqueeze(1) + norm_p.unsqueeze(0).unsqueeze(-1) - 2 * cross

    # Compute logits as negative squared distances.
    logits = -dist_sq
    # Apply softmax over the class dimension.
    probs = F.softmax(logits, dim=1)  # [B, num_classes, N]

    # Gather the probability for the ground-truth class.
    target_indices_exp = target_indices.unsqueeze(1)       # [B, 1, N]
    true_probs = torch.gather(probs, 1, target_indices_exp).squeeze(1)  # [B, N]

    # Compute Discriminative Cross Entropy (DCE) Loss.
    dce_loss = -torch.log(true_probs + 1e-8)
    dce_loss = (dce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    # Compute Variance Loss (VL): average squared distance for the correct class.
    true_dist_sq = torch.gather(dist_sq, 1, target_indices_exp).squeeze(1)  # [B, N]
    vl_loss = (true_dist_sq * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    hybrid_loss = dce_loss + lambda_vl * vl_loss

    return hybrid_loss, dce_loss, vl_loss