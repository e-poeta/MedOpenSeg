# memory_bank_voxelwise.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from adjustText import adjust_text 
    
class MemoryBankV(nn.Module):
    """
    Memory Bank for storing class prototypes in R^F, with voxel-level novelty detection.

    1) Each known class 'c' gets a single vector p_c in R^F (a prototype).
    2) During training, prototypes are updated via masked average of embeddings.
    3) During inference, we compare each voxel embedding to the known prototypes
       to get a voxel-level novelty map.
    """

    def __init__(
        self,
        feature_dim: int,
        memory_size: int = 100,
        similarity_threshold: float = 0.7,
        epoch_counter = 0,
        alpha: float = 0.9,
        save_path: str = "./prototypes"
    ):
        """
        Args:
            feature_dim: dimensionality F of the final embedding space (R^F).
            memory_size: max number of known classes to store.
            similarity_threshold: voxel-level novelty threshold for e.g. cosine similarity.
            alpha: exponential moving average factor for prototype updates.
            save_path: directory for saving / loading prototypes.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.epoch_counter = epoch_counter
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Dictionary of {class_id: prototype_vector in shape [feature_dim]}
        self.prototypes = {}
        UNKNOWN_CLASS_ID = 999

    def save_memory_bank(self,save_path):
        """
        Save current prototypes to disk.
        """
        if not self.prototypes:
            print("[WARNING] Memory bank is empty! Nothing to save.")
            return
        # Convert to CPU Tensors for saving
        proto_dict = {k: v.detach().cpu() for k, v in self.prototypes.items()}
        torch.save(proto_dict, save_path)
        print(f"[INFO] Prototypes saved: {len(proto_dict)} classes at {save_path}")

    def load_memory_bank(self, memory_bank_path, device="cuda"):
        """
        Load prototypes from disk.
        """
        if not os.path.exists(memory_bank_path):
            print(f"[ERROR] Cannot load memory bank, file not found: {memory_bank_path}")
            return
        memory_checkpoint = torch.load(memory_bank_path, map_location="cpu")
        self.prototypes = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in memory_checkpoint.items()
    }
        print(f"[INFO] Loaded {len(self.prototypes)} prototypes from {memory_bank_path}")
        
    def save_tsne_plot(self, perplexity=30.0, random_state=42):
        """
        Saves a t-SNE visualization of prototype embeddings in 2D.

        Args:
            perplexity (float): t-SNE perplexity parameter (20–50 is typical).
            random_state (int): for reproducible t-SNE results.
        """
        if len(self.prototypes) == 0:
            print("[WARNING] Memory Bank is EMPTY! No prototypes to visualize.")
            return

        # class_ids (keys) and embeddings (values)
        class_ids = sorted(self.prototypes.keys())
        embeddings = torch.stack([self.prototypes[c] for c in class_ids]).cpu().numpy()  # shape [num_classes, feature_dim]

        adjusted_perplexity = min(30, len(class_ids) - 1)  
        if len(class_ids) < 2:
            print("[WARNING] Not enough prototypes for t-SNE (need >=2). Skipping.")
            return

        # Create a t-SNE model and transform prototypes to 2D
        tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=random_state)
        embeddings_2d = tsne.fit_transform(embeddings)  # shape [num_classes, 2]

        # Create a color palette for plotting
        sns.set_style("whitegrid")
        sns.set_context("paper")
        colors = sns.color_palette("husl", n_colors=len(class_ids))
        plt.figure(figsize=(8, 6))

        # Plot each prototype in 2D
        texts = []
        for i, (x, y) in enumerate(embeddings_2d):
            label = class_ids[i]
            plt.scatter(x, y, color=colors[i], s=130, edgecolors='black', linewidth=0.5, alpha=0.7)
            text = plt.text(x, y-0.2, str(label), fontsize=12, ha='center', va='center', fontweight='bold', color='black')
            texts.append(text)

        # Optionally adjust text to avoid overlaps
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),only_move={'points':'y', 'text':'y'})

        plt.title("t-SNE of Prototype Embeddings", fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Dim 1", fontsize=12)
        plt.ylabel("t-SNE Dim 2", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.4)

        # Save to disk
        plot_filename = os.path.join(self.save_path,f"prototypes_epoch_{self.epoch_counter}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] t-SNE prototype visualization saved to: {plot_filename}")

    
    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor, device="cuda"):
        """
        Update prototypes with new embeddings for known classes.
        Typically called during training.

        Args:
            embeddings: [B, F, D, H, W], final-layer embeddings for each voxel.
            labels:     [B, D, H, W] integer labels for each voxel (0=background or ignore).
        """
        # Sanity checks
        UNKNOWN_CLASS_ID = 999 
        B, F_dim, D, H, W = embeddings.shape
        if labels.shape != (B, D, H, W):
            raise ValueError("labels must match embeddings' spatial shape except for channel dim")

        # For each sample in the batch
        for b_idx in range(B):
            label_map = labels[b_idx]  # shape [D, H, W]
            embed_map = embeddings[b_idx]  # shape [F, D, H, W]

            # Unique class IDs in this volume
            unique_cls = label_map.unique()
            for cls_id in unique_cls:
                cval = cls_id.item()
                # if cval == 0:
                #     # skip background or unknown label
                #     continue
                           # Special case: Force unknown regions to register as a class
                if cval > 10:
                    cval = UNKNOWN_CLASS_ID 

                # Create a mask for class cval
                # shape => [D, H, W], we want broadcast to [F, D, H, W]
                mask = (label_map == cval).unsqueeze(0).float()  # [1, D, H, W]
                # Weighted sum of embeddings
                sum_feat = (embed_map * mask).sum(dim=(-3, -2, -1))  # [F]
                count = mask.sum()

                if count > 0:
                    mean_feat = sum_feat / count
                    mean_feat = mean_feat.detach().clone().float() 

                    # Update memory using exponential moving average
                    if cval in self.prototypes:
                        old_proto = self.prototypes[cval]
                        new_proto = self.alpha * old_proto + (1.0 - self.alpha) * mean_feat
                        self.prototypes[cval] = new_proto.to(device)
                    else:
                        if len(self.prototypes) < self.memory_size:
                            self.prototypes[cval] = mean_feat.to(device)
                            print(f"[INFO] Added new prototype for class {cval}")
                        else:
                            print(f"[WARNING] Memory bank is full (size={self.memory_size}). Not storing class {cval}.")
                else:
                    print(f"[WARNING] No valid voxels for class {cval} in this batch element.")
        if self.epoch_counter % 10 == 0:
            self.save_tsne_plot(perplexity=30.0)

    def query_voxelwise_novelty(self, embedding_3d: torch.Tensor) -> torch.Tensor:
        """
        Compute voxel-level novelty mask by comparing each voxel's embedding
        to known-class prototypes using a simple similarity threshold.

        Args:
            embedding_3d: [F, D, H, W] or [B, F, D, H, W]

        Returns:
            novelty_mask: [D, H, W] or [B, D, H, W], 1=novel, 0=known
        """
        if not self.prototypes:
            print("[WARNING] No prototypes stored. Everything is novel!")
            return torch.ones_like(embedding_3d[...,0]).bool()  # shape [B, D, H, W] or [D,H,W]

        # For convenience, handle batch dimension
        if embedding_3d.ndim == 5:
            B, F_dim, D, H, W = embedding_3d.shape
        else:
            B = 1
            F_dim, D, H, W = embedding_3d.shape
            embedding_3d = embedding_3d.unsqueeze(0)  # -> [1, F, D, H, W]

        # Gather all prototypes [C, F]
        proto_list = []
        proto_labels = []
        
        for cval, proto_vec in self.prototypes.items():
            if not isinstance(proto_vec, torch.Tensor):
                proto_vec_t = torch.tensor(proto_vec, dtype=torch.float32)
            else:
                proto_vec_t = proto_vec.detach().clone().float()  # Clone to ensure materialization

            proto_vec_t = proto_vec_t.to(embedding_3d.device, non_blocking=True)
            proto_list.append(proto_vec_t)
            proto_labels.append(cval)  
             
        proto_matrix = torch.stack(proto_list, dim=0).to(embedding_3d.device, non_blocking=True)
        #print('proto_matrix:',proto_matrix)
        proto_matrix_norm = F.normalize(proto_matrix, dim=1)  # Normalize along feature dim


        # Flatten embeddings and do normalization to get cos sim => [B, F, D*H*W]
        emb_flat = embedding_3d.view(B, F_dim, -1)  # shape [B, F, D*H*W]
        emb_norm = F.normalize(emb_flat, dim=1)  # [B, F, D*H*W]
        # cos_sim => [B, C, D*H*W]
        # We do batched matrix multiplication or a simpler approach
        # matmul approach: if emb_norm => [B, D*H*W, F], proto => [F, C] => result => [B, D*H*W, C]
        emb_norm_t = emb_norm.permute(0, 2, 1)  # [B, D*H*W, F]
        proto_matrix_batch = proto_matrix_norm.unsqueeze(0).expand(B, -1, -1)
        proto_t = proto_matrix_batch.transpose(1, 2)        # [B, F, C]
        cos_sim = torch.bmm(emb_norm_t, proto_t)         # => [B, D*H*W, C]
        # For each voxel, we want max similarity among known prototypes
        # max_sim => [B, D*H*W]
        max_sim, _ = cos_sim.max(dim=2)
        # Reshape back => [B, D, H, W]
        max_sim_map = max_sim.view(B, D, H, W)
        # novelty => if max_sim < threshold => 1 else 0
        novelty_mask_sim = max_sim_map#(max_sim_map < self.similarity_threshold)
        
        # Compute Euclidean Distance Sum (EDS)
        eds = torch.cdist(emb_flat.permute(0, 2, 1), proto_matrix.unsqueeze(0), p=2).sum(dim=2)  # [B, D*H*W]
        novelty_mask_eds = eds.view(B, D, H, W)  # Reshape to [B, D, H, W]

        
        # If B=1, you might prefer novelty_mask => [D,H,W]
        if B==1:
            novelty_mask_sim=novelty_mask_sim.squeeze(0)
            novelty_mask_eds=novelty_mask_eds.squeeze(0)
        
        return novelty_mask_sim, novelty_mask_eds