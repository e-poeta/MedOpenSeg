import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import yaml
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import nibabel as nib
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from monai.data import Dataset, DataLoader,load_decathlon_datalist, CacheDataset, ThreadDataLoader,PersistentDataset
from models.memory_bank import MemoryBank
from models.novelty_decoder import NoveltyDecoder
from models.swin_unetr import get_swin_unetr_model, get_medopenseg
from preprocess.brats import create_body_mask
from transforms.data_transforms import get_btcv_transforms, get_brats_transforms, get_amos_transforms
from models.memory_bank_voxelwise import MemoryBankV

def load_transforms(config, device):
    dataset = config["data"]["dataset"]
    if dataset == "BRATS":
        return get_brats_transforms(device)
    elif dataset == "BTCV":
        return get_btcv_transforms(device)
    elif dataset == "AMOS":
        return get_amos_transforms(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
def find_slice_with_all_unseen(label):
    """
    Given a 3D label array of shape this function
    returns the slice index (along the third dimension) where the 2D slice
    contains all unseen classes (i.e. classes with values > 10).
    """
   
    unseen_classes = set(cls for cls in np.unique(label) if cls > 10)
    unseen_count = len(unseen_classes)

    for missing in range(unseen_count + 1):  # Start from 0 missing to all-1, all-2...
        for slice_id in range(label.shape[2]):
            unique_in_slice = set(np.unique(label[:, :, slice_id]))
            if len(unseen_classes - unique_in_slice) <= missing:  # Allow 'missing' unseen classes to be absent
                return slice_id  # Return the first matching slice

    return None 


def load_model(checkpoint_path, device, config):
    """ Load trained model from checkpoint. """    
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(0), weights_only=True)
    
    model = get_medopenseg(
        device=device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=(96,96,96),
        feature_size=config["model"]["feature_size"],
        embed_dim_final=config["model"]["embed_dim_final"],
        pre_trained_weights=config["training"].get("pretrained_weights"),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("[INFO] Loaded trained model for inference.")
    return model
    

def infer_and_segment(model, memory_bank, test_loader, config, device):
    """
    Perform inference and generate segmentation maps, incorporating novelty detection.
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass through the model
            with torch.amp.autocast(device.type):
                outputs, embedding= sliding_window_inference(inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,  # or something > 0.25
                mode="gaussian",predictor=model)
                print(f"[DEBUG] Model output shape: {outputs.shape}")
            pred_segmentation = torch.argmax(outputs, dim=1)  # [B, D, H, W]
            print(f"[DEBUG] Predicted segmentation shape: {pred_segmentation.shape}")
            
            #INput body mas
            ct_volume = inputs[0,0,:,:,:].cpu().numpy()  # example threshold
            body_mask=create_body_mask(ct_volume, min_val=-2000, connectivity=3)
            # plt.imshow(body_mask[:,:,190], cmap="gray")
            # plt.savefig("body_mask.png")
              
            # Novelty detection
            novelty_mask, novelty_mask_eds = memory_bank.query_voxelwise_novelty(embedding)  # [B, D, H, W]
            novelty_mask = novelty_mask * body_mask
            novelty_mask_eds = novelty_mask_eds * body_mask
            print(f"[DEBUG] Novelty mask EDS shape: {novelty_mask_eds.shape}")
               
            
            pred_seg_np = pred_segmentation[0,:,:,:].cpu().numpy()
            label_np = labels[0,0,:,:,:].cpu().numpy()
            print(f"[DEBUG] Label unique: {np.unique(label_np)}")
            
            flat_values = novelty_mask.flatten()
            percentile=95
            # Randomly sample 1 million values (adjust if needed)
            num_samples = min(1_000_000, flat_values.size)  # Ensure we don't exceed available data
            subset = flat_values[np.random.randint(0, flat_values.size, (num_samples,))]
            # Compute quantile on the subset
            threshold_value = np.quantile(subset, percentile / 100.0)
            print(f"[DEBUG] Threshold value: {threshold_value}")
            novelty_det = (novelty_mask < threshold_value).astype(np.float32)
            
            final_seg = np.where(novelty_det, -1, pred_seg_np)
            print(f"[DEBUG] Final segmentation shape: {final_seg.shape}") 

            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
            slice_idx = find_slice_with_all_unseen(label_np) #final_seg.shape[1] // 2 
            print(f"[DEBUG] Slice index with all unseen classes: {slice_idx}")
            #1) Ground truth
            slice_data = label_np[:, :, slice_idx]
            mask_known = slice_data <= 10
            mask_unseen = slice_data > 10  # Unseen classes
            known_data = np.ma.masked_where(~mask_known, slice_data)
            unseen_data = np.ma.masked_where(~mask_unseen, slice_data)
    
            im0 = axes[0].imshow(known_data, cmap="viridis")
            im1 = axes[0].imshow(unseen_data, cmap="Reds", alpha=0.7)
            axes[0].set_title("Ground truth")
            axes[0].axis("off")
            cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04,location='left')
            cbar.set_label("Known Id", rotation=270, labelpad=15)
            cbar.set_label("Unseen Id", rotation=270, labelpad=15)      
            
            # 2) Closed-Set Segmentation
            axes[1].imshow(pred_seg_np[:, :, slice_idx], cmap="viridis")
            axes[1].set_title("Closed-Set Segmentation")
            axes[1].axis("off")
            
            
            # # 1) Open-Set Segmentation
            # axes[2].imshow(final_seg[:, :, slice_idx], cmap="viridis")
            # axes[2].set_title("Open-Set Segmentation")
            # axes[2].axis("off")

            # 3) Anomaly Map (with colorbar)
            im = axes[2].imshow(novelty_mask[:, :, slice_idx], cmap="hot")
            axes[2].set_title("Similarity Map")
            axes[2].axis("off")
            # Add colorbar for the anomaly map
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Similarity Score", rotation=270, labelpad=15)
            
            # 3) Anomaly Map EDS (with colorbar)
            im = axes[3].imshow(novelty_mask_eds[:, :, slice_idx], cmap="hot")
            axes[3].set_title("Anomaly Map EDS")
            axes[3].axis("off")
            # Add colorbar for the anomaly map
            cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            cbar.set_label("Distance", rotation=270, labelpad=15)

            # # 4) Filtered Anomaly Map
            # axes[5].imshow(novelty_det[:, :, slice_idx], cmap="hot")
            # axes[5].set_title("Anomaly Map (Theresholding SimilarityMap)")
            # axes[5].axis("off")

            plt.tight_layout()
            plt.savefig("comparison_subplots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            
            results.append(final_seg)
       
    return results

def main():
    parser = argparse.ArgumentParser(description="MedOpenSeg Inference")
    parser.add_argument("--config", type=str, default="config_btcv", help="Config file name")
    parser.add_argument("--exp", type=str, default="btcv/memory_enc3", help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load config
    config_path = os.path.join("./configs", f"{args.config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
        
    exps_root = '/home/vargas/medopenseg/outputs'
    checkpoint_path = os.path.join(exps_root, args.exp, 'best_ft.pth')

    
    # Load model
    model = load_model(checkpoint_path, device, config)
    
    # Load memory bank
    embed_dim = config["training"].get("embed_dim_final", 128)
    memory_bank = MemoryBankV(memory_size=100, feature_dim=embed_dim, similarity_threshold=0.7).to(device)
    memory_bank_path = os.path.join(exps_root, args.exp, "memory_bank.pth")
    memory_bank.load_memory_bank(memory_bank_path, device=device)
    print("[INFO] Memory bank loaded successfully.")

    # Load dataset
    data_dir = config["data"]["data_dir"]
    split_json = config["data"]["split_json"]
    datasets = os.path.join(data_dir, split_json)
    train_transforms, val_transforms, test_transforms = load_transforms(config, device)
    test_files = load_decathlon_datalist(datasets, True, "validation")
    
    test_cache_dir =  f"./cache_{args.config}/test"

    # Ensure cache directories exist
    os.makedirs(test_cache_dir, exist_ok=True)

    # Create PersistentDataset for train and validation

    test_ds = PersistentDataset(data=test_files, transform=test_transforms, cache_dir=test_cache_dir)
    test_loader = ThreadDataLoader(test_ds, batch_size=1)

    # test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1., num_workers=4)
    # test_loader = ThreadDataLoader(test_ds, batch_size=1)

    # Run inference
    print("[INFO] Running inference...")
    results = infer_and_segment(model, memory_bank, test_loader,config, device)
    # Save results as .nii.gz
    output_dir = "output_segmentations"
    os.makedirs(output_dir, exist_ok=True)

    for i, segmentation in enumerate(results):
        seg_volume = segmentation.astype(np.int16)  # Ensure integer type for segmentation
        seg_nifti = nib.Nifti1Image(seg_volume, affine=np.eye(4))  # Identity affine if none provided
        nib.save(seg_nifti, os.path.join(output_dir, f"segmentation_{i}.nii.gz"))

    print(f"[INFO] Inference complete. Results saved in {output_dir}")
    
if __name__ == "__main__":
    main()