import os
import yaml
import torch
import argparse
from monai.data import CacheDataset, ThreadDataLoader, load_decathlon_datalist,PersistentDataset
from monai.transforms import Compose
from monai.losses import DiceCELoss
from trainV import train_model  ##CHECK
from models.swin_unetr import get_swin_unetr_model, get_medopenseg

from transforms.data_transforms import get_btcv_transforms, get_brats_transforms,get_amos_transforms,get_msdpancreas_transforms
import warnings
from monai.utils import set_determinism

warnings.filterwarnings("ignore")
torch.cuda.set_device(0)

def load_transforms(config, device):
    dataset = config["data"]["dataset"]
    if dataset == "BRATS":
        return get_brats_transforms(device)
    elif dataset == "BTCV":
        return get_btcv_transforms(device)
    elif dataset == "AMOS":
        return get_amos_transforms(device)
    elif dataset == "MSD_PANCREAS":
         return get_msdpancreas_transforms(device)
    # elif dataset == "TOTALSEG":
    #     return get_totalseg_transforms(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def parse_arguments():
    """
    Parse command-line arguments to dynamically select tasks or configurations.
    """
    parser = argparse.ArgumentParser(description="MedOpenSeg: Open-World Medical Segmentation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_amos", 
        help="Name of the config file (without .yaml)"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["train", "validate", "test"], 
        default="train", 
        help="Specify the task to execute"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Construct config file path based on dataset name
    config_path = os.path.join("./configs", f"{args.config.lower()}.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Set random seed for reproducibility
    set_determinism(seed=42)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = config["data"]["data_dir"]
    split_json = config["data"]["split_json"]

    datasets = os.path.join(data_dir, split_json)
    train_transforms,val_transforms,test_transforms= load_transforms(config, device) #get_btcv_transforms(device, num_samples=4)

    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1, num_workers=8)
    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1, num_workers=4)
    
    # Define cache directories
    dataset = config["data"]["dataset"]
    train_cache_dir = f"./cache_{dataset.lower()}/train"
    val_cache_dir =  f"./cache_{dataset.lower()}/val"
    test_cache_dir =  f"./cache_{dataset.lower()}/test"

    # Ensure cache directories exist
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)

    # Create PersistentDataset for train and validation
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=train_cache_dir)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=val_cache_dir)

    test_ds = PersistentDataset(data=val_files, transform=test_transforms, cache_dir=test_cache_dir)

    train_loader = ThreadDataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = ThreadDataLoader(val_ds, batch_size=1)
    test_loader = ThreadDataLoader(test_ds, batch_size=1)

    #Load model
    # model = get_swin_unetr_model(
    #     device=device,
    #     in_channels=config["model"]["in_channels"],
    #     out_channels=config["model"]["out_channels"],
    #     img_size=(96,96,96),
    #     feature_size=config["model"]["feature_size"],
    #     pre_trained_weights=config["training"].get("pretrained_weights"),
    # )
    model = get_medopenseg(
        device=device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=(96,96,96),
        feature_size=config["model"]["feature_size"],
        embed_dim_final=config["model"]["embed_dim_final"],
        pre_trained_weights=config["training"].get("pretrained_weights"),
    )
    
    # Training setup
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(lr=float(config["training"]["lr"]), params=model.parameters(), weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    # Train the model
    train_model(
        model,train_loader, val_loader, test_loader, config, loss_function, optimizer, scaler, config["training"]["checkpoint_dir"], device
    )


if __name__ == "__main__":
    main()