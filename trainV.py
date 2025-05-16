import os
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from tqdm import tqdm

from models.memory_bank_voxelwise import MemoryBankV
from losses.proto_loss import compute_prototype_pull_loss, get_warmup_alpha, contrastive_loss,compute_dce_and_vl_loss
from preprocess.brats import extract_one_shot_unknown_sample

def save_feature_embeddings(features, labels, epoch, save_dir="./feature_logs"):
    """
    Saves feature embeddings and corresponding labels every N epochs.

    Args:
        features (Tensor): Feature embeddings (batch_size, feature_dim).
        labels (Tensor): Corresponding class labels (batch_size,).
        epoch (int): Current epoch number.
        save_dir (str): Directory where embeddings are stored.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"features.pt")
    torch.save({"features": features.cpu(), "labels": labels.cpu()}, save_path)
    print(f" Feature embeddings saved at epoch {epoch} -> {save_path}")



def validation(model, val_loader, dice_metric, device, post_label, post_pred):
    """
    Validation function using sliding window inference and Dice metric computation.
    """
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
            with torch.amp.autocast("cuda"):
                val_outputs,_ = sliding_window_inference(
                    val_inputs, roi_size=(96, 96, 96), sw_batch_size=4, predictor=model
                )
            val_labels_list = decollate_batch(val_labels)
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

        # Aggregate Dice score
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice


def train_model(
    model, train_loader, val_loader, test_loader, config, loss_function, optimizer, scaler, checkpoint_dir,  device, 
):
    """
    Training function with validation, memory bank usage, and checkpoint saving.
    """
    max_iterations = config["training"]["max_iterations"]
    eval_num = config["training"]["eval_num"]
    use_memory_bank = config["training"].get("use_memory_bank", False)
    resume = config["training"].get("resume", False)
    embed_dim = config["training"].get("embed_dim_final", 128)
    memory_bank_path = os.path.join(checkpoint_dir, "memory_bank.pth")


     # Initialize Memory Bank only if enabled
    memory_bank = None
    if use_memory_bank:
        memory_bank = MemoryBankV(
            memory_size=config["training"]["memory_size"],
            feature_dim=embed_dim,
            similarity_threshold=config["training"]["similarity_threshold"],
            save_path=os.path.join(checkpoint_dir,f"prototypes_{checkpoint_dir.split('/')[-1]}")
        ).to(device)
        print(f"[INFO] Memory Bank initialized with embed_dim={embed_dim}.")

    post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])
    post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    start_epoch = 1
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    best_epoch=0
    UNKNOWN_CLASS_ID=999
    #epoch = 1

 
    if resume:
        resume_checkpoint = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        print(f"[INFO] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint_data = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint_data and checkpoint_data["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
            start_epoch = checkpoint_data["epoch"] + 1
            global_step = checkpoint_data.get("global_step", start_epoch * len(train_loader))
            dice_val_best = checkpoint_data.get("dice_val_best", 0.0)
            global_step_best = checkpoint_data.get("global_step_best",  start_epoch * len(train_loader))
            best_epoch = start_epoch
        if use_memory_bank and os.path.exists(memory_bank_path):
            memory_bank.load_memory_bank(memory_bank_path, device=device)
            print("[INFO] Memory bank reloaded for resume.")
            
    # One-Shot Unknown Class Registration
    one_shot_sample = extract_one_shot_unknown_sample(test_loader)
    if one_shot_sample is not None:
        print("[INFO] Using One-Shot Unknown Example for Registration")
        model.eval()  
        with torch.no_grad():
            inputs, labels = one_shot_sample["image"].to(device), one_shot_sample["label"].to(device)
            #_, embedding = model(inputs)  
            _, embedding= sliding_window_inference(inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,  # or something > 0.25
                mode="gaussian",predictor=model)
            labels[labels > 1] = UNKNOWN_CLASS_ID  #TO DO: CHANGE TO BE AUTOMATIC DEPENDING DATASET
            labels = labels.squeeze(1)  # Now shape (B, D, H, W)
            print(f"Unique Labels in One-Shot Sample: {torch.unique(labels, return_counts=True)}")
            print(f"embedding shape: {embedding.shape}")
            print(f"labels shape: {labels.shape}")
            memory_bank.update_prototypes(embedding, labels)  # Register in memory
            

    # Training loop
    #while global_step < max_iterations:
    num_epochs = max_iterations // len(train_loader) + 1 
    print(f"[INFO] Starting training from epoch={start_epoch} to {num_epochs}")
    # Progress bar for total iterations
    global_iterator = tqdm(
        total=max_iterations,
        desc="Total Progress",
        dynamic_ncols=True
    )

    #global_step = 0  # Track across all epochs
    global_iterator.update(global_step)
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_seg_loss = 0.0
        epoch_pull_loss = 0.0
        epoch_hyb_loss = 0.0
        epoch_vl_loss = 0.0
        epoch_dce_loss = 0.0
        
        all_embeddings = []
        all_labels = []
        
        if use_memory_bank:
            memory_bank.epoch_counter = epoch  
        # Progress bar for per-epoch progress
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            dynamic_ncols=True
        )

        for step, batch in enumerate(epoch_iterator):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            #print(f"Unique Labels in Batch: {torch.unique(labels, return_counts=True)}")
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                #print(f"Input shape: {inputs.shape}") 
                logits, embedding = model(inputs)
                seg_loss = loss_function(logits, labels)
                #pull_loss_val = 0.0
                hyb_loss_val = 0.0
                dce_loss_val = 0.0
                vl_loss_val = 0.0
                
                # Memory-Augmented Learning (Only if Enabled)
                if use_memory_bank:
                    labels = batch["label"].to(device).squeeze(1)
                    memory_bank.update_prototypes(embedding.detach(), labels)
                    # current_alpha = get_warmup_alpha(
                    # epoch, warmup_epochs=10,
                    # alpha_final=0.001, alpha_start=0.0001
                    # )
                    #pull_loss = compute_prototype_pull_loss(embedding, labels, memory_bank, alpha=current_alpha)
                    #pull_loss_val = pull_loss.item()
                    labels[labels > 1] = UNKNOWN_CLASS_ID  #TO DO: CHANGE TO BE AUTOMATIC DEPENDING DATASET
                    hybrid_loss, dce_loss, vl_loss = compute_dce_and_vl_loss(embedding, labels, memory_bank.prototypes, lambda_vl=0.01, ignore_index=UNKNOWN_CLASS_ID)
                    hyb_loss_val = hybrid_loss.item()
                    dce_loss_val = dce_loss.item()
                    vl_loss_val = vl_loss.item()
                    loss = seg_loss + 0.1 * hybrid_loss  
                    #loss = seg_loss + pull_loss
                else:
                    loss = seg_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            seg_loss_val = seg_loss.item()
            epoch_seg_loss += seg_loss_val
            #epoch_pull_loss += pull_loss_val
            epoch_hyb_loss += hyb_loss_val
            epoch_dce_loss += dce_loss_val
            epoch_vl_loss += vl_loss_val
 
            global_step += 1
            global_iterator.update(1)  # Update total progress
            epoch_iterator.set_postfix({
                "seg_loss": seg_loss_val,
                #"pull_loss": pull_loss_val,
                "dce_loss": dce_loss_val,
                "vl_loss": vl_loss_val,
                "hyb_loss": hyb_loss_val,
                "total_loss": loss.item()
            })# Show current loss
            

            all_embeddings.append(embedding.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Validation and checkpoint saving
        if (epoch % 100 == 0 and global_step != 0) or global_step == max_iterations:
            print("Saving embeddings...")
            if all_embeddings:
                all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
                all_labels_tensor = torch.cat(all_labels, dim=0)
                save_feature_embeddings(all_embeddings_tensor, all_labels_tensor, epoch, save_dir=os.path.join(checkpoint_dir,'feature_logs'))

        if (epoch % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            torch.cuda.empty_cache() 
            print("Starting validation...")
            mean_dice = validation(
                model, val_loader, dice_metric, device, post_label, post_pred
            )
            
            if use_memory_bank and len(memory_bank.prototypes) > 0:
                memory_bank_path = os.path.join(checkpoint_dir, "memory_bank.pth")
                memory_bank.save_memory_bank(memory_bank_path)
                print('Memory Bank saved!')

            if mean_dice > dice_val_best:
                dice_val_best = mean_dice
                global_step_best = global_step
                best_epoch = epoch
 
                # Save the model
                #torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_metric_model.pth"))
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                "global_step": global_step,
                "dice_val_best": dice_val_best,
                "global_step_best": global_step_best,
                }, os.path.join(checkpoint_dir, "best_checkpoint.pth"))
                
                print(
                    f"Model saved! Best Avg. Dice: {dice_val_best:.4f}, Current Avg. Dice: {mean_dice:.4f}"
                )
                
            else:
                print(
                    f"Model not saved. Best Avg. Dice: {dice_val_best:.4f}, Current Avg. Dice: {mean_dice:.4f}"
                )

            # Break loop if max iterations reached
            if global_step >= max_iterations:
                break

        print(f"Epoch {epoch} completed. Loss: {epoch_loss / (step + 1):.5f}")
        epoch += 1 
        #  Fine-Tuning Phase with One-Shot Learning
    # print("[INFO] Fine-tuning with One-Shot Unknown Example...")
    # fine_tune_batch_size = max(1, train_loader.batch_size // 2)  # Reduce batch size by half
    # print(f"[INFO] New fine-tune batch size: {fine_tune_batch_size}")
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # # Reduce learning rate for fine-tuning
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = param_group['lr'] * 0.1

    # for epoch in range(1, 5):  # Fine-tune for a small number of epochs
    #     model.train()
    #     epoch_loss = 0.0
    #     accumulation_steps = 2 
    #     for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Fine-Tuning Epoch {epoch}"):
    #         inputs, labels = batch["image"].to(device), batch["label"].to(device)

    #         with torch.amp.autocast("cuda"):
    #             logits, embedding = model(inputs)
    #             seg_loss = loss_function(logits, labels)
    #             labels[labels > 10] = UNKNOWN_CLASS_ID
    #             hybrid_loss, dce_loss, vl_loss = compute_dce_and_vl_loss(embedding, labels, memory_bank.prototypes, lambda_vl=0.01, ignore_index=UNKNOWN_CLASS_ID)
    #             loss = hybrid_loss #seg_loss + 0.1 * hybrid_loss  
    #             loss = loss / accumulation_steps
    #             # Add contrastive loss to separate unknowns
    #             #cont_loss = contrastive_loss(embedding, labels, tau=0.1) / accumulation_steps
    #             print(f"seg_loss: {seg_loss.item()}, dce_loss: {dce_loss.item()}, hybrid_loss: {hybrid_loss.item()}, 'vl_loss:' {vl_loss.item()}")
               

    #         scaler.scale(loss).backward()
    #         if (step + 1) % accumulation_steps == 0:  # 
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad(set_to_none=True)
    #         epoch_loss += loss.item()* accumulation_steps

    #     print(f"Fine-Tune Epoch {epoch} - Loss: {epoch_loss / len(train_loader):.4f}")

    # print("[INFO] Fine-Tuning Completed. Saving Updated Model...")
    # torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ft.pth"))

    print(f"Training completed! Best Avg. Dice: {dice_val_best:.4f} at epoch {best_epoch} - iteration {global_step_best}.")