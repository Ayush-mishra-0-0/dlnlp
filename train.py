import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.data_manager import DataManager
from utils.unlearning_trainer import UnlearningTrainer
from utils.model_loader import load_models
import transformers

import torch

def get_retain_batch(retain_set, batch_size):
    """Returns a random batch from the retain set."""
    retain_idx = torch.randint(0, len(retain_set["input_ids"]), (batch_size,))
    
    retain_batch = {
        "input_ids": torch.stack([retain_set["input_ids"][i] for i in retain_idx]),
        "attention_mask": torch.stack([retain_set["attention_mask"][i] for i in retain_idx])
    }
    
    return retain_batch

@hydra.main(config_path="./configs", config_name="model_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    base_model, unlearned_model, tokenizer = load_models(device)

    # Load data
    data_manager = DataManager(cfg)
    forget_set = data_manager.load_forget_set()
    retain_set = data_manager.load_retain_set()



    # Initialize trainer
    trainer = UnlearningTrainer(
        model=unlearned_model,
        device=device,
        cfg=cfg.training
    )

    # Training loop
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0

        for batch in DataLoader(forget_set, batch_size=cfg.training.batch_size):
            retain_batch = get_retain_batch(retain_set, cfg.training.batch_size)
            loss = trainer.unlearn_step(batch, retain_batch)
            epoch_loss += loss

        # Save checkpoints
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            torch.save(unlearned_model.state_dict(), f"models/checkpoint_{epoch}.pt")

    # Save final model
    torch.save(unlearned_model.state_dict(), "models/unlearned_final.pt")

if __name__ == "__main__":
    main()
