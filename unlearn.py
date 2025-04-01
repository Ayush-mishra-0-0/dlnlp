import torch
import torch.nn.functional as F

class UnlearningTrainer:
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def unlearn_step(self, forget_batch, retain_batch):
        """Perform an unlearning step with contrastive training."""
        self.model.train()

        forget_input = forget_batch["input_ids"].to(self.device)
        forget_mask = forget_batch["attention_mask"].to(self.device)

        retain_input = retain_batch["input_ids"].to(self.device)
        retain_mask = retain_batch["attention_mask"].to(self.device)

        # Compute loss for forget set (maximize loss)
        forget_outputs = self.model(forget_input, attention_mask=forget_mask)
        forget_loss = F.cross_entropy(forget_outputs.logits.view(-1, forget_outputs.logits.size(-1)), forget_input.view(-1))

        # Compute loss for retain set (minimize loss)
        retain_outputs = self.model(retain_input, attention_mask=retain_mask)
        retain_loss = F.cross_entropy(retain_outputs.logits.view(-1, retain_outputs.logits.size(-1)), retain_input.view(-1))

        # Final loss (Unlearning objective)
        loss = self.cfg.alpha * forget_loss - self.cfg.beta * retain_loss  # Forget more, retain less
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
