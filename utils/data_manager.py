# utils/data_manager.py
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from omegaconf import DictConfig
class DataManager:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def load_tokenizer(cls):
        return AutoTokenizer.from_pretrained("gpt2-medium")

    def load_forget_set(self):
        """Load and tokenize the forget set"""
        return self._load_and_process_data(
            Path(self.config.data.paths.forget_set)
        )
    def generate_response(model, tokenizer, text, max_length=50):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_retain_set(self):
        """Load and tokenize the retain set"""
        return self._load_and_process_data(
            Path(self.config.data.paths.retain_set)
        )

    def _load_and_process_data(self, file_path):
        """Internal method to load and tokenize data"""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file {file_path} not found")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return self.tokenize_data(data)

    def tokenize_data(self, data):
        """Tokenize a list of text samples"""
        return self.tokenizer(
            [d['text'] for d in data],
            padding='max_length',
            truncation=True,
            max_length=self.config.data.tokenizer.max_length,
            return_tensors="pt"
        )

    @staticmethod
    def create_splets(full_dataset):
        """Split data into forget/retain sets"""
        # Implementation details would go here
        # Typically you'd want to:
        # 1. Shuffle the dataset
        # 2. Split based on config.data.splits
        # 3. Save to forget_set.json and retain_set.json
        pass