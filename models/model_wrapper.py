from typing import Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerWrapper(nn.Module):
    """Wrapper for HuggingFace models to ensure consistent interface"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        attention_mask = torch.ones(batch_size, seq_length, device=x.device)
        outputs = self.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

class ModelWrapper:
    """Wrapper class to handle HuggingFace transformer models"""
    @staticmethod
    def load_model(model_name: str) -> nn.Module:
        """Load a HuggingFace model by name"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            wrapper = TransformerWrapper(model)
            print(f"Loaded model {model_name} with hidden size: {wrapper.hidden_size}")
            return wrapper
        except Exception as e:
            raise ValueError(f"Error loading HuggingFace model: {str(e)}")