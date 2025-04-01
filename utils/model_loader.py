# model trainer file


# utils/model_loader.py
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_compute_dtype="float32",  # Use float32 instead of float16 (better for CPU)
    bnb_4bit_use_double_quant=True,  # Enables double quantization for efficiency
)



def load_models(device):
    model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="cpu"  # Force CPU usage
    )
    unlearned_model = AutoModelForCausalLM.from_pretrained(model_name)
    return base_model.to(device), unlearned_model.to(device)