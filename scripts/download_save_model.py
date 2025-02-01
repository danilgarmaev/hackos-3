from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer_path = os.path.join(os.getcwd(), "models", "tokenizers", model_name)

if not os.path.exists(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/{model_name}")
    tokenizer.save_pretrained(tokenizer_path, from_pt=True)

model_path = os.path.join(os.getcwd(), "models", "weights", model_name)

if not os.path.exists(model_path):
    model = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{model_name}")
    model.save_pretrained(model_path, from_pt=True)