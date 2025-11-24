# app/model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------------------------------------
# USE GPT-2 LARGE
# -------------------------------------------------------
MODEL_NAME = "gpt2-large"

class NextWordModel:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading GPT-2 Large: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def predict_next(
        self,
        text,
        max_new_tokens=40,
        do_sample=True,
        num_return_sequences=3,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    ):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # If sampling disabled BUT multiple sequences requested → use beams
        if not do_sample and num_return_sequences > 1:
            gen_kwargs["num_beams"] = max(2, num_return_sequences)

        outputs = self.model.generate(**inputs, **gen_kwargs)

        results = []
        prompt_len = len(text)
        for out in outputs:
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            continuation = decoded[prompt_len:].strip() if decoded.startswith(text) else decoded.strip()
            results.append(continuation)

        return results


# ------------------------------
# Singleton + Compatibility Alias
# ------------------------------
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = NextWordModel()
    return _model_instance

def get_model_manager():  # Compatibility for your old code
    return get_model()
