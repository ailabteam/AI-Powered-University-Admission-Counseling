# src/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import GENERATOR_MODEL

class LLMGenerator:
    """
    Generator class using Vistral-7B with 4-bit quantization for efficiency.
    It uses model.generate() for precise control over the text generation process.
    """
    def __init__(self):
        print("Initializing LLMGenerator...")
        print(f"  -> Loading model: {GENERATOR_MODEL}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        
        # Configure 4-bit quantization to save VRAM and run efficiently
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            quantization_config=bnb_config,
            device_map="auto", # Let accelerate handle device mapping on the visible GPU
        )
        print("LLMGenerator initialized successfully.")

    def generate(self, prompt: str) -> str:
        """
        Generates a response based on the provided prompt using model.generate().
        
        Args:
            prompt (str): The full prompt including context and question.
            
        Returns:
            str: The generated answer from the language model.
        """
        print("Generating answer...")
        
        # Vistral uses the Mistral Instruct format: <s>[INST] {prompt} [/INST]
        # We build this prompt manually for full control.
        final_prompt = f"<s>[INST] {prompt} [/INST]"

        # Tokenize the final prompt and send it to the model's device
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate text using the model.generate() method
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        
        # Decode the output and extract only the newly generated part
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return answer.strip()
