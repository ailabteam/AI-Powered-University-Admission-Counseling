# src/generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import GENERATOR_MODEL, DEVICE

class LLMGenerator:
    """
    A generator class using a Hugging Face model to generate text.
    """
    def __init__(self):
        print("Initializing LLMGenerator...")
        print(f"  -> Loading model: {GENERATOR_MODEL}")
        
        # Sử dụng bitsandbytes để lượng tử hóa 4-bit, tiết kiệm VRAM
        # Yêu cầu `accelerate` và `bitsandbytes` đã được cài đặt
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # load_in_4bit=True, # Bỏ comment dòng này nếu VRAM không đủ
        )
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        
        # Tạo một pipeline để dễ dàng sử dụng
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512, # Giới hạn độ dài câu trả lời
            repetition_penalty=1.1, # Giảm thiểu việc lặp từ
        )
        print("LLMGenerator initialized successfully.")

    def generate(self, prompt: str) -> str:
        """
        Generates a response based on the provided prompt.
        
        Args:
            prompt (str): The full prompt including context and question.
            
        Returns:
            str: The generated answer from the language model.
        """
        print("Generating answer...")
        
        # Cấu trúc messages cho các model dạng chat
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        # Sử dụng apply_chat_template để định dạng prompt đúng chuẩn
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.pipe(formatted_prompt)
        generated_text = outputs[0]['generated_text']
        
        # Tách chỉ phần câu trả lời của assistant
        answer = generated_text.split("<|assistant|>")[-1].strip()
        
        return answer
