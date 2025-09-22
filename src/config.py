# src/config.py
import os
import torch

# --- Project Paths ---
# Lấy đường dẫn tuyệt đối của thư mục chứa file config.py (tức là src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn gốc của dự án (thư mục cha của src)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# --- Knowledge Base Paths ---
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, 'faq_index.faiss')
CONTEXTS_PATH = os.path.join(MODEL_DIR, 'faq_contexts.json')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# --- Model Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Embedding model for Retriever
EMBEDDING_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'

# Generator model (LLM)
# Gợi ý: Bắt đầu với một model nhỏ để dễ dàng chạy thử.
# Sau đó có thể đổi sang các model lớn hơn như 'Vistral-7B-Chat'.
GENERATOR_MODEL = 'vinai/PhoGPT-4B-Chat' 
# Hoặc thử nghiệm với các model khác:
# GENERATOR_MODEL = 'Viet-Mistral/Vistral-7B-Chat' 
# GENERATOR_MODEL = 'google/gemma-2-9b-it' # Cần token Hugging Face

# --- Search Configuration ---
# Số lượng context liên quan cần truy xuất
TOP_K_RETRIEVER = 3

# --- Prompt Template ---
PROMPT_TEMPLATE = """
Bạn là một trợ lý AI tư vấn tuyển sinh chuyên nghiệp và thân thiện của trường Đại học Bách khoa Đà Nẵng.
Dựa vào các thông tin tham khảo dưới đây, hãy trả lời câu hỏi của thí sinh một cách chính xác, tự nhiên và đầy đủ.
Nếu thông tin không có trong ngữ cảnh được cung cấp, hãy lịch sự trả lời rằng "Tôi chưa có thông tin về vấn đề này, bạn có thể liên hệ phòng đào tạo để biết thêm chi tiết."
Tuyệt đối không tự bịa đặt thông tin.

---
Thông tin tham khảo:
{context}
---

Câu hỏi của thí sinh: {question}

Câu trả lời của bạn:
"""
