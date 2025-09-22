# src/config.py
import os
import torch

# --- Project Paths ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# --- Knowledge Base Paths ---
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, 'faq_index.faiss')
CONTEXTS_PATH = os.path.join(MODEL_DIR, 'faq_contexts.json')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# --- Model Configuration ---
# Thiết bị sẽ được kiểm soát bởi biến môi trường trong app.py
# Nhưng chúng ta vẫn giữ một định nghĩa dự phòng ở đây.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Embedding model for Retriever
EMBEDDING_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'

# Generator model (LLM)
GENERATOR_MODEL = 'Viet-Mistral/Vistral-7B-Chat'

# --- Search Configuration ---
TOP_K_RETRIEVER = 3

# --- Prompt Template ---
# Đây là phần NỘI DUNG sẽ được đặt giữa các token [INST] và [/INST]
PROMPT_TEMPLATE = """
Bạn là một trợ lý AI tư vấn tuyển sinh chuyên nghiệp của trường Đại học Bách khoa Đà Nẵng.
Dựa vào các thông tin tham khảo dưới đây, hãy trả lời câu hỏi của thí sinh một cách **ngắn gọn, chính xác và đi thẳng vào vấn đề**.
Nếu thông tin không có trong ngữ cảnh được cung cấp, hãy trả lời: "Tôi chưa có thông tin về vấn đề này, bạn có thể liên hệ phòng đào tạo để biết thêm chi tiết."
Tuyệt đối không tự bịa đặt thông tin.

---
Thông tin tham khảo:
{context}
---

Câu hỏi của thí sinh: {question}

Câu trả lời của bạn:
"""
