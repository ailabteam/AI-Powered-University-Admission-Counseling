# app.py
import os
# Đặt biến môi trường NGAY LẬP TỨC, trước khi torch được import
# Điều này đảm bảo toàn bộ ứng dụng chỉ sử dụng GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import streamlit as st
from src.pipeline import QAPipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Mentor AI - Tư vấn Tuyển sinh",
    page_icon="🎓",
    layout="wide"
)

# --- State Management & Caching ---
# Sử dụng @st.cache_resource để khởi tạo pipeline một lần và tái sử dụng
@st.cache_resource
def load_pipeline():
    """Tải pipeline và cache lại."""
    print("--- First time initialization: Loading QA Pipeline... ---")
    return QAPipeline()

# --- UI Components ---
st.title("🎓 Mentor AI - Trợ lý Tư vấn Tuyển sinh ĐH Bách khoa Đà Nẵng")
st.markdown("""
Chào mừng bạn! Tôi là Mentor AI, trợ lý ảo sẵn sàng giải đáp các thắc mắc về tuyển sinh của Trường. 
Hãy đặt câu hỏi của bạn vào khung chat bên dưới nhé.
""")

# Tải pipeline từ cache
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống AI: {e}")
    st.stop()

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về tuyển sinh..."):
    # Hiển thị tin nhắn của người dùng
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Tạo và hiển thị câu trả lời của trợ lý AI
    with st.chat_message("assistant"):
        with st.spinner("Mentor AI đang suy nghĩ..."):
            result_dict = pipeline.ask(prompt)
            response = result_dict.get("answer", "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời.")
            st.markdown(response)
    
    # Thêm câu trả lời vào lịch sử chat
    st.session_state.messages.append({"role": "assistant", "content": response})
