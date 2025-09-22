# app.py
import streamlit as st
from src.pipeline import QAPipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Mentor AI - Tư vấn Tuyển sinh",
    page_icon="🎓",
    layout="wide"
)

# --- State Management ---
# Lưu trữ pipeline trong session state để không phải tải lại model mỗi lần re-run
if 'pipeline' not in st.session_state:
    with st.spinner("Khởi tạo Mentor AI... Quá trình này có thể mất vài phút."):
        st.session_state.pipeline = QAPipeline()
        
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Components ---
st.title("🎓 Mentor AI - Trợ lý Tư vấn Tuyển sinh ĐH Bách khoa Đà Nẵng")
st.markdown("Chào mừng bạn! Hãy đặt câu hỏi về thông tin tuyển sinh của Trường Đại học Bách khoa - ĐH Đà Nẵng.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Câu hỏi của bạn là gì?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.spinner("Mentor AI đang suy nghĩ..."):
        response = st.session_state.pipeline.ask(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
