import streamlit as st
import pandas as pd
from docx import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()
genai.configure(api_key=api_key)

# Hàm tải lịch sử chat từ file
def load_chat_history():
    try:
        with open("rag_chat_history.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Hàm lưu lịch sử chat vào file
def save_chat_history(history):
    with open("rag_chat_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

def get_text_from_documents(docs):
    """Extract text from uploaded documents (PDF, TXT, XLSX, CSV, DOCX)."""
    text = ""
    try:
        for file in docs:
            suffix = os.path.splitext(file.name)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            # Loaders theo loại file
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                pages = loader.load_and_split()
                text += "\n".join([p.page_content for p in pages])

            elif suffix == ".txt":
                loader = TextLoader(tmp_path)
                pages = loader.load_and_split()
                text += "\n".join([p.page_content for p in pages])

            elif suffix == ".csv":
                df = pd.read_csv(tmp_path)
                text += df.to_string(index=False)

            elif suffix == ".xlsx":
                try:
                    sheets = pd.read_excel(tmp_path, sheet_name=None)
                    for name, df in sheets.items():
                        text += f"\n--- Sheet: {name} ---\n"
                        text += df.to_string(index=False)
                except Exception as e:
                    st.warning(f"Lỗi đọc Excel: {e}")
                    continue
                
            elif suffix == ".docx":
                try:
                    doc = Document(tmp_path)
                    for para in doc.paragraphs:
                        text += para.text + "\n"

                    for table in doc.tables:
                        for row in table.rows:
                            for cell in row.cells:
                                text += cell.text + " "
                        text += "\n" # Thêm xuống dòng sau mỗi bảng
                        
                    text += "\n" # Thêm xuống dòng sau mỗi file docx để tách biệt
                except Exception as e:
                    st.warning(f"Lỗi đọc file DOCX: {e}")
                    continue
                
            else:
                st.warning(f"Không hỗ trợ định dạng file: {suffix}")
                continue

            os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Lỗi xử lý tài liệu: {str(e)}")
        return ""
    return text

def get_text_chunks(text):
    """Split text into chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Create and save FAISS vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Tài liệu đã phân tích xong, sẵn sàng để trả lời câu hỏi.")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

def get_prompt_template():
    """Returns the prompt template."""
    prompt_template = """
    Trả lời câu hỏi một cách chi tiết nhất có thể dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không có trong ngữ cảnh được cung cấp, hãy nói, "Câu trả lời không có trong ngữ cảnh."
    Không cung cấp thông tin sai lệch.

    Ngữ cảnh: {context}
    Câu hỏi: {question}

    Answer:
    """
    try:
        return PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    except Exception as e:
        st.error(f"Error creating prompt template: {str(e)}")
        return None

def user_input(user_question):
    """Process user question and return streamed response, integrated with chat history.""" 
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if not os.path.exists("faiss_index"):
            st.error("Không tìm thấy FAISS index. Hãy tải tài liệu lên trước.")
            return
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        prompt_template_obj = get_prompt_template() # Lấy prompt template
        if not prompt_template_obj:
            return

        # Nối nội dung các tài liệu tìm được vào ngữ cảnh
        context = "\n\n".join([doc.page_content for doc in docs])

        # Tạo prompt cuối cùng đã bổ sung ngữ cảnh RAG
        rag_augmented_question = prompt_template_obj.format(context=context, question=user_question)
        
        # Chuyển đổi lịch sử chat của Streamlit sang định dạng của Google Generative AI
        # Loại bỏ tin nhắn hiện tại của người dùng khỏi lịch sử để tránh trùng lặp khi gửi message
        gemini_history = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_history.append({"role": "model", "parts": [msg["content"]]})

        # Khởi tạo model và chat session với lịch sử
        model = genai.GenerativeModel("gemini-2.0-flash")
        chat = model.start_chat(history=gemini_history)

        st.write("Reply:") 
        response_placeholder = st.empty()
        full_response_text = ""

        # Stream phản hồi từ mô hình sử dụng chat.send_message
        for chunk in chat.send_message(rag_augmented_question, stream=True):
            if chunk.text:
                full_response_text += chunk.text
                response_placeholder.markdown(full_response_text)
        
        # Lưu phản hồi của bot vào session state và file
        st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
        save_chat_history(st.session_state.chat_history)

    except Exception as e:
        st.error(f"Lỗi xử lý câu hỏi: {str(e)}")

def main():
    st.set_page_config(page_title="Read Documents", page_icon="📄")
    st.header("Demo phân tích tài liệu 📄")

    # Khởi tạo và tải lịch sử chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_question = st.chat_input("Bạn hãy hỏi sau khi tài liệu đã được phân tích xong")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Tải tài liệu lên (PDF, TXT, XLSX, CSV, DOCX)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "xlsx", "csv"]
        )
        if st.button("Phân tích tài liệu"):
            if not pdf_docs:
                st.error("Vui lòng tải tài liệu của bạn lên.")
                return
            with st.spinner("Đang xử lý..."):
                raw_text = get_text_from_documents(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                    else:
                        st.error("Kiểm tra lại nội dung trong tài liệu.")
                else:
                    st.error("Không có nội dung nào được trong tài liệu.")
        
        # Nút xóa lịch sử chat
        if st.button("Xóa lịch sử chat"):
            st.session_state.chat_history = []
            if os.path.exists("rag_chat_history.json"):
                os.remove("rag_chat_history.json")
            st.rerun()

if __name__ == "__main__":
    main()