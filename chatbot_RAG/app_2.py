import os
import time
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain


# =========================
# Thiết lập
# =========================
APP_TITLE = "Tư vấn thủ tục: Đăng ký kết hôn (Công dân Việt Nam)"
APP_DIR = Path(__file__).resolve().parent
KB_PDF_PATH = APP_DIR / "ChiTietTTHC_1.000894.pdf"
MODEL_NAME = "gemini-2.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"

MIN_SECONDS_BETWEEN_REQUESTS = 2
MAX_REQUESTS_PER_DAY = 30

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 4

MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.2


# =========================
# Chống spam (test)
# =========================
def allow_request():
    now = time.time()
    today = str(date.today())

    st.session_state.setdefault("last_req", 0.0)
    st.session_state.setdefault("count_today", 0)
    st.session_state.setdefault("day", today)

    if st.session_state["day"] != today:
        st.session_state["day"] = today
        st.session_state["count_today"] = 0

    if now - st.session_state["last_req"] < MIN_SECONDS_BETWEEN_REQUESTS:
        return False, f"Bạn đang gửi quá nhanh. Đợi {MIN_SECONDS_BETWEEN_REQUESTS} giây nhé."
    if st.session_state["count_today"] >= MAX_REQUESTS_PER_DAY:
        return False, f"Bạn đã đạt giới hạn {MAX_REQUESTS_PER_DAY} câu hỏi hôm nay."

    st.session_state["last_req"] = now
    st.session_state["count_today"] += 1
    return True, ""


# =========================
# KB build/load
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"Không thấy file KB: {pdf_path}")

    reader = PdfReader(str(p))
    text = "\n".join([(page.extract_text() or "") for page in reader.pages]).strip()
    if not text:
        raise ValueError("Không trích xuất được text từ PDF.")
    return text


@st.cache_resource(show_spinner=True)
def load_kb_vectorstore(api_key: str):
    raw_text = extract_text_from_pdf(KB_PDF_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    return vs


@st.cache_resource(show_spinner=False)
def load_qa_chain_cached(api_key: str):
    prompt_template = """
Bạn là một trợ lý ảo chuyên nghiệp hỗ trợ giải đáp các thủ tục hành chính về Đăng ký kết hôn tại Việt Nam.
Sử dụng thông tin có trong NGỮ CẢNH để trả lời câu hỏi của người dùng một cách chính xác, thân thiện và dễ hiểu.
Nếu thông tin không có trong ngữ cảnh nhưng VẪN liên quan đăng ký kết hôn:
- Trả lời dựa trên kiến thức chung về luật pháp Việt Nam.

Nếu câu hỏi KHÔNG liên quan đăng ký kết hôn
- Hãy từ chối trả lời một cách lịch sự.
- Hướng dẫn liên hệ cơ quan có thẩm quyền để được hỗ trợ thêm.

QUY TẮC:
- Trả lời bằng tiếng Việt.
- Không trích dẫn nguyên văn, không sao chép câu chữ từ NGỮ CẢNH.
- Không giải thích dài dòng, không diễn giải luật.
- Trả lời đúng trọng tâm câu hỏi, ưu tiên câu trả lời ngắn.

CÁCH TRẢ LỜI:
- Mỗi bullet tối đa 1 câu, dưới 20 từ.
- Không lặp lại ý.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)


def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Bạn muốn hỏi gì về thủ tục đăng ký kết hôn?"}]


# =========================
# Hỏi nhanh
# =========================
def quick_answer(option: str) -> str:
    if option == "Cách nộp hồ sơ":
        return (
            "- Nộp trực tiếp tại Trung tâm/ Bộ phận một cửa nơi có thẩm quyền.\n"
            "- Nộp trực tuyến trên Cổng DVCQG hoặc Cổng DVC cấp tỉnh (đính kèm bản chụp/bản sao điện tử theo quy định).\n"
            "- Có thể nộp qua dịch vụ bưu chính (nếu địa phương hỗ trợ).\n"
        )
    if option == "Thời hạn giải quyết":
        return (
            "- Ngay trong ngày tiếp nhận hồ sơ.\n"
            "- Nếu nhận hồ sơ sau 15 giờ mà chưa giải quyết được ngay: trả kết quả trong ngày làm việc tiếp theo.\n"
            "- Nếu cần xác minh điều kiện kết hôn: không quá 05 ngày làm việc.\n"
        )
    if option == "Lệ phí":
        return (
            "- Miễn lệ phí đăng ký kết hôn.\n"
            "- Nếu yêu cầu cấp bản sao Trích lục kết hôn: thu phí theo quy định hiện hành.\n"
        )
    if option == "Điều kiện kết hôn":
        return (
            "- Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi.\n"
            "- Hai bên tự nguyện.\n"
            "- Không mất năng lực hành vi dân sự.\n"
            "- Không thuộc các trường hợp cấm kết hôn; Nhà nước không thừa nhận hôn nhân giữa những người cùng giới tính.\n"
        )
    return "answer is not available in the context"


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📄")
    st.title(APP_TITLE)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    if not api_key:
        st.error("Thiếu GOOGLE_API_KEY (set trong Streamlit Secrets hoặc .env).")
        st.stop()

    if "messages" not in st.session_state:
        clear_chat()

    # Style cho mục hỏi nhanh 
    st.markdown(
        """
        <style>
        .quick-box button {
            width: 100%;
            border-radius: 14px !important;
            border: 1px solid #ddd !important;
            padding: 0.65rem 0.9rem !important;
            margin-bottom: 0.55rem !important;
            background: #f9fafb !important;
            text-align: left !important;
            font-weight: 600 !important;
        }
        .quick-box button:hover {
            border-color: #4f46e5 !important;
            background: #eef2ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Hỏi nhanh")

        quick_items = [
            "Cách nộp hồ sơ",
            "Thời hạn giải quyết",
            "Lệ phí",
            "Điều kiện kết hôn",
        ]

        # Mỗi mục là 1 box 
        # Click là trả lời ngay
        # st.markdown('<div class="quick-box">', unsafe_allow_html=True)
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 5px;
        }
        </style>""", unsafe_allow_html=True)
        
        for item in quick_items:
            if st.button(item, key=f"quick_{item}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": item})
                st.session_state.messages.append({"role": "assistant", "content": quick_answer(item)})
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        # st.caption("Chống spam (theo session):")
        # st.caption(f"- {MAX_REQUESTS_PER_DAY} câu/ngày")
        # st.caption(f"- tối thiểu {MIN_SECONDS_BETWEEN_REQUESTS}s/câu")
        st.button("Xóa lịch sử chat", on_click=clear_chat)

    # lịch sử
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Load KB 
    vs = load_kb_vectorstore(api_key)
    chain = load_qa_chain_cached(api_key)

    # input
    if question := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        ok, msg = allow_request()
        if not ok:
            st.warning(msg)
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu thủ tục..."):
                docs = vs.similarity_search(question, k=TOP_K)
                out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                answer = (out or {}).get("output_text", "answer is not available in the context")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
