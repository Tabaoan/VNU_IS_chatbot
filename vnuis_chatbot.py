# ===================== IMPORTS =====================
import os, re
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from langdetect import detect


# ===================== ENV =====================
OPENAI__API_KEY = os.getenv("OPENAI__API_KEY")
OPENAI__EMBEDDING_MODEL = os.getenv("OPENAI__EMBEDDING_MODEL")
OPENAI__MODEL_NAME = os.getenv("OPENAI__MODEL_NAME")
OPENAI__TEMPERATURE = os.getenv("OPENAI__TEMPERATURE")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_DIM = 3072


LANG_MODEL_API_KEY = os.getenv("LANG_MODEL_API_KEY")
# ===================== INIT LLM =====================
llm = ChatOpenAI(
    api_key=OPENAI__API_KEY,
    model_name=OPENAI__MODEL_NAME,
    temperature=float(OPENAI__TEMPERATURE) if OPENAI__TEMPERATURE else 0
)
lang_llm = ChatOpenAI(
    api_key=LANG_MODEL_API_KEY,
    model_name="gpt-4o-mini",
    temperature=0
)
# Khởi tạo Pinecone Client (Serverless API)
if PINECONE_API_KEY:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
else:
    pc = None
    print("❌ Lỗi: Không tìm thấy PINECONE_API_KEY. Pinecone sẽ không hoạt động.")

emb = OpenAIEmbeddings(api_key=OPENAI__API_KEY, model=OPENAI__EMBEDDING_MODEL)

vectordb = None
retriever = None


# ===================== SYSTEM PROMPT (ĐÃ RÚT GỌN) =====================
PDF_READER_SYS = (
    "Bạn là một trợ lý AI có nhiệm vụ trích xuất và trả lời thông tin dựa trên nội dung tài liệu. "
    "Luôn tuân thủ các nguyên tắc sau:\n\n"
    "1. Chỉ trả lời dựa trên nội dung có trong tài liệu được cung cấp (context).\n"
    "2. Nếu thông tin không có trong tài liệu, hãy nói rõ rằng tài liệu không chứa thông tin liên quan.\n"
    "3. Nếu tài liệu có thông tin, phải trả lời đầy đủ và chính xác theo đúng nội dung đó.\n"
    "4. Không được tự suy diễn hoặc thêm kiến thức bên ngoài.\n"
    "5. Luôn trả lời bằng đúng ngôn ngữ mà người dùng sử dụng.\n"
    "6. Văn phong rõ ràng, trung lập.\n"
    "7. Tránh sử dụng các cụm từ như 'Dựa trên tài liệu được cung cấp' trong câu trả lời.\n"
    "8. Trả lời theo đúng ngôn ngữ của người dùng nhập vào.\n"
)

# ===================== LANGUAGE UTILS =====================
# 🔵 NEW: Phát hiện ngôn ngữ bằng OpenAI
def detect_language_openai(text: str) -> str:
    try:
        res = lang_llm.invoke([
            SystemMessage(content=(
                "Bạn là module phát hiện ngôn ngữ. "
                "Chỉ trả về mã ISO-639-1 như: vi, en, ja, ko, zh, fr, es. "
                "Không giải thích thêm."
            )),
            HumanMessage(content=text)
        ]).content
        return res.strip().lower()
    except:
        return "vi"


# 🔵 NEW: Dịch output đúng ngôn ngữ người dùng
def convert_language(text: str, target_lang: str) -> str:

    lang_mapping = {
        "vi": "Vietnamese",
        "en": "English",
        "ko": "Korean",
        "ja": "Japanese",
        "zh": "Chinese",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "th": "Thai"
    }
    target_lang_name = lang_mapping.get(target_lang, target_lang)

    try:
        translated = lang_llm.invoke([
            SystemMessage(content="Bạn là một phiên dịch chuyên nghiệp. Chỉ trả về bản dịch."),
            HumanMessage(
                content=f"Dịch đoạn văn sau sang {target_lang_name} ({target_lang}):\n{text}"
            )
        ]).content
        return translated.strip()
    except:
        return text

# ===================== VECTORDB UTILS =====================
def _list_index_names() -> List[str]:
    """
    Trả về danh sách tên index từ Pinecone, hỗ trợ nhiều dạng trả về
    của các version client khác nhau.
    """
    if pc is None:
        return []
    try:
        res = pc.list_indexes()
        # Một số version trả về object có .names()
        if hasattr(res, "names"):
            return list(res.names())
        # Docs mới: trả về list[dict] hoặc dict{'indexes': [...]}
        if isinstance(res, dict) and "indexes" in res:
            return [idx.get("name") for idx in res["indexes"] if "name" in idx]
        if isinstance(res, list):
            names = []
            for idx in res:
                if isinstance(idx, dict) and "name" in idx:
                    names.append(idx["name"])
                elif isinstance(idx, str):
                    names.append(idx)
            return names
        return []
    except Exception as e:
        print(f"⚠️ Lỗi khi list_indexes: {e}")
        return []


def build_context_from_hits(hits, max_chars: int = 6000) -> str:
    ctx = []
    total = 0
    for h in hits:
        source = h.metadata.get('source', 'unknown')
        seg = f"[Nguồn: {source}]\n{h.page_content.strip()}"
        if total + len(seg) > max_chars:
            break
        ctx.append(seg)
        total += len(seg)
    return "\n\n".join(ctx)


def load_vectordb():
    """Load Pinecone index (dùng cho cả CLI và server)."""
    global vectordb, retriever, pc

    if pc is None or not PINECONE_INDEX_NAME:
        print("❌ Pinecone client chưa được khởi tạo hoặc thiếu PINECONE_INDEX_NAME.")
        return None

    try:
        index_names = _list_index_names()
        if PINECONE_INDEX_NAME not in index_names:
            print(f"❌ Index '{PINECONE_INDEX_NAME}' không tồn tại trong Pinecone.")
            return None

        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()

        if stats.get("total_vector_count", 0) == 0:
            print("❌ Index không chứa document nào.")
            return None

        vectordb = Pinecone(index=index, embedding=emb, text_key="text")
        retriever = vectordb.as_retriever(search_kwargs={"k": 15})
        print(f" VectorDB loaded: {PINECONE_INDEX_NAME} với {stats.get('total_vector_count', 0)} vectors")
        return vectordb

    except Exception as e:
        print(f" Lỗi load vectordb: {e}")
        return None


def check_vectordb_exists() -> bool:
    """
    Hàm này được Flask dùng trong /api/status.
    Tự động load vectordb nếu chưa có.
    """
    global vectordb, retriever

    if pc is None or not PINECONE_INDEX_NAME:
        return False

    try:
        index_names = _list_index_names()
        if PINECONE_INDEX_NAME not in index_names:
            return False

        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            return False

        # Nếu retriever chưa khởi tạo thì khởi tạo
        if retriever is None:
            vectordb = Pinecone(index=index, embedding=emb, text_key="text")
            retriever = vectordb.as_retriever(search_kwargs={"k": 15})

        return True
    except Exception as e:
        print(f"⚠️ Lỗi check_vectordb_exists: {e}")
        return False


def get_vectordb_stats():
    """
    Dùng cho API /api/status trên Flask server.
    """
    if pc is None or not PINECONE_INDEX_NAME:
        return {"exists": False, "total_documents": 0}

    index_names = _list_index_names()
    if PINECONE_INDEX_NAME not in index_names:
        return {"exists": False, "total_documents": 0}

    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        return {
            "exists": True,
            "name": PINECONE_INDEX_NAME,
            "total_documents": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", EMBEDDING_DIM)
        }
    except Exception as e:
        print(f"⚠️ Lỗi get_vectordb_stats: {e}")
        return {"exists": False, "total_documents": 0}


# ===================== CLEANING =====================
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def clean_question_remove_uris(text: str) -> str:
    txt = _URL_RE.sub(" ", text or "")
    toks = re.split(r"\s+", txt)
    toks = [t for t in toks if not t.lower().endswith(".pdf")]
    return " ".join(toks).strip()


def convert_language(text: str, target_lang: str) -> str:
    try:
        translated = llm.invoke([
            SystemMessage(content="Hãy dịch chính xác đoạn văn sang ngôn ngữ được yêu cầu."),
            HumanMessage(content=f"Dịch sang {target_lang}. Chỉ trả về bản dịch:\n{text}")
        ]).content
        return translated.strip()
    except:
        return text


# ===================== PROCESS QUESTION =====================
def process_pdf_question(i: Dict[str, Any]) -> str:
    global retriever

    message = i["message"]
    history = i.get("history", [])

    clean_question = clean_question_remove_uris(message)

    # 🔵 NEW: Detect ngôn ngữ bằng OpenAI
    try:
        user_lang = detect_language_openai(message)
    except:
        user_lang = "vi"

    # Nếu retriever chưa khởi tạo
    if retriever is None:
        load_vectordb()

    if retriever is None:
        err = "VectorDB chưa được load hoặc không có dữ liệu."
        return convert_language(err, user_lang)

    # 🔍 Query VectorDB
    try:
        hits = retriever.invoke(clean_question)
        if not hits:
            msg = "Tài liệu không chứa thông tin liên quan."
            return convert_language(msg, user_lang)

        context = build_context_from_hits(hits)

        # System prompt kèm ngôn ngữ
        system_msg = (
            PDF_READER_SYS +
            f"\n\nNgười dùng đang dùng ngôn ngữ: {user_lang}."
        )

        messages = [SystemMessage(content=system_msg)]

        if history:
            messages.extend(history[-10:])

        # User message gửi vào LLM
        user_message = (
            f"Câu hỏi: {clean_question}\n\n"
            f"Context:\n{context}\n\n"
            f"Hãy trả lời dựa trên context và bằng ngôn ngữ: {user_lang}."
        )
        messages.append(HumanMessage(content=user_message))

        # 🧠 LLM trả lời
        response = llm.invoke(messages).content

        # 🔵 NEW: Nếu output không đúng ngôn ngữ → dịch lại
        detected_out_lang = detect_language_openai(response)
        if detected_out_lang != user_lang:
            response = convert_language(response, user_lang)

        return response

    except Exception as e:
        msg = f"Lỗi xử lý: {str(e)}"
        return convert_language(msg, user_lang)


# ===================== CHATBOT WRAPPER =====================
pdf_chain = RunnableLambda(process_pdf_question)
store: Dict[str, ChatMessageHistory] = {}


def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chatbot = RunnableWithMessageHistory(
    pdf_chain,
    get_history,
    input_messages_key="message",
    history_messages_key="history"
)


# ===================== TỰ ĐỘNG LOAD KHI DÙNG VỚI SERVER =====================
# Khi file này được import bởi Flask, đoạn dưới sẽ chạy một lần
if pc is not None and PINECONE_INDEX_NAME:
    print("📥 Auto-loading Pinecone Index cho server...")
    load_vectordb()
else:
    print("⚠️ Pinecone chưa cấu hình đầy đủ, VectorDB sẽ không hoạt động.")


# ===================== CLI HELPERS (TÙY CHỌN) =====================
def print_help():
    print("\n" + "=" * 60)
    print("📚 CÁC LỆNH CÓ SẴN:")
    print("=" * 60)
    print(" - exit / quit  : Thoát chương trình")
    print(" - clear        : Xóa lịch sử hội thoại")
    print(" - status       : Kiểm tra trạng thái Pinecone Index")
    print(" - help         : Hiển thị hướng dẫn này")
    print("=" * 60 + "\n")


def handle_command(command: str, session: str) -> bool:
    cmd = command.lower().strip()

    if cmd in {"exit", "quit"}:
        print("\n👋 Tạm biệt!")
        return False

    elif cmd == "clear":
        if session in store:
            store[session].clear()
            print("🧹 Đã xóa lịch sử hội thoại.\n")
        return True

    elif cmd == "status":
        stats = get_vectordb_stats()
        print("\n" + "=" * 60)
        print("📊 TRẠNG THÁI PINECONE INDEX")
        print("=" * 60)
        if stats["exists"]:
            print(f"✅ Index: {stats['name']}")
            print(f"📚 Tổng documents: {stats['total_documents']}")
        else:
            print("❌ Index không tồn tại hoặc không có dữ liệu.")
        print("=" * 60 + "\n")
        return True

    elif cmd == "help":
        print_help()
        return True

    return True


# ===================== MAIN (CHẠY CLI, KHÔNG ẢNH HƯỞNG SERVER) =====================
if __name__ == "__main__":
    session = "pdf_reader_session"

    if not all([OPENAI__API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
        print("❌ Thiếu biến môi trường.")
        exit(1)

    print("\n" + "=" * 80)
    print("🤖 CHATBOT TÀI LIỆU (VECTOR + LLM)")
    print("=" * 80)
    print_help()

    print("📥 Đang load Pinecone Index...")
    result = load_vectordb()

    if result is None:
        print("❌ Không thể load Pinecone Index.")
        exit(1)

    stats = get_vectordb_stats()
    print(f"✅ Pinecone sẵn sàng với {stats['total_documents']} documents\n")
    print("💬 Sẵn sàng trả lời câu hỏi!\n")

    while True:
        try:
            message = input("👤 Bạn: ").strip()
            if not message:
                continue

            if not handle_command(message, session):
                break

            # Nếu là lệnh, không xử lý tiếp
            if message.lower() in ["clear", "status", "help"]:
                continue

            print("🔎 Đang tìm kiếm trong Pinecone...")

            response = chatbot.invoke(
                {"message": message},
                config={"configurable": {"session_id": session}}
            )

            print(f"\n🤖 Bot: {response}\n")
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break

        except Exception as e:
            print(f"\n❌ Lỗi: {e}\n")
