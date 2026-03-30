import streamlit as st
import numpy as np
import json
import os
import time
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from docx import Document
from tavily import TavilyClient
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================
# 1. 页面配置 & 样式注入
# =========================
st.set_page_config(page_title="增强版 RAG 助手 v2", page_icon="🛡️", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
            [data-testid="stSidebarContent"] { padding-top: 1.5rem !important; }
            [data-testid="stVerticalBlock"] > div { gap: 0.8rem !important; }
            [data-testid="stFileUploader"] section > div { display: none; }
            [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
            [data-testid="stFileUploader"] section::before {
                content: "拖拽文档至此";
                color: #555; font-size: 14px; display: block; margin-bottom: 10px;
            }
            [data-testid="stFileUploader"] section::after {
                content: "支持格式：TXT, PDF, DOCX";
                color: #888; font-size: 12px; display: block; margin-top: 5px;
            }
            [data-testid="stFileUploader"] button { font-size: 0 !important; }
            [data-testid="stFileUploader"] button::after {
                content: "选择文件";
                font-size: 14px !important;
            }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()
st.title("🛡️ 智能搜索助手 v2")

# =========================
# 2. 访问控制（增加登录失败次数限制）
# =========================
CORRECT_PASSWORD = st.secrets.get("ACCESS_PASSWORD", "")
MAX_LOGIN_ATTEMPTS = 10

if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

with st.sidebar:
    st.header("🔑 认证")

    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("🚫 登录尝试次数过多，请刷新页面后重试。")
        st.stop()

    input_password = st.text_input("口令", type="password", label_visibility="collapsed")

    if not CORRECT_PASSWORD:
        st.error("⚠️ 未配置 ACCESS_PASSWORD，请在 secrets 中设置。")
        st.stop()

    if input_password != CORRECT_PASSWORD:
        if input_password != "":
            st.session_state.login_attempts += 1
            remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
            st.warning(f"⚠️ 验证失败（剩余 {remaining} 次）")
        st.stop()
    else:
        st.session_state.login_attempts = 0
        st.success("✅ 已授权")

# =========================
# 3. 安全配置与模型加载
# =========================
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", "")
DS_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
BAIDU_TOKEN = st.secrets.get("BAIDU_BEARER_TOKEN", "")
BAIDU_APP_ID = st.secrets.get("BAIDU_APP_ID", "")
OR_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-zh")

embedding_model = load_embedding_model()

INDEX_DIR = "rag_index_v2"
DOCS_PATH = os.path.join(INDEX_DIR, "docs.json")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npz")


def save_index(docs, embeddings):
    """使用 JSON + npz 替代 pickle，避免反序列化安全风险。"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    np.savez_compressed(EMBEDDINGS_PATH, embeddings=np.array(embeddings))


OLD_INDEX_PATH = "rag_index.pkl"


def _migrate_old_index():
    """将旧版 pickle 索引迁移为新格式（JSON + npz），迁移后自动删除旧文件。"""
    if not os.path.exists(OLD_INDEX_PATH):
        return None, None
    try:
        import pickle
        with open(OLD_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        docs = list(data.get("docs", []))
        embeddings = list(data.get("embeddings", []))
        if docs and embeddings:
            save_index(docs, embeddings)
            os.remove(OLD_INDEX_PATH)
            logger.info(f"旧索引已迁移至 {INDEX_DIR}/，原 {OLD_INDEX_PATH} 已删除")
            return docs, embeddings
    except Exception as e:
        logger.warning(f"旧索引迁移失败: {e}")
    return None, None


def load_index():
    """加载索引，返回 (docs, embeddings)。优先加载新格式，自动迁移旧格式。"""
    # 优先加载新格式
    if os.path.exists(DOCS_PATH) and os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                docs = json.load(f)
            embeddings = list(np.load(EMBEDDINGS_PATH)["embeddings"])
            return docs, embeddings
        except Exception as e:
            logger.warning(f"索引加载失败，将重置: {e}")
    # 尝试从旧格式迁移
    docs, embeddings = _migrate_old_index()
    if docs is not None:
        return docs, embeddings
    return [], []


if "docs" not in st.session_state:
    st.session_state.docs, st.session_state.embeddings = load_index()

# 缓存 numpy 数组，避免每次查询重复转换
if "embeddings_np" not in st.session_state or st.session_state.get("_emb_version", 0) != len(st.session_state.embeddings):
    if st.session_state.embeddings:
        st.session_state.embeddings_np = np.array(st.session_state.embeddings)
    else:
        st.session_state.embeddings_np = np.array([])
    st.session_state._emb_version = len(st.session_state.embeddings)


# =========================
# 4. 缓存 LLM 客户端（避免每次对话重复创建）
# =========================
@st.cache_resource
def get_or_client():
    return OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")

@st.cache_resource
def get_ds_client():
    return OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")

@st.cache_resource
def get_baidu_client():
    return OpenAI(
        api_key=BAIDU_TOKEN,
        base_url="https://qianfan.baidubce.com/v2",
        default_headers={"appid": BAIDU_APP_ID},
    )


# =========================
# 5. 实用功能函数
# =========================
# 复用的文本分割器（避免循环内重复创建）
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

SYSTEM_PROMPT = (
    "你是一个专业的知识问答助手。请基于提供的参考资料回答用户问题。"
    "如果资料中没有相关信息，请诚实说明。回答要准确、有条理、简洁。"
    "不要编造不在资料中的信息。"
)


def web_search(query):
    """使用 Tavily 进行联网搜索（原函数名 google_search 有误导）。"""
    if not TAVILY_KEY:
        return "⚠️ 未配置搜索 Key"
    tavily = TavilyClient(api_key=TAVILY_KEY)
    current_year = datetime.now().year
    try:
        search_result = tavily.search(
            query=f"{current_year}年 {query}",
            search_depth="advanced",
            max_results=3,
        )
        results = [
            f"来源: {r.get('url')}\n内容: {r.get('content', '')[:700]}"
            for r in search_result["results"]
        ]
        return "\n\n".join(results)[:2500]
    except Exception as e:
        logger.error(f"联网搜索异常: {e}")
        return f"联网搜索异常：{str(e)}"


def estimate_tokens(text):
    """粗略估算 Token 数（中文约 1.5 token/字）。"""
    if not text:
        return 0
    zh_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return int(zh_count * 1.5 + (len(text) - zh_count) * 0.4)


def extract_text(file):
    fname = file.name.lower()
    text = ""
    try:
        if fname.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif fname.endswith(".docx"):
            doc = Document(file)
            text = "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"文件解析失败 [{file.name}]: {e}")
        st.error(f"解析失败: {e}")
    return text


# =========================
# 6. 侧边栏 UI & 逻辑
# =========================
model_mapping = {
    "⭐ Step-3.5 (首选)": "stepfun/step-3.5-flash:free",
    "🌐 OR-Auto (避堵)": "openrouter/free",
    "🧠 GLM-4.5 (推理)": "z-ai/glm-4.5-air:free",
    "🔥 Gemma-3-27B (旗舰)": "google/gemma-3-27b-it:free",
    "🐋 Nemotron (120B)": "nvidia/nemotron-3-super-120b-a12b:free",
    "⚡ Trinity-L (极速)": "arcee-ai/trinity-large-preview:free",
    "💭 Liquid-Think (思维链)": "liquid/lfm-2.5-1.2b-thinking:free",
    "🏎️ Liquid-Ins (1.0s)": "liquid/lfm-2.5-1.2b-instruct:free",
    "⚖️ Gemma-3-12B (平衡)": "google/gemma-3-12b-it:free",
    "💎 Gemma-3n-e4b (稳)": "google/gemma-3n-e4b-it:free",
    "🤖 Nemotron-Nano (混)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "📉 Trinity-M (1.8s)": "arcee-ai/trinity-mini:free",
    "🍃 Nemotron-9B": "nvidia/nemotron-nano-9b-v2:free",
    "🪶 Gemma-3-4B": "google/gemma-3-4b-it:free",
    "🫧 Gemma-3n-e2b": "google/gemma-3n-e2b-it:free",
    "📷 Nemotron-VL": "nvidia/nemotron-nano-12b-v2-vl:free",
    "🛡️ DeepSeek (官方)": "deepseek-chat",
    "🏢 百度文心 (官方)": "ernie-3.5-8k",
}

with st.sidebar:
    st.subheader("📂 知识库")

    # 显示当前知识库状态
    doc_count = len(st.session_state.docs)
    if doc_count > 0:
        st.caption(f"当前已有 **{doc_count}** 个知识切片")
    else:
        st.caption("知识库为空，请上传文档")

    uploaded_files = st.file_uploader(
        "上传",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="u_v2",
    )

    if st.button("🚀 更新索引", use_container_width=True):
        if uploaded_files:
            all_new_chunks = []
            with st.spinner("正在解析文档并提取向量..."):
                for f in uploaded_files:
                    f.seek(0)
                    raw_text = extract_text(f)

                    if not raw_text.strip():
                        st.warning(f"文件 {f.name} 内容为空，已跳过。")
                        continue

                    chunks = TEXT_SPLITTER.split_text(raw_text)
                    all_new_chunks.extend(chunks)

                if all_new_chunks:
                    new_vecs = embedding_model.encode(all_new_chunks)

                    st.session_state.docs.extend(all_new_chunks)
                    current_embeddings = list(st.session_state.embeddings)
                    current_embeddings.extend(list(new_vecs))
                    st.session_state.embeddings = current_embeddings

                    save_index(st.session_state.docs, st.session_state.embeddings)

                    st.success(f"成功导入 {len(all_new_chunks)} 个知识切片！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("解析失败，未发现有效文字内容。")
        else:
            st.info("请先上传文件再点击更新。")

    # 清空知识库按钮
    if doc_count > 0:
        if st.button("🗑️ 清空知识库", use_container_width=True, type="secondary"):
            st.session_state.docs = []
            st.session_state.embeddings = []
            if os.path.exists(DOCS_PATH):
                os.remove(DOCS_PATH)
            if os.path.exists(EMBEDDINGS_PATH):
                os.remove(EMBEDDINGS_PATH)
            st.success("知识库已清空。")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    st.subheader("⚙️ 模型设置")
    selected_display_name = st.selectbox(
        "模型", list(model_mapping.keys()), index=0, label_visibility="collapsed"
    )

    web_on = st.toggle("🌐 联网增强", value=False)

    c1, c2 = st.columns(2)
    with c1:
        ui_top_k = st.number_input("Top-K", 1, 15, 5)
    with c2:
        ui_threshold = st.number_input("阈值", 0.0, 1.0, 0.25, step=0.05)

    st.divider()

    # 清空聊天记录
    if st.button("🧹 清空聊天记录", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# =========================
# 7. 核心对话逻辑
# =========================
def search_local(query, top_k, threshold):
    if not st.session_state.docs or st.session_state.embeddings_np.size == 0:
        return []
    query_vec = embedding_model.encode(query)
    scores = cosine_similarity([query_vec], st.session_state.embeddings_np)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [st.session_state.docs[i] for i in top_indices if scores[i] > threshold]


def llm_answer(query, context_docs, selected_display_name, web_enabled):
    all_context = ""
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    if context_docs:
        all_context += "【本地库资料】：\n" + "\n".join(context_docs) + "\n"

    if web_enabled:
        search_res = web_search(query)
        all_context += f"\n【互联网实时资料】：\n{search_res}"

    prompt_content = f"当前时间：{curr_time}\n\n参考资料：\n{all_context[:6500]}\n\n用户问题：{query}"
    input_tokens = estimate_tokens(prompt_content)

    or_client = get_or_client()
    ds_client = get_ds_client()
    baidu_client = get_baidu_client()

    special_clients = {"deepseek-chat": ds_client, "ernie-3.5-8k": baidu_client}
    selected_id = model_mapping[selected_display_name]

    retry_queue = []
    retry_queue.append(
        (special_clients.get(selected_id, or_client), selected_id, f"首选-{selected_display_name}")
    )

    if selected_id != "stepfun/step-3.5-flash:free":
        retry_queue.append((or_client, "stepfun/step-3.5-flash:free", "⚡ 快速备选-Step3.5"))

    if selected_id != "openrouter/free":
        retry_queue.append((or_client, "openrouter/free", "OR-Auto 免费避堵"))

    paid_backups = [
        ("deepseek-chat", "🛡️ DeepSeek 官方", ds_client),
        ("ernie-3.5-8k", "🏢 百度文心", baidu_client),
    ]
    for p_id, p_label, p_client in paid_backups:
        if selected_id != p_id:
            retry_queue.append((p_client, p_id, f"💰 收费兜底-{p_label}"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content},
    ]

    for idx, (client, m_id, label) in enumerate(retry_queue):
        logger.info(f"尝试链路: {label}")
        try:
            extra_h = (
                {"HTTP-Referer": "https://streamlit.io", "X-Title": "RAG_v2"}
                if client is or_client
                else None
            )
            response = client.chat.completions.create(
                model=m_id,
                messages=messages,
                stream=True,
                extra_headers=extra_h,
                timeout=25,
            )

            full_text = ""
            has_content = False
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text += content
                    has_content = True
                    yield content

            if has_content:
                st.session_state["last_meta"] = (
                    f"🟢 {label} | 📊 ~{input_tokens}/{estimate_tokens(full_text)} Tokens"
                )
                return

        except Exception as e:
            err_msg = str(e)
            logger.warning(f"{label} 失败: {err_msg[:100]}")
            if "429" in err_msg:
                st.toast(f"{label} 拥堵，切换备选...", icon="⏳")
                # 429 限流后短暂等待再尝试下一个
                time.sleep(1.5)
            continue

    yield "❌ 抱歉，所有免费和收费线路均暂时不可用。"


# =========================
# 8. 聊天渲染
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "meta" in m:
            st.caption(m["meta"])

if q := st.chat_input("输入问题...", key="chat_input_v2"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        relevant_docs = search_local(q, ui_top_k, ui_threshold)
        container = st.empty()
        container.markdown("*🤔 正在组织语言...*")

        if web_on:
            with st.status("🌐 正在抓取实时网络数据...", expanded=False) as s:
                time.sleep(0.1)
                s.update(label="✅ 网络资料已就绪", state="complete")

        try:
            full_response = container.write_stream(
                llm_answer(q, relevant_docs, selected_display_name, web_on)
            )
            meta_info = st.session_state.get("last_meta", "")
            st.caption(meta_info)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "meta": meta_info}
            )
        except Exception as e:
            logger.error(f"模型调用异常: {e}")
            container.error(f"❌ 抱歉，连接模型时出错了: {str(e)}")
