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
st.set_page_config(page_title="RAG 知识库助手 v3", page_icon="🛡️", layout="wide")


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
st.title("🛡️ 智能知识库助手 v3")

# =========================
# 2. 多用户认证
# =========================
# secrets.toml 配置格式示例：
# [users]
# admin = {password = "admin123", role = "admin"}
# zhangsan = {password = "zs666", role = "user"}
# lisi = {password = "ls888", role = "user"}

MAX_LOGIN_ATTEMPTS = 10

if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "current_role" not in st.session_state:
    st.session_state.current_role = None


def get_users_config():
    """从 secrets 中读取用户配置。"""
    users_raw = st.secrets.get("users", {})
    users = {}
    for username, info in users_raw.items():
        if isinstance(info, dict):
            users[username] = {
                "password": info.get("password", ""),
                "role": info.get("role", "user"),
            }
    return users


USERS = get_users_config()

with st.sidebar:
    st.header("🔑 登录")

    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("🚫 登录尝试次数过多，请刷新页面后重试。")
        st.stop()

    if not USERS:
        st.error("⚠️ 未配置用户，请在 secrets.toml [users] 中添加。")
        st.stop()

    input_username = st.text_input("用户名", key="login_user")
    input_password = st.text_input("密码", type="password", key="login_pass")

    if input_username == "" and input_password == "":
        st.stop()

    user_info = USERS.get(input_username)
    if not user_info or user_info["password"] != input_password:
        if input_username != "" or input_password != "":
            st.session_state.login_attempts += 1
            remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
            st.warning(f"⚠️ 用户名或密码错误（剩余 {remaining} 次）")
        st.stop()
    else:
        st.session_state.login_attempts = 0
        st.session_state.current_user = input_username
        st.session_state.current_role = user_info["role"]
        role_label = "管理员" if user_info["role"] == "admin" else "普通用户"
        st.success(f"✅ {input_username}（{role_label}）")

CURRENT_USER = st.session_state.current_user
IS_ADMIN = st.session_state.current_role == "admin"

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

# =========================
# 4. 公共库 / 私有库 索引管理
# =========================
INDEX_ROOT = "rag_index_v2"
PUBLIC_DIR = os.path.join(INDEX_ROOT, "public")


def _get_private_dir(username):
    return os.path.join(INDEX_ROOT, "private", username)


def _docs_path(index_dir):
    return os.path.join(index_dir, "docs.json")


def _embeddings_path(index_dir):
    return os.path.join(index_dir, "embeddings.npz")


def save_index(index_dir, docs, embeddings):
    """保存索引到指定目录。"""
    os.makedirs(index_dir, exist_ok=True)
    with open(_docs_path(index_dir), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    np.savez_compressed(_embeddings_path(index_dir), embeddings=np.array(embeddings))


def load_index(index_dir):
    """从指定目录加载索引，返回 (docs, embeddings)。"""
    dp = _docs_path(index_dir)
    ep = _embeddings_path(index_dir)
    if os.path.exists(dp) and os.path.exists(ep):
        try:
            with open(dp, "r", encoding="utf-8") as f:
                docs = json.load(f)
            embeddings = list(np.load(ep)["embeddings"])
            return docs, embeddings
        except Exception as e:
            logger.warning(f"索引加载失败 [{index_dir}]: {e}")
    return [], []


def clear_index(index_dir):
    """清空指定目录的索引文件。"""
    dp = _docs_path(index_dir)
    ep = _embeddings_path(index_dir)
    if os.path.exists(dp):
        os.remove(dp)
    if os.path.exists(ep):
        os.remove(ep)


# --- 初始化 session state ---
def _init_library(key_prefix, index_dir):
    """初始化某个库的 session state。"""
    docs_key = f"{key_prefix}_docs"
    emb_key = f"{key_prefix}_embeddings"
    if docs_key not in st.session_state:
        docs, embeddings = load_index(index_dir)
        st.session_state[docs_key] = docs
        st.session_state[emb_key] = embeddings


# 公共库
_init_library("public", PUBLIC_DIR)
# 私有库
PRIVATE_DIR = _get_private_dir(CURRENT_USER)
_init_library("private", PRIVATE_DIR)


def _get_embeddings_np(key_prefix):
    """获取缓存的 numpy 数组。"""
    np_key = f"{key_prefix}_embeddings_np"
    ver_key = f"{key_prefix}_emb_version"
    emb_key = f"{key_prefix}_embeddings"
    emb_list = st.session_state.get(emb_key, [])
    if np_key not in st.session_state or st.session_state.get(ver_key, 0) != len(emb_list):
        if emb_list:
            st.session_state[np_key] = np.array(emb_list)
        else:
            st.session_state[np_key] = np.array([])
        st.session_state[ver_key] = len(emb_list)
    return st.session_state[np_key]


# =========================
# 5. 缓存 LLM 客户端
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
# 6. 实用功能函数
# =========================
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

SYSTEM_PROMPT = (
    "你是一个专业的知识问答助手。请基于提供的参考资料回答用户问题。"
    "如果资料中没有相关信息，请诚实说明。回答要准确、有条理、简洁。"
    "不要编造不在资料中的信息。"
)


def web_search(query):
    """使用 Tavily 进行联网搜索。"""
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


def process_upload(uploaded_files, target_prefix, target_dir):
    """处理上传文件并写入指定库。返回是否有新数据写入。"""
    if not uploaded_files:
        return False
    file_fingerprint = str(sorted((f.name, f.size) for f in uploaded_files))
    fp_key = f"_last_upload_fp_{target_prefix}"
    if file_fingerprint == st.session_state.get(fp_key):
        return False

    all_new_chunks = []
    with st.spinner("正在自动解析文档并更新索引..."):
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

            docs_key = f"{target_prefix}_docs"
            emb_key = f"{target_prefix}_embeddings"
            st.session_state[docs_key].extend(all_new_chunks)
            current_emb = list(st.session_state[emb_key])
            current_emb.extend(list(new_vecs))
            st.session_state[emb_key] = current_emb

            save_index(target_dir, st.session_state[docs_key], st.session_state[emb_key])
            st.session_state[fp_key] = file_fingerprint
            st.success(f"自动导入 {len(all_new_chunks)} 个知识切片！")
            time.sleep(1)
            st.rerun()
        else:
            st.error("解析失败，未发现有效文字内容。")
    return False


# =========================
# 7. 侧边栏 UI & 逻辑
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
    # --- 公共知识库 ---
    st.subheader("� 公共知识库")
    pub_count = len(st.session_state.get("public_docs", []))
    st.caption(f"共 **{pub_count}** 个切片（所有人可搜索）")

    if IS_ADMIN:
        pub_files = st.file_uploader(
            "上传到公共库",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="upload_public",
        )
        if pub_files:
            process_upload(pub_files, "public", PUBLIC_DIR)

        if pub_count > 0:
            if st.button("🗑️ 清空公共库", use_container_width=True, type="secondary", key="clear_pub"):
                st.session_state.public_docs = []
                st.session_state.public_embeddings = []
                clear_index(PUBLIC_DIR)
                st.success("公共知识库已清空。")
                time.sleep(0.5)
                st.rerun()
    else:
        st.caption("*仅管理员可维护公共库*")

    st.divider()

    # --- 私有知识库 ---
    st.subheader(f"🔒 我的私有库（{CURRENT_USER}）")
    priv_count = len(st.session_state.get("private_docs", []))
    if priv_count > 0:
        st.caption(f"共 **{priv_count}** 个切片（仅自己可见）")
    else:
        st.caption("私有库为空，上传文档后仅自己可搜索")

    priv_files = st.file_uploader(
        "上传到私有库",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="upload_private",
    )
    if priv_files:
        process_upload(priv_files, "private", PRIVATE_DIR)

    if priv_count > 0:
        if st.button("🗑️ 清空我的私有库", use_container_width=True, type="secondary", key="clear_priv"):
            st.session_state.private_docs = []
            st.session_state.private_embeddings = []
            clear_index(PRIVATE_DIR)
            st.success("私有知识库已清空。")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # --- 模型设置 ---
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

    if st.button("🧹 清空聊天记录", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()


# =========================
# 8. 核心搜索逻辑（合并公共库 + 私有库）
# =========================
def search_local(query, top_k, threshold):
    """搜索公共库和当前用户私有库，合并排序返回 Top-K 结果。"""
    query_vec = embedding_model.encode(query)
    all_results = []  # [(score, doc_text), ...]

    # 搜索公共库
    pub_docs = st.session_state.get("public_docs", [])
    pub_np = _get_embeddings_np("public")
    if pub_docs and pub_np.size > 0:
        scores = cosine_similarity([query_vec], pub_np)[0]
        for i, s in enumerate(scores):
            if s > threshold:
                all_results.append((float(s), pub_docs[i]))

    # 搜索私有库
    priv_docs = st.session_state.get("private_docs", [])
    priv_np = _get_embeddings_np("private")
    if priv_docs and priv_np.size > 0:
        scores = cosine_similarity([query_vec], priv_np)[0]
        for i, s in enumerate(scores):
            if s > threshold:
                all_results.append((float(s), priv_docs[i]))

    # 按相似度降序排列，取 Top-K
    all_results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in all_results[:top_k]]


# =========================
# 9. LLM 回答逻辑
# =========================
def llm_answer(query, context_docs, selected_display_name, web_enabled):
    all_context = ""
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    if context_docs:
        all_context += "【知识库资料】：\n" + "\n".join(context_docs) + "\n"

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
        logger.info(f"[{CURRENT_USER}] 尝试链路: {label}")
        try:
            extra_h = (
                {"HTTP-Referer": "https://streamlit.io", "X-Title": "RAG_v3"}
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
                time.sleep(1.5)
            continue

    yield "❌ 抱歉，所有免费和收费线路均暂时不可用。"


# =========================
# 10. 聊天渲染
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "meta" in m:
            st.caption(m["meta"])

if q := st.chat_input("输入问题...", key="chat_input_v3"):
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
