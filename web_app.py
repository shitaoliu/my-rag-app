import streamlit as st
import time as _time
_BOOT = _time.time()
import json
import os
import time
import logging
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _perf(label):
    logger.info(f"[PERF] {label}: {_time.time()-_BOOT:.2f}s")

_perf("stdlib imports done")

# numpy 延迟导入：首次使用时才加载
_np_module = None
def _get_np():
    global _np_module
    if _np_module is None:
        import numpy
        _np_module = numpy
        _perf("numpy loaded")
    return _np_module

# =========================
# 1. 页面配置 & 样式注入
# =========================
st.set_page_config(page_title="RAG 知识库助手 v3", page_icon="🛡️", layout="wide")
_perf("page_config done")


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
_perf("CSS + title done")

# =========================
# 2. 用户管理（users.json + 密码哈希）
# =========================
# secrets.toml 只需配置：
#   INVITE_CODE = "你的邀请码"
#   ADMIN_USER = "admin"
#   ADMIN_PASSWORD = "admin123"
#   （以及各种 API Key）

USERS_FILE = "users.json"
MAX_LOGIN_ATTEMPTS = 10


def _hash_password(password):
    """SHA-256 哈希密码。"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users():
    """加载用户数据。首次运行时从 secrets 初始化管理员账号。"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"用户文件加载失败: {e}")

    # 首次运行：从 secrets 创建管理员
    admin_user = st.secrets.get("ADMIN_USER", "admin")
    admin_pass = st.secrets.get("ADMIN_PASSWORD", "")
    if not admin_pass:
        return {}
    users = {
        admin_user: {
            "password_hash": _hash_password(admin_pass),
            "role": "admin",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    }
    _save_users(users)
    return users


def _save_users(users):
    """持久化用户数据。"""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _get_invite_code():
    """获取当前邀请码。优先从 users.json 的元数据读取，否则从 secrets 读取。"""
    users = _load_users()
    meta = users.get("__meta__", {})
    if isinstance(meta, dict) and meta.get("invite_code"):
        return meta["invite_code"]
    return st.secrets.get("INVITE_CODE", "")


def _set_invite_code(new_code):
    """管理员修改邀请码（存入 users.json 元数据，无需改 secrets）。"""
    users = _load_users()
    if "__meta__" not in users or not isinstance(users.get("__meta__"), dict):
        users["__meta__"] = {}
    users["__meta__"]["invite_code"] = new_code
    _save_users(users)


def register_user(username, password, invite_code):
    """注册新用户。返回 (success, message)。"""
    if not username or not password:
        return False, "用户名和密码不能为空"
    if len(username) < 2 or len(username) > 20:
        return False, "用户名长度需要 2-20 个字符"
    if len(password) < 4:
        return False, "密码至少 4 个字符"
    if username.startswith("__"):
        return False, "用户名不能以 __ 开头"

    correct_code = _get_invite_code()
    if not correct_code:
        return False, "邀请码未配置，请联系管理员"
    if invite_code != correct_code:
        return False, "邀请码错误"

    users = _load_users()
    if username in users:
        return False, "用户名已存在"

    users[username] = {
        "password_hash": _hash_password(password),
        "role": "user",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save_users(users)
    logger.info(f"新用户注册: {username}")
    return True, "注册成功，请登录"


def verify_user(username, password):
    """验证用户登录。返回 (success, role)。"""
    users = _load_users()
    user_info = users.get(username)
    if not user_info or not isinstance(user_info, dict):
        return False, None
    if user_info.get("password_hash") != _hash_password(password):
        return False, None
    return True, user_info.get("role", "user")


# --- 认证 UI ---
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "current_role" not in st.session_state:
    st.session_state.current_role = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

with st.sidebar:
    with st.expander("🔑 账号"):
        if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
            st.error("🚫 尝试次数过多，请刷新页面后重试。")
            st.stop()

        users_data = _load_users()
        if not users_data:
            st.error("⚠️ 未配置管理员，请在 secrets 中设置 ADMIN_USER 和 ADMIN_PASSWORD。")
            st.stop()

        # 登录 / 注册 切换
        auth_mode = st.radio(
            "操作",
            ["登录", "注册"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_radio",
        )

        if auth_mode == "登录":
            input_username = st.text_input("用户名", key="login_user")
            input_password = st.text_input("密码", type="password", key="login_pass")

            if input_username == "" or input_password == "":
                st.stop()

            ok, role = verify_user(input_username, input_password)
            if not ok:
                st.session_state.login_attempts += 1
                remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.warning(f"⚠️ 用户名或密码错误（剩余 {remaining} 次）")
                st.stop()
            else:
                st.session_state.login_attempts = 0
                st.session_state.current_user = input_username
                st.session_state.current_role = role
                role_label = "管理员" if role == "admin" else "普通用户"
                st.success(f"✅ {input_username}（{role_label}）")

        else:  # 注册
            reg_user = st.text_input("用户名", key="reg_user")
            reg_pass = st.text_input("密码", type="password", key="reg_pass")
            reg_pass2 = st.text_input("确认密码", type="password", key="reg_pass2")
            reg_code = st.text_input("邀请码", type="password", key="reg_code")

            if st.button("注册", use_container_width=True, key="btn_register"):
                if reg_pass != reg_pass2:
                    st.error("两次密码不一致")
                else:
                    ok, msg = register_user(reg_user, reg_pass, reg_code)
                    if ok:
                        st.success(f"✅ {msg}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
            st.stop()

CURRENT_USER = st.session_state.current_user
IS_ADMIN = st.session_state.current_role == "admin"
_perf("auth done")

# =========================
# 3. 安全配置与 Embedding 策略
# =========================
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", "")
DS_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
BAIDU_TOKEN = st.secrets.get("BAIDU_BEARER_TOKEN", "")
BAIDU_APP_ID = st.secrets.get("BAIDU_APP_ID", "")
OR_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

# ---------- Embedding 统一接口 ----------
# 优先用 API（秒开），失败时回退到本地模型（懒加载）

@st.cache_resource
def _get_embedding_client():
    """获取用于 embedding 的 API 客户端（百度优先，其次 OpenRouter）。"""
    from openai import OpenAI
    if BAIDU_TOKEN and BAIDU_APP_ID:
        return OpenAI(
            api_key=BAIDU_TOKEN,
            base_url="https://qianfan.baidubce.com/v2",
            default_headers={"appid": BAIDU_APP_ID},
        ), "bge-large-zh"
    if OR_KEY:
        return OpenAI(
            api_key=OR_KEY,
            base_url="https://openrouter.ai/api/v1",
        ), "BAAI/bge-small-zh"
    return None, None


def _api_encode(texts):
    """通过 API 获取 embedding 向量。成功返回 list[ndarray]，失败返回 None。"""
    np = _get_np()
    client, model = _get_embedding_client()
    if client is None:
        return None
    try:
        # API 每次最多处理一定数量，分批
        batch_size = 32
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            all_vecs.extend([np.array(item.embedding) for item in resp.data])
        return all_vecs
    except Exception as e:
        logger.warning(f"API embedding 失败，回退到本地模型: {e}")
        return None


def _get_local_model():
    """懒加载本地 embedding 模型，仅在 API 不可用时才触发。"""
    if "_local_emb_model" not in st.session_state:
        try:
            with st.spinner("API 不可用，正在加载本地向量模型（仅首次）..."):
                from sentence_transformers import SentenceTransformer
                st.session_state._local_emb_model = SentenceTransformer("BAAI/bge-small-zh")
        except ImportError:
            logger.error("sentence-transformers 未安装，本地模型不可用")
            return None
    return st.session_state.get("_local_emb_model")


def encode_texts(texts):
    """统一 encoding 入口：先尝试 API，失败回退本地模型。"""
    if not texts:
        return []
    if isinstance(texts, str):
        texts = [texts]
    # 先尝试 API
    result = _api_encode(texts)
    if result is not None:
        return result
    # 回退到本地模型
    model = _get_local_model()
    if model is None:
        st.error("❌ Embedding 服务不可用：API 调用失败且本地模型未安装。请检查 API Key 配置。")
        return []
    return list(model.encode(texts))


def encode_query(text):
    """单条文本 encoding，返回一维向量。"""
    vecs = encode_texts([text])
    return vecs[0]

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


def _sources_path(index_dir):
    return os.path.join(index_dir, "sources.json")


def save_index(index_dir, docs, embeddings, sources=None):
    np = _get_np()
    os.makedirs(index_dir, exist_ok=True)
    with open(_docs_path(index_dir), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    np.savez_compressed(_embeddings_path(index_dir), embeddings=np.array(embeddings))
    if sources is not None:
        with open(_sources_path(index_dir), "w", encoding="utf-8") as f:
            json.dump(sources, f, ensure_ascii=False)


def load_index(index_dir):
    np = _get_np()
    dp = _docs_path(index_dir)
    ep = _embeddings_path(index_dir)
    if os.path.exists(dp) and os.path.exists(ep):
        try:
            with open(dp, "r", encoding="utf-8") as f:
                docs = json.load(f)
            embeddings = list(np.load(ep)["embeddings"])
            # 加载来源信息（兼容旧索引）
            sp = _sources_path(index_dir)
            sources = []
            if os.path.exists(sp):
                with open(sp, "r", encoding="utf-8") as f:
                    sources = json.load(f)
            # 旧索引没有 sources，用空字符串填充
            if len(sources) != len(docs):
                sources = [""] * len(docs)
            return docs, embeddings, sources
        except Exception as e:
            logger.warning(f"索引加载失败 [{index_dir}]: {e}")
    return [], [], []


def clear_index(index_dir):
    for p in [_docs_path(index_dir), _embeddings_path(index_dir), _sources_path(index_dir)]:
        if os.path.exists(p):
            os.remove(p)


def delete_file_from_index(index_dir, filename, docs_key, emb_key, src_key):
    """从索引中删除指定文件的所有切片，同时删除原始文件。"""
    docs = st.session_state.get(docs_key, [])
    embs = list(st.session_state.get(emb_key, []))
    srcs = st.session_state.get(src_key, [])

    # 如果 sources 全为空（旧索引），则需要重建 sources：
    # 重新解析该文件获取其文本，匹配 docs 中的切片来标记来源
    has_valid_sources = any(s != "" for s in srcs)

    if not has_valid_sources and docs:
        # 旧索引没有 sources 信息，无法按文件名精准过滤
        # 尝试重新解析该文件，找到其产生的切片内容，从索引中移除
        fpath = os.path.join(_get_files_dir(index_dir), filename)
        file_chunks_set = set()
        if os.path.exists(fpath):
            try:
                import io
                with open(fpath, "rb") as rf:
                    class _FakeFile:
                        def __init__(self, buf, name):
                            self._buf = buf
                            self.name = name
                        def read(self, *a): return self._buf.read(*a)
                        def seek(self, *a): return self._buf.seek(*a)
                    fake = _FakeFile(io.BytesIO(rf.read()), filename)
                    raw_text = extract_text(fake)
                    if raw_text.strip():
                        chunks = _get_text_splitter().split_text(raw_text)
                        file_chunks_set = set(chunks)
            except Exception as e:
                logger.warning(f"重建来源信息失败 [{filename}]: {e}")

        if file_chunks_set:
            new_docs, new_embs, new_srcs = [], [], []
            for d, e, s in zip(docs, embs, srcs):
                if d not in file_chunks_set:
                    new_docs.append(d)
                    new_embs.append(e)
                    new_srcs.append(s)
        else:
            # 无法识别切片归属，保持不变（只删文件）
            new_docs, new_embs, new_srcs = docs, embs, srcs
    else:
        # 有 sources 信息，按文件名精准过滤
        new_docs, new_embs, new_srcs = [], [], []
        for d, e, s in zip(docs, embs, srcs):
            if s != filename:
                new_docs.append(d)
                new_embs.append(e)
                new_srcs.append(s)

    st.session_state[docs_key] = new_docs
    st.session_state[emb_key] = new_embs
    st.session_state[src_key] = new_srcs

    # 更新磁盘索引
    if new_docs:
        save_index(index_dir, new_docs, new_embs, new_srcs)
    else:
        clear_index(index_dir)

    # 删除原始文件
    fpath = os.path.join(_get_files_dir(index_dir), filename)
    if os.path.exists(fpath):
        os.remove(fpath)

    # 更新当前 session 的 mtime 缓存，避免 _init_library 再次覆盖
    key_prefix = docs_key.replace("_docs", "")
    st.session_state[f"{key_prefix}_index_mtime"] = _get_index_mtime(index_dir)


def _get_index_mtime(index_dir):
    """获取索引文件的最新修改时间，用于检测磁盘变更。"""
    dp = _docs_path(index_dir)
    if os.path.exists(dp):
        return os.path.getmtime(dp)
    return 0


def _init_library(key_prefix, index_dir):
    docs_key = f"{key_prefix}_docs"
    emb_key = f"{key_prefix}_embeddings"
    src_key = f"{key_prefix}_sources"
    mtime_key = f"{key_prefix}_index_mtime"

    disk_mtime = _get_index_mtime(index_dir)
    cached_mtime = st.session_state.get(mtime_key, -1)

    # 首次加载，或磁盘索引已被其他用户更新
    if docs_key not in st.session_state or disk_mtime != cached_mtime:
        docs, embeddings, sources = load_index(index_dir)
        st.session_state[docs_key] = docs
        st.session_state[emb_key] = embeddings
        st.session_state[src_key] = sources
        st.session_state[mtime_key] = disk_mtime


_perf("before init_library")
_init_library("public", PUBLIC_DIR)
PRIVATE_DIR = _get_private_dir(CURRENT_USER)
_init_library("private", PRIVATE_DIR)
_perf("init_library done")


def _get_embeddings_np(key_prefix):
    np = _get_np()
    np_key = f"{key_prefix}_embeddings_np"
    ver_key = f"{key_prefix}_emb_version"
    emb_key = f"{key_prefix}_embeddings"
    emb_list = st.session_state.get(emb_key, [])
    current_ver = id(emb_list)  # 用对象 id 确保任何变更都能感知
    if np_key not in st.session_state or st.session_state.get(ver_key) != current_ver:
        if emb_list:
            st.session_state[np_key] = np.array(emb_list)
        else:
            st.session_state[np_key] = np.array([])
        st.session_state[ver_key] = current_ver
    return st.session_state[np_key]


# =========================
# 5. 缓存 LLM 客户端
# =========================
@st.cache_resource
def get_or_client():
    from openai import OpenAI
    return OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")


@st.cache_resource
def get_ds_client():
    from openai import OpenAI
    return OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")


@st.cache_resource
def get_baidu_client():
    from openai import OpenAI
    return OpenAI(
        api_key=BAIDU_TOKEN,
        base_url="https://qianfan.baidubce.com/v2",
        default_headers={"appid": BAIDU_APP_ID},
    )


# =========================
# 6. 实用功能函数
# =========================
_text_splitter_cache = None
def _get_text_splitter():
    global _text_splitter_cache
    if _text_splitter_cache is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _text_splitter_cache = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        _perf("text_splitter loaded")
    return _text_splitter_cache

SYSTEM_PROMPT = (
    "你是一个专业的知识问答助手。请基于提供的参考资料回答用户问题。"
    "如果资料中没有相关信息，请诚实说明。回答要准确、有条理、简洁。"
    "不要编造不在资料中的信息。"
)


def web_search(query):
    if not TAVILY_KEY:
        return "⚠️ 未配置搜索 Key"
    from tavily import TavilyClient
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
            # 先读取文件内容到内存，避免 pdfplumber 底层 C 库段错误导致服务崩溃
            import io
            file.seek(0)
            pdf_bytes = file.read()
            if len(pdf_bytes) < 100:
                raise ValueError("PDF 文件过小，可能已损坏")
            # 检查 PDF 魔术字节
            if not pdf_bytes[:5] == b"%PDF-":
                raise ValueError("不是有效的 PDF 文件（缺少 %PDF- 头）")
            pdf_stream = io.BytesIO(pdf_bytes)
            pages_text = []
            import pdfplumber
            with pdfplumber.open(pdf_stream) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        pages_text.append(page_text)
                    except Exception as page_err:
                        logger.warning(f"PDF 第{i+1}页解析失败: {page_err}")
                        pages_text.append("")
            text = "\n".join(pages_text)
        elif fname.endswith(".docx"):
            from docx import Document
            doc = Document(file)
            text = "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"文件解析失败 [{file.name}]: {e}", exc_info=True)
        st.error(f"解析失败: {e}")
    return text


def _get_files_dir(index_dir):
    """获取某个库的原始文件存储目录。"""
    return os.path.join(index_dir, "files")


def _save_uploaded_file(index_dir, uploaded_file):
    """保存上传的原始文件到对应库的 files/ 目录。"""
    files_dir = _get_files_dir(index_dir)
    os.makedirs(files_dir, exist_ok=True)
    dest = os.path.join(files_dir, uploaded_file.name)
    uploaded_file.seek(0)
    with open(dest, "wb") as out:
        out.write(uploaded_file.read())


def _list_uploaded_files(index_dir):
    """列出某个库已上传的原始文件。返回 [(文件名, 文件路径, 大小字符串), ...]。"""
    files_dir = _get_files_dir(index_dir)
    if not os.path.exists(files_dir):
        return []
    result = []
    for fname in sorted(os.listdir(files_dir)):
        fpath = os.path.join(files_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1048576:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / 1048576:.1f}MB"
            result.append((fname, fpath, size_str))
    return result


def process_upload(uploaded_files, target_prefix, target_dir):
    if not uploaded_files:
        return False
    file_fingerprint = str(sorted((f.name, f.size) for f in uploaded_files))
    fp_key = f"_last_upload_fp_{target_prefix}"
    if file_fingerprint == st.session_state.get(fp_key):
        return False

    try:
        all_new_chunks = []
        all_new_sources = []
        with st.spinner("正在自动解析文档并更新索引..."):
            for f in uploaded_files:
                try:
                    # 先保存原始文件（避免后续 seek 失败）
                    f.seek(0)
                    _save_uploaded_file(target_dir, f)

                    # 再解析文本
                    f.seek(0)
                    raw_text = extract_text(f)
                    if not raw_text.strip():
                        st.warning(f"文件 {f.name} 内容为空，已跳过。")
                        continue
                    chunks = _get_text_splitter().split_text(raw_text)
                    all_new_chunks.extend(chunks)
                    all_new_sources.extend([f.name] * len(chunks))
                except Exception as file_err:
                    logger.error(f"文件 {f.name} 处理失败: {file_err}", exc_info=True)
                    st.warning(f"⚠️ 文件 {f.name} 处理失败：{str(file_err)[:100]}，已跳过。")

            if all_new_chunks:
                # 分批编码，降低单次内存峰值
                batch_size = 64
                all_vecs = []
                for i in range(0, len(all_new_chunks), batch_size):
                    batch = all_new_chunks[i:i + batch_size]
                    all_vecs.extend(encode_texts(batch))

                docs_key = f"{target_prefix}_docs"
                emb_key = f"{target_prefix}_embeddings"
                src_key = f"{target_prefix}_sources"
                st.session_state[docs_key].extend(all_new_chunks)
                current_emb = list(st.session_state[emb_key])
                current_emb.extend(all_vecs)
                st.session_state[emb_key] = current_emb
                st.session_state.setdefault(src_key, []).extend(all_new_sources)
                save_index(target_dir, st.session_state[docs_key], st.session_state[emb_key], st.session_state[src_key])
                st.session_state[fp_key] = file_fingerprint
                # 同步 mtime 缓存，防止 _init_library 用旧数据覆盖
                st.session_state[f"{target_prefix}_index_mtime"] = _get_index_mtime(target_dir)
                # 递增上传组件 key，清空 file_uploader 已选文件显示
                ukey = f"_upload_ver_{target_prefix}"
                st.session_state[ukey] = st.session_state.get(ukey, 0) + 1
                st.success(f"自动导入 {len(all_new_chunks)} 个知识切片！")
                time.sleep(1)
                st.rerun()
            else:
                st.error("解析失败，未发现有效文字内容。")
    except Exception as e:
        logger.error(f"上传处理异常: {e}", exc_info=True)
        st.error(f"❌ 上传处理出错：{str(e)[:200]}")
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

_perf("before sidebar UI")
with st.sidebar:
    with st.expander(f"📚 公共知识库（{len(st.session_state.get('public_docs', []))} 切片）"):
        pub_count = len(st.session_state.get("public_docs", []))
        st.caption("所有人可搜索")

        # 文件列表
        pub_file_list = _list_uploaded_files(PUBLIC_DIR)
        if pub_file_list:
            st.caption(f"📎 已上传 {len(pub_file_list)} 个文件：")
            for fname, fpath, size_str in pub_file_list:
                if IS_ADMIN:
                    col_name, col_del = st.columns([4, 1])
                    col_name.text(f"📄 {fname} ({size_str})")
                    if col_del.button("🗑", key=f"delpub_{fname}", help=f"删除 {fname}"):
                        delete_file_from_index(PUBLIC_DIR, fname, "public_docs", "public_embeddings", "public_sources")
                        st.success(f"已删除 {fname}")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.text(f"📄 {fname} ({size_str})")

        if IS_ADMIN:
            pub_upload_key = f"upload_public_{st.session_state.get('_upload_ver_public', 0)}"
            pub_files = st.file_uploader(
                "上传到公共库",
                type=["txt", "pdf", "docx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=pub_upload_key,
            )
            if pub_files:
                process_upload(pub_files, "public", PUBLIC_DIR)

            if pub_count > 0 and len(pub_file_list) >= 2:
                if st.button("🗑️ 清空公共库", use_container_width=True, type="secondary", key="clear_pub"):
                    st.session_state.public_docs = []
                    st.session_state.public_embeddings = []
                    st.session_state.public_sources = []
                    clear_index(PUBLIC_DIR)
                    st.session_state.public_index_mtime = _get_index_mtime(PUBLIC_DIR)
                    # 同时删除原始文件
                    pub_files_dir = _get_files_dir(PUBLIC_DIR)
                    if os.path.exists(pub_files_dir):
                        import shutil
                        shutil.rmtree(pub_files_dir)
                    st.success("公共知识库已清空。")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.caption("*仅管理员可维护公共库*")

    # --- 私有知识库 ---
    with st.expander(f"🔒 我的私有库（{len(st.session_state.get('private_docs', []))} 切片）"):
        priv_count = len(st.session_state.get("private_docs", []))
        st.caption(f"用户：{CURRENT_USER}，仅自己可见")

        # 文件列表
        priv_file_list = _list_uploaded_files(PRIVATE_DIR)
        if priv_file_list:
            st.caption(f"📎 已上传 {len(priv_file_list)} 个文件：")
            for fname, fpath, size_str in priv_file_list:
                col_name, col_del = st.columns([4, 1])
                col_name.text(f"📄 {fname} ({size_str})")
                if col_del.button("🗑", key=f"delpriv_{fname}", help=f"删除 {fname}"):
                    delete_file_from_index(PRIVATE_DIR, fname, "private_docs", "private_embeddings", "private_sources")
                    st.success(f"已删除 {fname}")
                    time.sleep(0.5)
                    st.rerun()

        priv_upload_key = f"upload_private_{st.session_state.get('_upload_ver_private', 0)}"
        priv_files = st.file_uploader(
            "上传到私有库",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=priv_upload_key,
        )
        if priv_files:
            process_upload(priv_files, "private", PRIVATE_DIR)

        if priv_count > 0 and len(priv_file_list) >= 2:
            if st.button("🗑️ 清空我的私有库", use_container_width=True, type="secondary", key="clear_priv"):
                st.session_state.private_docs = []
                st.session_state.private_embeddings = []
                st.session_state.private_sources = []
                clear_index(PRIVATE_DIR)
                st.session_state.private_index_mtime = _get_index_mtime(PRIVATE_DIR)
                # 同时删除原始文件
                priv_files_dir = _get_files_dir(PRIVATE_DIR)
                if os.path.exists(priv_files_dir):
                    import shutil
                    shutil.rmtree(priv_files_dir)
                st.success("私有知识库已清空。")
                time.sleep(0.5)
                st.rerun()

    # --- 模型设置 ---
    with st.expander("⚙️ 模型设置"):
        selected_display_name = st.selectbox(
            "模型", list(model_mapping.keys()), index=0, label_visibility="collapsed"
        )

        web_on = st.toggle("🌐 联网增强", value=False)

        c1, c2 = st.columns(2)
        with c1:
            ui_top_k = st.number_input("Top-K", 1, 15, 5)
        with c2:
            ui_threshold = st.number_input("阈值", 0.0, 1.0, 0.25, step=0.05)

    # --- 修改自己的密码（所有用户可用）---
    with st.expander("🔐 修改密码"):
        old_pass = st.text_input("当前密码", type="password", key="self_old_pass")
        new_pass1 = st.text_input("新密码", type="password", key="self_new_pass1")
        new_pass2 = st.text_input("确认新密码", type="password", key="self_new_pass2")
        if st.button("✅ 确认修改", key="btn_change_pass"):
            ok, _ = verify_user(CURRENT_USER, old_pass)
            if not ok:
                st.error("当前密码错误")
            elif len(new_pass1) < 4:
                st.error("新密码至少 4 个字符")
            elif new_pass1 != new_pass2:
                st.error("两次新密码不一致")
            else:
                users = _load_users()
                users[CURRENT_USER]["password_hash"] = _hash_password(new_pass1)
                _save_users(users)
                st.success("密码修改成功，请重新登录")
                time.sleep(1)
                st.rerun()

    # --- 管理员：用户管理面板 ---
    if IS_ADMIN:
        with st.expander("👥 用户管理"):
            all_users = _load_users()
            user_list = [(u, info) for u, info in all_users.items() if u != "__meta__" and isinstance(info, dict)]

            st.caption(f"共 **{len(user_list)}** 个用户")
            for uname, uinfo in user_list:
                role_tag = "👑" if uinfo.get("role") == "admin" else "👤"
                created = uinfo.get("created_at", "未知")
                st.text(f"{role_tag} {uname}（{created}）")

            deletable = [u for u, _ in user_list if u != CURRENT_USER]
            if deletable:
                del_target = st.selectbox("选择要删除的用户", deletable, key="del_user_select")
                if st.button("❌ 删除该用户", key="btn_del_user"):
                    users = _load_users()
                    if del_target in users:
                        del users[del_target]
                        _save_users(users)
                        priv_dir = _get_private_dir(del_target)
                        clear_index(priv_dir)
                        # 清除该用户的上传文件
                        del_files_dir = _get_files_dir(priv_dir)
                        if os.path.exists(del_files_dir):
                            import shutil
                            shutil.rmtree(del_files_dir)
                        st.success(f"用户 {del_target} 已删除")
                        time.sleep(0.5)
                        st.rerun()

            resetable = [u for u, _ in user_list if u != CURRENT_USER]
            if resetable:
                reset_target = st.selectbox("选择要重置密码的用户", resetable, key="reset_user_select")
                new_pass = st.text_input("新密码", type="password", key="reset_new_pass")
                if st.button("🔄 重置密码", key="btn_reset_pass"):
                    if len(new_pass) < 4:
                        st.error("密码至少 4 个字符")
                    else:
                        users = _load_users()
                        if reset_target in users:
                            users[reset_target]["password_hash"] = _hash_password(new_pass)
                            _save_users(users)
                            st.success(f"用户 {reset_target} 密码已重置")
                            time.sleep(0.5)
                            st.rerun()

        with st.expander("📩 邀请码管理"):
            current_code = _get_invite_code()
            st.text(f"当前邀请码：{current_code if current_code else '未设置'}")
            new_code = st.text_input("新邀请码", key="new_invite_code")
            if st.button("✏️ 更新邀请码", key="btn_update_code"):
                if new_code.strip():
                    _set_invite_code(new_code.strip())
                    st.success("邀请码已更新")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("邀请码不能为空")

        with st.expander("🛠️ 云端文件浏览器"):
            st.caption(f"工作目录：`{os.getcwd()}`")

            # 检查关键文件/目录是否存在
            for check_path in ["users.json", "rag_index_v2"]:
                if os.path.exists(check_path):
                    st.success(f"✅ {check_path} 存在")
                else:
                    st.warning(f"❌ {check_path} 不存在")

            # 树形展示目录结构
            def _list_tree(path, prefix="", depth=0, max_depth=3):
                items = []
                if depth > max_depth:
                    return [f"{prefix}..."]
                try:
                    entries = sorted(os.listdir(path))
                except PermissionError:
                    return [f"{prefix}[无权限]"]
                for entry in entries:
                    if entry.startswith(".") or entry in ("ai-env", "__pycache__"):
                        continue
                    full = os.path.join(path, entry)
                    if os.path.isdir(full):
                        items.append(f"{prefix}📁 {entry}/")
                        items.extend(_list_tree(full, prefix + "  ", depth + 1, max_depth))
                    else:
                        size = os.path.getsize(full)
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1048576:
                            size_str = f"{size / 1024:.1f}KB"
                        else:
                            size_str = f"{size / 1048576:.1f}MB"
                        items.append(f"{prefix}📄 {entry} ({size_str})")
                return items

            tree = _list_tree(".")
            st.code("\n".join(tree) if tree else "(空目录)", language=None)

            st.divider()

            # 直接查看 users.json 内容
            if os.path.exists("users.json"):
                st.caption("📋 users.json 内容")
                with open("users.json", "r", encoding="utf-8") as f:
                    users_content = json.load(f)
                # 隐藏密码哈希，只显示前8位
                display_users = {}
                for k, v in users_content.items():
                    if isinstance(v, dict) and "password_hash" in v:
                        v_copy = dict(v)
                        v_copy["password_hash"] = v_copy["password_hash"][:8] + "..."
                        display_users[k] = v_copy
                    else:
                        display_users[k] = v
                st.json(display_users)

            st.divider()

            # 下载按钮区域
            st.caption("📥 文件下载")
            if os.path.exists("users.json"):
                with open("users.json", "rb") as f:
                    st.download_button(
                        "下载 users.json",
                        f.read(),
                        file_name="users.json",
                        use_container_width=True,
                        key="dl_users",
                    )

            # rag_index_v2 下的文件下载
            if os.path.exists(INDEX_ROOT):
                for dirpath, _, filenames in os.walk(INDEX_ROOT):
                    for fname in filenames:
                        fpath = os.path.join(dirpath, fname)
                        rel_path = os.path.relpath(fpath, ".").replace("\\", "/")
                        size = os.path.getsize(fpath)
                        size_str = f"{size / 1024:.1f}KB" if size < 1048576 else f"{size / 1048576:.1f}MB"
                        with open(fpath, "rb") as f:
                            st.download_button(
                                f"下载 {rel_path} ({size_str})",
                                f.read(),
                                file_name=fname,
                                use_container_width=True,
                                key=f"dl_{rel_path}",
                            )

    # --- 清空聊天记录（放在侧边栏最底部）---
    with st.expander("🧹 清空聊天记录"):
        st.caption("清空后不可恢复")
        if st.button("确认清空", use_container_width=True, type="secondary", key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()


# =========================
# 8. 核心搜索逻辑（合并公共库 + 私有库）
# =========================
def _cosine_scores(query_vec, matrix):
    """用 numpy 计算余弦相似度，替代 sklearn.cosine_similarity。"""
    np = _get_np()
    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-10:
        return np.zeros(matrix.shape[0])
    mat_norms = np.linalg.norm(matrix, axis=1)
    mat_norms = np.maximum(mat_norms, 1e-10)
    return (matrix @ query_vec) / (mat_norms * query_norm)


def search_local(query, top_k, threshold):
    query_vec = encode_query(query)
    all_results = []

    pub_docs = st.session_state.get("public_docs", [])
    pub_np = _get_embeddings_np("public")
    if pub_docs and pub_np.size > 0:
        scores = _cosine_scores(query_vec, pub_np)
        for i, s in enumerate(scores):
            if s > threshold:
                all_results.append((float(s), pub_docs[i]))

    priv_docs = st.session_state.get("private_docs", [])
    priv_np = _get_embeddings_np("private")
    if priv_docs and priv_np.size > 0:
        scores = _cosine_scores(query_vec, priv_np)
        for i, s in enumerate(scores):
            if s > threshold:
                all_results.append((float(s), priv_docs[i]))

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

_perf("script execution complete")
