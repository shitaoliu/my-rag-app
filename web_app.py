import streamlit as st
import numpy as np
import pickle
import os
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from docx import Document
from tavily import TavilyClient
from datetime import datetime

# =========================
# 1️⃣ 页面配置 & 样式注入 (紧凑型布局)
# =========================
st.set_page_config(page_title="2026 增强版 RAG 助手", page_icon="🛡️", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
            /* 1. 压缩侧边栏整体容器的顶部间距 */
            [data-testid="stSidebarContent"] {
                padding-top: 1rem !important;
            }
            /* 2. 压缩侧边栏各个 Widget (输入框、下拉框) 之间的间距 */
            [data-testid="stVerticalBlock"] > div {
                gap: 0.5rem !important;
                padding-bottom: 0rem !important;
            }
            /* 3. 专门压缩 st.divider() 的高度 */
            hr {
                margin: 0.5rem 0 !important;
            }
            /* 4. 压缩标题和文本的上下边距 */
            .stMarkdown h2, .stMarkdown h3 {
                margin-bottom: 0.2rem !important;
                margin-top: 0.5rem !important;
            }
            /* 5. 缩小上传组件的占地面积 */
            [data-testid="stFileUploader"] {
                padding-bottom: 0rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()
st.title("🛡️ 智能搜索助手 (紧凑美观版)")

# =========================
# 2️⃣ 访问控制 (保持不变)
# =========================
CORRECT_PASSWORD = st.secrets.get("ACCESS_PASSWORD", "666888")

with st.sidebar:
    st.header("🔑 认证")
    input_password = st.text_input("口令", type="password", label_visibility="collapsed")
    
    if input_password != CORRECT_PASSWORD:
        st.warning("⚠️ 验证失败")
        st.stop()
    else:
        st.success("✅ 已授权")

# =========================
# 3️⃣ 安全配置与模型加载
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
INDEX_PATH = "rag_index.pkl"

if "docs" not in st.session_state:
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                st.session_state.docs = list(data["docs"])
                st.session_state.embeddings = list(data["embeddings"])
        except:
            st.session_state.docs, st.session_state.embeddings = [], []
    else:
        st.session_state.docs, st.session_state.embeddings = [], []

# =========================
# 4️⃣ 实用功能函数
# =========================
def google_search(query):
    if not TAVILY_KEY: return "⚠️ 未配置搜索 Key"
    tavily = TavilyClient(api_key=TAVILY_KEY)
    try:
        search_result = tavily.search(query=f"2026年 {query}", search_depth="advanced", max_results=3)
        results = [f"来源: {r.get('url')}\n内容: {r.get('content', '')[:700]}" for r in search_result['results']]
        return "\n\n".join(results)[:2500]
    except Exception as e:
        return f"联网搜索异常：{str(e)}"

def estimate_tokens(text):
    if not text: return 0
    zh_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    return int(zh_count * 2 + (len(text) - zh_count) * 0.5)

def extract_text(file):
    fname = file.name.lower()
    text = ""
    try:
        if fname.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        elif fname.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"解析失败: {e}")
    return text

# =========================
# 5️⃣ 侧边栏 UI (紧凑化处理)
# =========================
with st.sidebar:
    st.divider()
    st.subheader("📂 知识库")
    uploaded_files = st.file_uploader("上传文档", type=["txt", "pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files and st.button("🚀 更新索引", use_container_width=True):
        all_new_chunks = []
        with st.spinner("处理中..."):
            for f in uploaded_files:
                raw_text = extract_text(f)
                if raw_text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    all_new_chunks.extend(text_splitter.split_text(raw_text))
            if all_new_chunks:
                new_vecs = embedding_model.encode(all_new_chunks)
                st.session_state.docs.extend(all_new_chunks)
                st.session_state.embeddings.extend(list(new_vecs))
                with open(INDEX_PATH, "wb") as f:
                    pickle.dump({"docs": st.session_state.docs, "embeddings": st.session_state.embeddings}, f)
                st.success("同步完成")
                st.rerun()

    st.divider()
    st.subheader("⚙️ 设置")
    
    # =========================
    # 5️⃣ 侧边栏 UI (全模型补全 + 视觉美化)
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
        "🏢 百度文心 (官方)": "ernie-3.5-8k"
    }

    with st.sidebar:
        st.divider()
        st.subheader("📂 知识库")
        # 1. 文件上传
        uploaded_files = st.file_uploader(
            "上传文档", 
            type=["txt", "pdf", "docx"], 
            accept_multiple_files=True, 
            label_visibility="collapsed",
            key="unique_file_uploader_2026"
        )
        
        if uploaded_files and st.button("🚀 更新索引", use_container_width=True, key="update_btn"):
            # ... 这里保留你原本的索引更新逻辑 ...
            pass

        st.divider()
        st.subheader("⚙️ 模型设置")
        
        # 2. 模型选择 (加上 key 保证唯一)
        selected_display_name = st.selectbox(
            "选择回答模型", 
            list(model_mapping.keys()), 
            index=0, 
            label_visibility="collapsed",
            key="main_model_select"
        )
        
        # 3. 联网开关
        web_on = st.toggle("🌐 联网增强", value=False, key="main_web_toggle")
        
        # 4. 并排的数字输入框 (代替旧的 slider)
        col1, col2 = st.columns(2)
        with col1:
            ui_top_k = st.number_input("Top-K", 1, 10, 5, key="input_top_k")
        with col2:
            ui_threshold = st.number_input("阈值", 0.0, 1.0, 0.25, step=0.05, key="input_threshold")

# =========================
# 6️⃣ 核心对话逻辑
# =========================
def search_local(query, top_k, threshold):
    if not st.session_state.docs: return []
    query_vec = embedding_model.encode(query)
    scores = cosine_similarity([query_vec], np.array(st.session_state.embeddings))[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [st.session_state.docs[i] for i in top_indices if scores[i] > threshold]

# =========================
# 6️⃣ 核心对话逻辑 (优化流式传导)
# =========================

def llm_answer(query, context_docs, selected_display_name, web_enabled):
    all_context = ""
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 1. 组装上下文 (在生成器内部处理搜索状态)
    if context_docs:
        all_context += "【本地库资料】：\n" + "\n".join(context_docs) + "\n"
    
    if web_enabled:
        # 搜索状态显示在外面，不要阻塞 yield
        search_res = google_search(query)
        all_context += f"\n【互联网实时资料】：\n{search_res}"

    prompt_content = f"时间：{curr_time}\n资料：\n{all_context[:6500]}\n问题：{query}"
    input_tokens = estimate_tokens(prompt_content)

    # 客户端配置
    or_client = OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")
    ds_client = OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")
    baidu_client = OpenAI(api_key=BAIDU_TOKEN, base_url="https://qianfan.baidubce.com/v2", default_headers={"appid": BAIDU_APP_ID})
    
    special_clients = {"deepseek-chat": ds_client, "ernie-3.5-8k": baidu_client}
    selected_id = model_mapping[selected_display_name]

    # 构造重试队列
    retry_queue = [(special_clients.get(selected_id, or_client), selected_id, selected_display_name)]
    if selected_id not in ["openrouter/free", "deepseek-chat"]:
        retry_queue.append((or_client, "openrouter/free", "OR-Auto 避堵"))
    if selected_id != "deepseek-chat":
        retry_queue.append((ds_client, "deepseek-chat", "DeepSeek 官方"))

    # 核心迭代
    for client, m_id, label in retry_queue:
        try:
            extra_h = {"HTTP-Referer": "https://streamlit.io", "X-Title": "RAG_2026"} if client == or_client else None
            response = client.chat.completions.create(
                model=m_id,
                messages=[{"role": "user", "content": prompt_content}],
                stream=True,
                extra_headers=extra_h,
                timeout=15 # 防止死等
            )

            full_text = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text += content
                    yield content # 关键：必须直接 yield
            
            # 记录元数据
            st.session_state["last_meta"] = f"🟢 {label} | 📊 {input_tokens}/{estimate_tokens(full_text)} Tokens"
            return # 成功则退出

        except Exception as e:
            continue # 失败则尝试下一个

    yield "❌ 线路全部感冒，请检查网络或 API 状态。"

# =========================
# 7️⃣ 聊天渲染 (修复嵌套导致的折叠问题)
# =========================
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "meta" in m: st.caption(m["meta"])

if q := st.chat_input("输入问题..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    
    with st.chat_message("assistant"):
        # 1. 显示搜索状态 (独立显示，不包裹正文)
        if web_on:
            with st.status("🌐 正在同步 2026 全网动态...", expanded=False) as s:
                relevant_docs = search_local(q, ui_top_k, ui_threshold)
                s.update(label="✅ 资料检索完成", state="complete")
        else:
            relevant_docs = search_local(q, ui_top_k, ui_threshold)

        # 2. 流式输出正文 (直接在 chat_message 里吐字)
        # 不要把 st.write_stream 放在 st.status 里面！
        full_response = st.write_stream(llm_answer(q, relevant_docs, selected_display_name, web_on))
        
        # 3. 显示标签
        meta_info = st.session_state.get("last_meta", "")
        st.caption(meta_info)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response, 
            "meta": meta_info
        })

# =========================
# 7️⃣ 聊天渲染
# =========================
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "meta" in m: st.caption(m["meta"])

if q := st.chat_input("输入问题..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        relevant_docs = search_local(q, ui_top_k, ui_threshold)
        full_response = st.write_stream(llm_answer(q, relevant_docs, selected_display_name, web_on))
        meta_info = st.session_state.get("last_meta", "")
        st.caption(meta_info)
        st.session_state.messages.append({"role": "assistant", "content": full_response, "meta": meta_info})




