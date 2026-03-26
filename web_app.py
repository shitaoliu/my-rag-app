import streamlit as st
import numpy as np
import pickle
import os
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
from docx import Document
from tavily import TavilyClient
from datetime import datetime
import pdfplumber

# =========================
# 1️⃣ 页面配置 & 样式
# =========================
st.set_page_config(page_title="2026 安全增强版 RAG 助手", page_icon="🛡️", layout="wide")
st.title("🛡️ 智能搜索助手 (全量模型集成版)")

# =========================
# 2️⃣ 访问控制 (防止 Token 盗刷)
# =========================
CORRECT_PASSWORD = st.secrets.get("ACCESS_PASSWORD", "666888")

with st.sidebar:
    st.header("🔑 身份验证")
    input_password = st.text_input("输入访问口令", type="password", help="请输入预设密码以解锁功能")
    
    if input_password != CORRECT_PASSWORD:
        st.warning("⚠️ 请输入正确的访问口令")
        st.stop()  # 阻断后续所有代码运行
    else:
        st.success("✅ 认证通过")

# =========================
# 3️⃣ 安全配置与模型加载
# =========================
# 读取所有必要的 API Keys
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", "")
DS_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
SF_API_KEY = st.secrets.get("SF_API_KEY", "")
BAIDU_TOKEN = st.secrets.get("BAIDU_BEARER_TOKEN", "")
BAIDU_APP_ID = st.secrets.get("BAIDU_APP_ID", "")
OR_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

@st.cache_resource
def load_embedding_model():
    # 2026年主流的中文向量模型
    return SentenceTransformer("BAAI/bge-small-zh")

embedding_model = load_embedding_model()
INDEX_PATH = "rag_index.pkl"

# 初始化 Session State
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
# 4️⃣ 功能函数 (解析与搜索)
# =========================

def google_search(query):
    if not TAVILY_KEY: return "⚠️ 未配置搜索 Key"
    tavily = TavilyClient(api_key=TAVILY_KEY)
    try:
        # 强制加入 2026 时间戳确保实时性
        search_result = tavily.search(query=f"2026年 {query}", search_depth="advanced", max_results=3)
        results = [f"来源: {r.get('url')}\n内容: {r.get('content', '')[:700]}" for r in search_result['results']]
        return "\n\n".join(results)[:2500]
    except Exception as e:
        return f"联网搜索异常：{str(e)}"

def needs_web_search(query):
    ks = ["今天", "战况", "最新", "2026", "股价", "现状", "天气", "动态"]
    return any(k in query.lower() for k in ks)

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
        st.error(f"文件解析失败: {e}")
    return text

# =========================
# 5️⃣ 侧边栏 UI (模型配置)
# =========================

with st.sidebar:
    st.divider()
    st.header("📂 知识库管理")
    uploaded_files = st.file_uploader("上传文档 (PDF/Word/TXT)", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if uploaded_files and st.button("🚀 更新索引"):
        all_new_chunks = []
        with st.spinner("正在构建向量索引..."):
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
                st.success(f"成功导入 {len(all_new_chunks)} 条知识")
                st.rerun()

    st.divider()
    st.header("⚙️ 对话与模型设置")
    
    # 按照实际测试表现排序的 ID 映射表
    model_mapping = {
        "Step-3.5-Flash (首选/中文报销单专家)": "stepfun/step-3.5-flash:free",
        "OpenRouter-Auto (万能自动避堵)": "openrouter/free",
        "GLM-4.5-Air (深度推理/长文本)": "z-ai/glm-4.5-air:free",
        "Gemma-3-27B (Google旗舰性能)": "google/gemma-3-27b-it:free",
        "Nemotron-3-Super (120B超大型)": "nvidia/nemotron-3-super-120b-a12b:free",
        "Trinity-Large (极速响应0.9s)": "arcee-ai/trinity-large-preview:free",
        "Liquid-Instruct (极速/1.0s)": "liquid/lfm-2.5-1.2b-instruct:free",
        "Liquid-Thinking (极速思维链)": "liquid/lfm-2.5-1.2b-thinking:free",
        "Nemotron-3-Nano (Nvidia混合架构)": "nvidia/nemotron-3-nano-30b-a3b:free",
        "Trinity-Mini (稳/1.8s)": "arcee-ai/trinity-mini:free",             
        "Gemma-3-12B (平衡型)": "google/gemma-3-12b-it:free",
        "Gemma-3n-e4b (稳定型)": "google/gemma-3n-e4b-it:free",            
        "Nemotron-Nano-9B (Nvidia)":"nvidia/nemotron-nano-9b-v2:free",      
        "Gemma-3-4B (轻量)": "google/gemma-3-4b-it:free",                   
        "Gemma-3n-e2b (极轻)":  "google/gemma-3n-e2b-it:free",              
        "Nemotron-VL (多模态备选)": "nvidia/nemotron-nano-12b-v2-vl:free",  
        "DeepSeek-V3 (官方接口/付费稳定)": "deepseek-chat",
        "百度文心 (官方接口/付费稳定)": "ernie-3.5-8k"
    }

    selected_display_name = st.selectbox("首选回答模型：", list(model_mapping.keys()), index=0)
    web_on = st.checkbox("🌐 开启 2026 联网增强", value=True)

with st.expander("🔍 高级检索参数"):
    ui_top_k = st.slider("匹配条数 (Top-K)", 1, 20, 5)
    ui_threshold = st.slider("语义相关度阈值", 0.0, 1.0, 0.25)

# =========================
# 6️⃣ 核心对话逻辑 (修正后的智能路由)
# =========================

def search_local(query, top_k, threshold):
    if not st.session_state.docs: return []
    query_vec = embedding_model.encode(query)
    scores = cosine_similarity([query_vec], np.array(st.session_state.embeddings))[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [st.session_state.docs[i] for i in top_indices if scores[i] > threshold]

def llm_answer(query, context_docs, selected_display_name, web_enabled):
    all_context = ""
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 1. 组装上下文
    if context_docs:
        all_context += "【本地库资料】：\n" + "\n".join(context_docs) + "\n"
    if web_enabled and needs_web_search(query):
        with st.status("🔍 联网检索实时动态...", expanded=False):
            all_context += f"\n【互联网资料】：\n{google_search(query)}"

    prompt_content = f"当前时间：{curr_time}\n资料：\n{all_context[:6000]}\n问题：{query}"
    messages = [{"role": "user", "content": prompt_content}]

    # 2. 初始化所有客户端
    or_client = OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")
    ds_client = OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")
    baidu_client = OpenAI(api_key=BAIDU_TOKEN, base_url="https://qianfan.baidubce.com/v2", default_headers={"appid": BAIDU_APP_ID})

    # 3. 建立 Client 映射表 (修正核心：根据 ID 找正确的 Client)
    special_clients = {
        "deepseek-chat": ds_client,
        "ernie-3.5-8k": baidu_client
    }

    selected_id = model_mapping[selected_display_name]

    # 4. 构造【多级容错重试队列】 (Client, ID, Label)
    primary_client = special_clients.get(selected_id, or_client)
    retry_queue = [(primary_client, selected_id, selected_display_name)]

    # 自动加入免费保底 (如果首选挂了，且首选不是 Auto)
    if selected_id not in ["openrouter/free", "deepseek-chat", "ernie-3.5-8k"]:
        retry_queue.append((or_client, "openrouter/free", "OpenRouter-Auto (自动分流)"))

    # 最后官方付费保底
    if selected_id != "deepseek-chat":
        retry_queue.append((ds_client, "deepseek-chat", "DeepSeek-V3 (官方兜底)"))

    # 5. 执行调用循环
    for client, m_id, label in retry_queue:
        try:
            with st.status(f"🚀 {label} 响应中...", expanded=False):
                # 只有 OpenRouter 需要 Header 校验
                extra_h = {}
                if client == or_client:
                    extra_h = {"HTTP-Referer": "https://streamlit.io", "X-Title": "Tax_RAG_2026"}
                
                response = client.chat.completions.create(
                    model=m_id,
                    messages=messages,
                    stream=True,
                    extra_headers=extra_h if extra_h else None
                )

                full_text = ""
                for chunk in response:
                    # 兼容各厂商返回的差异化结构
                    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_text += content
                        yield content
                
                if full_text: return  # 只要有一个模型成功吐字，就结束整个流程

        except Exception as e:
            st.warning(f"⚠️ {label} 异常，正在尝试切换备选路线... (错误提示: {str(e)[:40]})")
            continue

    yield "❌ 抱歉，当前所有模型线路均已限流或配置有误，请稍后再试。"

# =========================
# 7️⃣ 聊天渲染
# =========================
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if q := st.chat_input("问问文档里的打车费，或搜索 2026 最新动态..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    
    with st.chat_message("assistant"):
        # 本地语义检索
        relevant_docs = search_local(q, ui_top_k, ui_threshold)
        
        # 流式回答渲染
        full_response = st.write_stream(llm_answer(q, relevant_docs, selected_display_name, web_on))
        
        # 存入历史记录
        st.session_state.messages.append({"role": "assistant", "content": full_response})




