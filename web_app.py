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
import requests
import json
from tavily import TavilyClient
from datetime import datetime

# =========================
# 1️⃣ 页面配置 & 样式
# =========================
st.set_page_config(page_title="全格式 RAG 助手", page_icon="📑", layout="wide")
st.title("📑 大模型搜索 (2026 增强版)")

# =========================
# 2️⃣ 安全与配置 (从 Secrets 读取)
# =========================
# 优先从 Secrets 读取，本地调试时若为空则会报错提示
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", "")
DS_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
SF_API_KEY = st.secrets.get("SF_API_KEY", "")

# 百度配置：在侧边栏提供手动输入，因为 Token 只有 24 小时寿命
DEFAULT_BAIDU_TOKEN = st.secrets.get("BAIDU_BEARER_TOKEN", "")
BAIDU_APP_ID = st.secrets.get("BAIDU_APP_ID", "")

# 初始化模型
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-zh")

model = load_embedding_model()

# 相对路径适配 (Streamlit Cloud 专用)
INDEX_PATH = "rag_index.pkl"

def load_local_index():
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            return list(data["docs"]), list(data["embeddings"])
        except Exception:
            pass
    return [], []

if "docs" not in st.session_state:
    st.session_state.docs, st.session_state.embeddings = load_local_index()

# =========================
# 3️⃣ 功能函数
# =========================

def google_search(query, default_location="深圳"):
    if not TAVILY_KEY: return "⚠️ 未配置 Tavily API Key"
    tavily = TavilyClient(api_key=TAVILY_KEY)
    query_l = query.lower()
    
    # A. 自动补全
    if any(k in query_l for k in ["天气", "温度", "降雨"]):
        if not any(city in query_l for city in ["北京", "上海", "广州", "深圳"]):
            query = f"{default_location} {query} 实时"
    
    if any(k in query_l for k in ["近况", "现状", "动态", "怎么了"]) or len(query_l) < 5:
        if "2026" not in query: query = f"2026年 {query} 最新动态 现状"
    elif "202" not in query:
        query = f"2026年 {query}"

    try:
        search_result = tavily.search(query=query, search_depth="advanced", max_results=3)
        results = []
        for r in search_result['results']:
            snippet = r.get('content', '')[:700] 
            results.append(f"来源: {r.get('url')}\n内容: {snippet}")
        context = "\n\n".join(results)
        return context[:2500] # 百度 5120 token 限制保护
    except Exception as e:
        return f"搜索失败：{str(e)}"

def needs_web_search(query):
    query_l = query.strip().lower()
    ks = ["今天", "战况", "最新", "2026", "股价", "冲突", "谁是", "近况", "天气", "怎么了", "现状"]
    if any(k in query_l for k in ks) or (2 <= len(query_l) <= 5):
        return True
    return False

def extract_text(file):
    fname = file.name.lower()
    text = ""
    try:
        if fname.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"
        elif fname.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"解析文件 {fname} 出错: {e}")
    return text

# =========================
# 4️⃣ 侧边栏 UI
# =========================
with st.sidebar:
    st.header("🔑 鉴权设置")
    current_baidu_token = st.text_input("百度 bce-v3 Token", value=DEFAULT_BAIDU_TOKEN, type="password", help="Token 经常过期，请在此实时更新")
    
    st.divider()
    st.header("📂 知识库管理")
    uploaded_files = st.file_uploader("上传文档", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if uploaded_files and st.button("🚀 存入索引"):
        all_new_chunks = []
        with st.spinner("解析中..."):
            for f in uploaded_files:
                raw_text = extract_text(f)
                if raw_text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                    chunks = text_splitter.split_text(raw_text)
                    all_new_chunks.extend(chunks)
            if all_new_chunks:
                new_vecs = model.encode(all_new_chunks)
                st.session_state.docs.extend(all_new_chunks)
                st.session_state.embeddings.extend(list(new_vecs))
                with open(INDEX_PATH, "wb") as f:
                    pickle.dump({"docs": st.session_state.docs, "embeddings": st.session_state.embeddings}, f)
                st.success(f"已存入 {len(all_new_chunks)} 个知识切片")
                st.rerun()

    st.divider()
    st.header("🤖 模型选择")
    model_option = st.selectbox("首选回答模型：", ["自动轮询 (推荐)", "仅使用 DeepSeek", "仅使用 硅基流动", "仅使用 百度文心"])
    web_search_enabled = st.checkbox("开启联网增强", value=True)
    ui_top_k = st.slider("检索数量", 1, 10, 3)
    ui_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.35)

# =========================
# 5️⃣ 核心对话逻辑
# =========================
def search_local(query, top_k, threshold):
    if not st.session_state.docs: return []
    query_vec = model.encode(query)
    scores = cosine_similarity([query_vec], np.array(st.session_state.embeddings))[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [st.session_state.docs[i] for i in top_indices if scores[i] > threshold]

def llm_answer(query, context_docs, selected_mode, web_enabled):
    all_context = ""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    if context_docs:
        all_context += "【本地库资料】：\n" + "\n".join(context_docs) + "\n"
    
    if web_enabled and needs_web_search(query):
        with st.status("🔍 检索 2026 最新动态...", expanded=False) as s:
            web_info = google_search(query)
            all_context += f"\n【互联网实时资料】：\n{web_info}"
            s.update(label="✅ 获取成功", state="complete")

    system_prompt = f"当前时间：{current_time}，位置：深圳。"
    prompt_content = f"{system_prompt}\n请结合资料回答。注意时效性，天气单位用摄氏度。\n\n资料：\n{all_context[:3000]}\n\n问题：{query}"
    messages = [{"role": "user", "content": prompt_content}]

    # 初始化客户端
    ds_client = OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")
    sf_client = OpenAI(api_key=SF_API_KEY, base_url="https://api.siliconflow.cn/v1")
    baidu_client = OpenAI(api_key=current_baidu_token, base_url="https://qianfan.baidubce.com/v2", default_headers={"appid": BAIDU_APP_ID})

    all_models = {
        "百度文心": (baidu_client, "ernie-3.5-8k", "百度文心"),
        "DeepSeek": (ds_client, "deepseek-chat", "DeepSeek"),
        "硅基流动": (sf_client, "deepseek-ai/DeepSeek-V3", "硅基流动")
    }

    # 排序逻辑
    if selected_mode == "自动轮询 (推荐)":
        active_list = [all_models["百度文心"], all_models["DeepSeek"], all_models["硅基流动"]]
    else:
        target = selected_mode.replace("仅使用 ", "")
        active_list = [all_models[target]] + [m for n, m in all_models.items() if n != target]

    for client, m_name, label in active_list:
        try:
            with st.status(f"🚀 {label} 思考中...", expanded=False):
                res = client.chat.completions.create(model=m_name, messages=messages, temperature=0.7, timeout=45)
                return res.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "401" in err: st.warning(f"🔑 {label} Token过期或错误")
            elif "400" in err: st.warning(f"📏 {label} 内容过长，已尝试截断")
            else: st.warning(f"⚠️ {label} 异常: {err[:50]}")
            continue
    return "抱歉，所有模型均暂时不可用。"

# =========================
# 6️⃣ 渲染聊天
# =========================
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if q := st.chat_input("问问你的文档或搜索最新资讯..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    
    with st.spinner("正在处理..."):
        relevant = search_local(q, ui_top_k, ui_threshold)
        # 即使本地没有，只要开启联网，llm_answer 内部也会处理
        ans = llm_answer(q, relevant, model_option, web_search_enabled)
        
    with st.chat_message("assistant"): st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})




