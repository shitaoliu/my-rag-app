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
st.set_page_config(page_title="安全增强版 RAG 助手", page_icon="🛡️", layout="wide")
st.title("🛡️ 智能搜索助手 (2026 安全强化版)")

# =========================
# 2️⃣ 访问控制 (防止 Token 盗刷)
# =========================
# 建议在 Streamlit Secrets 中设置 ACCESS_PASSWORD，否则默认为 666888
CORRECT_PASSWORD = st.secrets.get("ACCESS_PASSWORD", "666888")

with st.sidebar:
    st.header("🔑 身份验证")
    input_password = st.text_input("输入访问口令", type="password", help="请输入预设密码以解锁功能")
    
    if input_password != CORRECT_PASSWORD:
        st.warning("⚠️ 请输入正确的访问口令")
        st.info("受安全策略影响，未授权用户无法调用后端模型。")
        st.stop()  # 阻断后续所有代码运行
    else:
        st.success("✅ 认证通过")

# =========================
# 3️⃣ 安全配置读取
# =========================
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", "")
DS_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
SF_API_KEY = st.secrets.get("SF_API_KEY", "")
BAIDU_TOKEN = st.secrets.get("BAIDU_BEARER_TOKEN", "")
BAIDU_APP_ID = st.secrets.get("BAIDU_APP_ID", "")

# 初始化模型
@st.cache_resource
def load_embedding_model():
    # 注意：Streamlit Cloud 下载此模型可能需要几分钟
    return SentenceTransformer("BAAI/bge-small-zh")

embedding_model = load_embedding_model()

INDEX_PATH = "rag_index.pkl"

if "docs" not in st.session_state:
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                st.session_state.docs, st.session_state.embeddings = list(data["docs"]), list(data["embeddings"])
        except:
            st.session_state.docs, st.session_state.embeddings = [], []
    else:
        st.session_state.docs, st.session_state.embeddings = [], []

# =========================
# 4️⃣ 功能函数
# =========================

def google_search(query, default_location="深圳"):
    if not TAVILY_KEY: return "⚠️ 未配置搜索 Key"
    tavily = TavilyClient(api_key=TAVILY_KEY)
    query_l = query.lower()
    
    # 智能补全逻辑
    if any(k in query_l for k in ["天气", "温度", "降雨"]) and not any(city in query_l for city in ["北京", "上海", "广州", "深圳"]):
        query = f"{default_location} {query} 实时"
    
    if any(k in query_l for k in ["近况", "现状", "动态", "怎么了"]) or len(query_l) < 5:
        if "2026" not in query: query = f"2026年 {query} 最新现状"
    if "202" not in query:
        query = f"2026年 {query}"

    try:
        search_result = tavily.search(query=query, search_depth="advanced", max_results=3)
        results = [f"来源: {r.get('url')}\n内容: {r.get('content', '')[:700]}" for r in search_result['results']]
        return "\n\n".join(results)[:2500]
    except Exception as e:
        return f"联网搜索异常：{str(e)}"

def needs_web_search(query):
    query_l = query.strip().lower()
    ks = ["今天", "战况", "最新", "2026", "股价", "冲突", "谁是", "近况", "天气", "现状"]
    return any(k in query_l for k in ks) or (2 <= len(query_l) <= 5)

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
# 5️⃣ 侧边栏 UI (管理)
# =========================
with st.sidebar:
    st.divider()
    st.header("📂 知识库管理")
    uploaded_files = st.file_uploader("上传文档 (PDF/Word/TXT)", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if uploaded_files and st.button("🚀 更新索引"):
        all_new_chunks = []
        with st.spinner("正在提取向量..."):
            for f in uploaded_files:
                raw_text = extract_text(f)
                if raw_text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len)
                    chunks = text_splitter.split_text(raw_text)
                    all_new_chunks.extend(chunks)
            if all_new_chunks:
                new_vecs = embedding_model.encode(all_new_chunks)
                st.session_state.docs.extend(all_new_chunks)
                st.session_state.embeddings.extend(list(new_vecs))
                with open(INDEX_PATH, "wb") as f:
                    pickle.dump({"docs": st.session_state.docs, "embeddings": st.session_state.embeddings}, f)
                st.success(f"成功导入 {len(all_new_chunks)} 条知识")
                st.rerun()

    st.divider()
    st.header("⚙️ 对话与模型设置") # 合并为一个标题，更整洁
    model_option = st.selectbox(
        "首选回答模型：", 
        ["自动轮询 (推荐)", "仅使用 Gemini-Flash (免费)", "仅使用 GPT-4o-Mini (极速)", "仅使用 Claude-3.5-Sonnet", "仅使用 DeepSeek-V3","仅适用 百度文心"],
        help = "Gemini 和 GPT 系列通过 OpenRouter 接入，支持超长上下文" # 增加悬浮提示
    )
    web_on = st.checkbox("🌐 开启 2026 联网增强", value=True)

with st.expander("🔍 高级检索参数"): # 将滑块收纳进折叠栏，节省空间
    ui_top_k = st.slider("匹配条数 (Top-K)", 1, 20, 5, help="增加条数可减少打车记录等信息的丢失")
    ui_threshold = st.slider("语义相关度阈值", 0.0, 1.0, 0.30)

# =========================
# 6️⃣ 核心对话逻辑
# =========================
def search_local(query, top_k, threshold):
    if not st.session_state.docs: return []
    query_vec = embedding_model.encode(query)
    scores = cosine_similarity([query_vec], np.array(st.session_state.embeddings))[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [st.session_state.docs[i] for i in top_indices if scores[i] > threshold]

def llm_answer(query, context_docs, selected_mode, web_enabled):
    all_context = ""
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 1. 组装上下文
    if context_docs:
        all_context += "【本地库资料】：\n" + "\n".join(context_docs) + "\n"
    
    if web_enabled and needs_web_search(query):
        with st.status("🔍 检索实时动态...", expanded=False) as s:
            web_info = google_search(query)
            all_context += f"\n【互联网资料】：\n{web_info}"
            s.update(label="✅ 搜索成功", state="complete")

    prompt_content = f"当前时间：{curr_time}\n资料：\n{all_context[:5000]}\n问题：{query}"
    messages = [{"role": "user", "content": prompt_content}]

    # 2. 初始化所有客户端 (包含 OpenRouter)
    OR_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    # OpenRouter 客户端 (一套代码调所有国外模型)
    or_client = OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")
    
    # 原有的国内模型客户端
    ds_client = OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")
    baidu_client = OpenAI(api_key=BAIDU_TOKEN, base_url="https://qianfan.baidubce.com/v2", default_headers={"appid": BAIDU_APP_ID})

    # 3. 定义模型字典 (模型 ID 需严格遵守 OpenRouter 规范)
    clients = {
        # 备选 1：目前 OpenRouter 官方推荐的最强免费版 (Gemini 2.0 Flash 体验版)
        "Gemini-Flash (免费)": (or_client, "google/gemini-2.0-flash-exp:free"), 
        # 备选 2：如果上面那个报 404，请换成这个 (Gemini 1.5 Flash 完整版)
        # "Gemini-Flash (免费)": (or_client, "google/gemini-flash-1.5:free"),
        "GPT-4o-Mini (极速)": (or_client, "openai/gpt-4o-mini"),
        "Claude-3.5-Sonnet": (or_client, "anthropic/claude-3.5-sonnet"),
        "DeepSeek-V3": (ds_client, "deepseek-chat"),
        "百度文心": (baidu_client, "ernie-3.5-8k")
    }

    # 4. 确定调用逻辑
    if "仅使用" in selected_mode:
        target = selected_mode.replace("仅使用 ", "")
        active_labels = [target] if target in clients else ["DeepSeek-V3"]
    else:
        # 自动轮询模式：你可以根据喜好排顺序
        active_labels = ["Gemini-Flash (免费)","GPT-4o-Mini (极速)","Claude-3.5-Sonnet", "DeepSeek-V3", "百度文心"]

    # 5. 流式输出循环
    for label in active_labels:
        client, m_id = clients[label]
        try:
            with st.status(f"🚀 {label} 思考中...", expanded=False):
                # 💡 OpenRouter 必须包含额外的 headers 才能正常统计
                response = client.chat.completions.create(
                    model=m_id,
                    messages=messages,
                    stream=True,
                    extra_headers={
                        "HTTP-Referer": "https://streamlit.io", # 选填
                        "X-Title": "My-RAG-App" 
                    }
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return 
        except Exception as e:
            # 这样如果报错，你会看到具体的错误原因（比如：Key 没钱了，或者 ID 错位）
            st.error(f"❌ {label} 调用细节: {str(e)}") 
            continue
    yield "所有模型均不可用，请检查 OpenRouter 余额或网络。"

# =========================
# 7️⃣ 聊天渲染
# =========================
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if q := st.chat_input("问问你的文档或搜索 2026 最新动态..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    
    with st.chat_message("assistant"):
        # 1. 先进行本地搜索（非流式）
        relevant_docs = search_local(q, ui_top_k, ui_threshold)
        
        # 2. 调用流式回答
        # st.write_stream 会自动处理生成器并实时显示文字
        full_response = st.write_stream(llm_answer(q, relevant_docs, model_option, web_on))
        
        # 3. 将最终完整的回复存入历史记录
        st.session_state.messages.append({"role": "assistant", "content": full_response})




