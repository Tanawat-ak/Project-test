import streamlit as st
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# ==========================================
# 1. ตั้งค่าหน้าเว็บ Streamlit
# ==========================================
st.set_page_config(page_title="MFU Research Grant Advisor", page_icon="🎓", layout="wide")
st.title("🎓 MFU Research Grant Disbursement Advisor")

st.write("ระบบที่ปรึกษาด้านการเบิกจ่ายเงินทุนสนับสนุนการวิจัย มหาวิทยาลัยแม่ฟ้าหลวง")

# ==========================================
# 2. ส่วน Group No. และรายชื่อสมาชิก (Sidebar)
# ==========================================
with st.sidebar:
    st.markdown("""
    ### **BDA_Project2_Group9**
    **รายชื่อสมาชิก:**
    * 6631501028 - CHAT JAISAN
    * 6631501041 - TANAKRIT SOMBOON
    * 6631501047 - TANAWAT KEUNKAEW
    * 6631501065 - NANTAWUT PANAN  
    ---
    """)
    st.info("ระบบนี้ใช้ข้อมูลจากเอกสารระเบียบการเบิกจ่ายเงินทุนวิจัย มฟล. เพื่อตอบคำถามของคุณ")
    
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = "" # ถ้าหาใน Secrets ไม่เจอ (เช่น รันบน Local) จะให้เป็นค่าว่าง หรือคุณจะใส่ Token ของคุณตรงนี้ชั่วคราวตอนทดสอบบน Local ก็ได้ครับ

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# ==========================================
# 3. โหลดเอกสารและสร้าง Vector Store
# ==========================================
@st.cache_resource(show_spinner="กำลังเตรียมฐานข้อมูลเอกสารอ้างอิง กรุณารอสักครู่...")
def initialize_vector_store():
    pdf_files = [
        "fund_service_steps.pdf",
        "learning_dev_fund_2023.pdf",
        "learning_dev_fund_eng.pdf"
    ]
    
    documents = []
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        else:
            st.sidebar.warning(f"⚠️ ไม่พบไฟล์: {file}")

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = initialize_vector_store()

# ==========================================
# 4. ส่วนคำถามแนะนำ (Suggested Questions)
# ==========================================
st.markdown("#### 💡 คำถามแนะนำ")
cols = st.columns(2)
suggestions = [
    "อาจารย์จะตั้งงบวิจัยอย่างไร",
    "ผู้วิจัยจะได้รับเงินเมื่อไหร่"
]

# ถ้าผู้ใช้กดปุ่มคำถามแนะนำ ให้เก็บคำถามลง Session State
for i, suggestion in enumerate(suggestions):
    if cols[i].button(suggestion, use_container_width=True):
        st.session_state.suggested_prompt = suggestion

st.markdown("---")

# ==========================================
# 5. ส่วนของหน้าต่างแชท (Chat Interface)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ดึงค่า prompt (ไม่ว่าจะมาจากการพิมพ์ปกติ หรือการกดปุ่มคำถามแนะนำ)
user_prompt = st.chat_input("พิมพ์ข้อความสอบถามที่นี่...")

if "suggested_prompt" in st.session_state:
    user_prompt = st.session_state.suggested_prompt
    del st.session_state.suggested_prompt # ลบออกหลังจากดึงมาใช้แล้ว

# แสดงประวัติแชท
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# จัดการเมื่อมีคำถามเข้ามา
if user_prompt:
    if not HF_TOKEN:
        st.error("กรุณาใส่ Hugging Face Token ที่แถบด้านซ้ายก่อนครับ")
    elif not retriever:
        st.error("ระบบไม่พร้อมทำงาน (ไม่พบ API Token)")
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังวิเคราะห์ระเบียบการ มฟล..."):
                try:
                    relevant_docs = retriever.invoke(user_prompt)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                    system_prompt = (
                        "คุณคือแชทบอทที่ปรึกษาด้านการจ่ายเงินทุนสนับสนุนการวิจัยของมหาวิทยาลัยแม่ฟ้าหลวง (MFU) "
                        "จงใช้ข้อมูลจากเอกสารอ้างอิง (Context) ด้านล่างนี้เพื่อตอบคำถามเท่านั้น ห้ามเดาข้อมูลเอง\n\n"
                        "Context:\n"
                        f"{context_text}"
                    )

                    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN.strip())
                    chat_history = [{"role": "system", "content": system_prompt}]
                    chat_history.extend([msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]])

                    response = client.chat_completion(
                        messages=chat_history,
                        max_tokens=512,
                        temperature=0.3 
                    )
                    
                    output_text = response.choices[0].message.content
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                    st.session_state.messages.pop()
