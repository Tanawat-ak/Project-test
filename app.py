import streamlit as st
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Grant Advisor", layout="centered")
st.title("🎓 ระบบที่ปรึกษาด้านทุนวิจัย (via HF API)")

# รับ Token และกำหนดโมเดล
HF_TOKEN = st.sidebar.text_input("ใส่ Hugging Face Token", type="password")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# ==========================================
# 1. โหลดเอกสารและสร้าง Vector Store (ทำครั้งเดียว)
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

    # ตัดคำ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # สร้าง Embeddings (ใช้ CPU หรือ GPU อัตโนมัติ)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # สร้าง FAISS Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# โหลดตัวค้นหา
retriever = initialize_vector_store()

# ==========================================
# 2. ส่วนของหน้าต่างแชท (Chat Interface)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติแชท
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("ตัวอย่างเช่น: อาจารย์จะตั้งงบวิจัยอย่างไร?"):
    if not HF_TOKEN:
        st.error("กรุณาใส่ Hugging Face Token ที่แถบด้านซ้ายก่อนครับ")
    elif not retriever:
        st.error("ระบบไม่พร้อมทำงาน เนื่องจากไม่พบไฟล์ PDF สำหรับอ้างอิง")
    else:
        # นำคำถามขึ้นจอ
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูลจากระเบียบการ และวิเคราะห์คำตอบ..."):
                try:
                    # 1. ค้นหาเอกสารที่เกี่ยวข้อง (Retrieval)
                    relevant_docs = retriever.invoke(prompt)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # 2. สร้าง System Prompt (รวมบริบทที่ค้นหามาได้)
                    system_prompt = (
                        "คุณคือแชทบอทที่ปรึกษาด้านการจ่ายเงินทุนสนับสนุนการวิจัยของมหาวิทยาลัย "
                        "จงใช้ข้อมูลจากเอกสารอ้างอิง (Context) ด้านล่างนี้เพื่อตอบคำถามเท่านั้น ห้ามเดาข้อมูลเอง\n\n"
                        "นี่คือตัวอย่างสไตล์การตอบคำถามที่คุณควรเลียนแบบ (ให้ตอบสั้น กระชับ เป็นข้อๆ):\n"
                        "คำถาม: อาจารย์จะตั้งงบวิจัยอย่างไร\n"
                        "คำตอบ: อาจารย์สามารถตั้งงบวิจัยได้ตามค่าใช้จ่ายที่คาดว่าจะเกิดขึ้นจริงในโครงการ โดยอ้างอิงรายการค่าใช้จ่ายตามระเบียบของมหาวิทยาลัย\n\n"
                        "Context:\n"
                        f"{context_text}"
                    )

                    # 3. เตรียมประวัติสนทนาให้ API
                    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN.strip())
                    chat_history = [{"role": "system", "content": system_prompt}]
                    
                    # เอาประวัติเก่ามาต่อท้าย (ส่งเฉพาะ user กับ assistant)
                    chat_history.extend([msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]])

                    # 4. เรียกใช้งาน Chat API
                    response = client.chat_completion(
                        messages=chat_history,
                        max_tokens=512,
                        temperature=0.3 # ค่า 0.3 เพื่อให้ตอบอิงตามข้อมูลจริงไม่แต่งเติม
                    )
                    
                    # ดึงข้อความและแสดงผล
                    output_text = response.choices[0].message.content
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                    # ลบคำถามล่าสุดออกเพื่อให้พิมพ์ถามใหม่ได้
                    st.session_state.messages.pop()
