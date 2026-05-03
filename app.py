import streamlit as st
import requests

st.set_page_config(page_title="Qwen2.5 API Chatbot", layout="centered")
st.title("🤖 Qwen2.5-1.5B (via HF API)")

# แนะนำให้ใส่ Token ใน Streamlit Secrets ตอน Deploy จริง
# แต่สำหรับการทดสอบเบื้องต้นสามารถวางตรงๆ หรือใช้ st.text_input ได้
HF_TOKEN = st.sidebar.text_input("ใส่ Hugging Face Token", type="password")
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("พิมพ์ข้อความที่นี่..."):
    if not HF_TOKEN:
        st.error("กรุณาใส่ Hugging Face Token ที่แถบด้านซ้ายก่อนครับ")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังคิด..."):
                # สร้าง Prompt ตาม Format ของ Qwen
                full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                output = query({
                    "inputs": full_prompt,
                    "parameters": {"max_new_tokens": 512, "return_full_text": False}
                })
                
                # ตรวจสอบว่า API ส่ง Error มาหรือไม่ (เช่น Model กำลังโหลด)
                if isinstance(output, dict) and "error" in output:
                    response = f"❌ Error: {output['error']}"
                else:
                    try:
                        response = output[0]['generated_text']
                    except:
                        response = "ขออภัย เกิดข้อผิดพลาดในการดึงข้อมูล"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
