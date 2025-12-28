"""
[Colab 专用脚本] TinyLlama RAG 演示
注意：此脚本需要 T4 GPU 环境 + bitsandbytes 库。

流程：
1. 读取本地 LEARNING_LOG.md
2. 切分并存入 Chroma 向量库
3. 加载 TinyLlama-1.1B (4-bit 量化)
4. 检索 + 生成回答
"""

import torch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===================== 1. 准备知识库 =====================
print("正在处理知识库...")
# ⚠️ 注意：在 Colab 运行时，需要先上传这个文件
try:
    loader = TextLoader("./LEARNING_LOG.md", encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding_model)
    print(f"✅ 知识库构建完成！共 {len(chunks)} 个切片。")
except Exception as e:
    print(f"❌ 知识库加载失败 (可能是文件没上传): {e}")
    exit()

# ===================== 2. 加载模型 (4-bit) =====================
print("\n正在加载 TinyLlama (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# ===================== 3. RAG 核心函数 =====================
def ask_with_rag(question):
    print(f"\n🤔 用户提问: {question}")
    
    # 检索
    docs = db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # 构造 Prompt (中文优化版)
    messages = [
        {
            "role": "system", 
            "content": "你是一个有用的助手。请根据下面的【参考资料】回答用户的问题。如果参考资料里没有答案，就说'我不知道'。请务必用中文回答。"
        },
        {
            "role": "user", 
            "content": f"【参考资料】:\n{context}\n\n【问题】: {question}"
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=200, 
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# ===================== 4. 测试 =====================
if __name__ == "__main__":
    q = "Minmin-winter 创建了什么账号？"
    ans = ask_with_rag(q)
    print(f"\n🤖 TinyLlama 回答:\n{ans}")