import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# --- 核心配置：代理设置 ---
# 1. 请确保你已经在系统环境变量中设置了 OPENAI_API_KEY
#    或者直接在这里赋值：os.environ['OPENAI_API_KEY'] = "你的智增增Key"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 2. 关键一步：修改 Base URL 为代理平台地址
#    这样所有发给 OpenAI 的请求都会被转发到智增增
os.environ['OPENAI_API_BASE'] = "https://api.zhizengzeng.com/v1"

# --- 以下是你原本的业务逻辑，无需修改 ---
# 代理设置（如果你本地有科学上网工具，这行可以保留或删除，不影响代理平台转发）
# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'

# LangChain 调用链配置
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "LangchainDemo"

# 聊天机器人案例
# 初始化模型
model = ChatOpenAI(model='gpt-3.5-turbo')

# 准备测试数据 ，假设我们提供的文档数据如下：
documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

# 实例化向量数据库
vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

# 相似度的查询：返回相似的分数，分数越低相似度越高
# print(vector_store.similarity_search_with_score('咖啡猫'))

# 检索器: bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# print(retriever.batch(['咖啡猫', '鲨鱼']))

# 提示模版
message = """
使用提供的上下文仅回答这个问题：
{question}
上下文：
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([
    ('human', message)
])

chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model

resp = chain.invoke('请介绍一下猫？')

print(resp.content)
