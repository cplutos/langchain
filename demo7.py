import os
from typing import Optional, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.v1 import BaseModel, Field

# --- 核心配置：代理设置 ---
# 1. 请确保你已经在系统环境变量中设置了 OPENAI_API_KEY
#    或者直接在这里赋值：os.environ['OPENAI_API_KEY'] = "你的智增增Key"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 2. 关键一步：修改 Base URL 为代理平台地址
#    这样所有发给 OpenAI 的请求都会被转发到智增增
os.environ['OPENAI_API_BASE'] = "https://api.zhizengzeng.com/v1"

# --- 以下是你原本的业务逻辑，无需修改 ---
# 代理设置（如果你本地有科学上网工具，这行可以保留或删除，不影响代理平台转发）
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

# LangChain 调用链配置
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "LangchainDemo"

# 聊天机器人案例
# 初始化模型
model = ChatOpenAI(model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

persist_dir = 'chroma_data_dir'

# 一些YouTube的视频连接
# 由于API变更，无法构建向量数据库
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# docs = [] # document数组
# for url in urls:
#     # 一个YouTube的视频对应一个document
#     docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())
#
# print(len(docs))
# print(docs[0])
# # 给doc添加额外的元数据：视频发布的年份
# for doc in docs:
#     doc.metadata['publish_year'] = int(
#         datetime.datetime.strftime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
#
# print(docs[0].metadata)
# print(docs[0].page_content[:500]) # 第一个视频的字幕内容
#
# # 根据多个doc构建向量数据库
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
# split_doc = text_splitter.split_documents(docs)
# # 向量数据库的持久化
# vectorstore = Chroma.from_documents(split_doc, embeddings, persist_directory=persist_dir) #并且把向量数据库持久化


# 加载磁盘中的向量数据库
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 测试向量数据库的相似搜索
# result = vectorstore.similarity_search_with_score('how do I build a RAG agent?')
# print(result[0])
# print(result[0][0].metadata['publish_year'])


system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# pydantic 用于构建模型
class Search(BaseModel):
    """
    定义了一个数据模型
    """
    # 内容的相似性和发布年份
    query: str = Field(None, description='Similarity search query applied to video transcripts.')
    publish_year: Optional[int] = Field(None, description='Year video was published')


chain = {'question': RunnablePassthrough()} | prompt | model.with_structured_output(Search)

resp1 = chain.invoke('how do I build a RAG agent?')
print(resp1)

resp2 = chain.invoke('videos on RAG published in 2023')
print(resp2)


def retrieval(search: Search) -> List[Document]:
    _filter = None
    if search.publish_year:
        # 根据publish_year存在，得到一个检索条件
        _filter = {'publish_year': {"$eq": search.publish_year}}

    return vectorstore.similarity_search(search.query, filter=_filter)

new_chain = chain | retrieval

# result = new_chain.invoke('video on RAG published in 2023')
result = new_chain.invoke('RAG tutorial')
print([(doc.metadata['title'],doc.metadata['publish_year'])for doc in result])
