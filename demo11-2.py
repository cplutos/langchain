import os

from langchain_classic.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

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
# 创建模型
# 情感属于固定，不需要多变的温度，设置为0
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# 加载文档
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load() # 得到整篇文章

# 第二种
# 第一步：切割阶段
# 每一个小docs为1000个token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 第二步：map阶段
map_template = """以下是一组文档(documents)
"{docs}"
根据这个文档列表，请给出总结摘要:"""
map_prompt = PromptTemplate.from_template(map_template)
map_llm_chain = LLMChain(llm=model, prompt=map_prompt)

# 第三步：reduce阶段：（combine和最终的reduce）
reduce_template = """以下是一组总结摘要:
{docs}
将这些内容提炼成一个最终的、统一的总结摘要:"""
reduce_template = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=model, prompt=reduce_template)

'''
reduce的思路:
如果map之后文档的累积token数超过了 4000个，那么我们将递归地将文档以<= 4000 个token的批次传递给我们的 StuffDocumentsChain 来创建批量摘要。
一旦这些批量摘要的累积大小小于 4000 个token，我们将它们全部传递给 StuffDocumentsChain 最后一次，以创建最终摘要。
'''

# 定义一个combine的chain
combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name='docs')

reduce_chain = ReduceDocumentsChain(
    # 这是最终调用的链
    combine_documents_chain=combine_chain,
    # 中间的汇总的链
    collapse_documents_chain=combine_chain,
    # 将文档分组的最大令牌数
    token_max= 4000
)

# 第四步：合并所有链

map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_llm_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name='docs',
    return_intermediate_steps=False
)

# 第五步：调用最终的链
result = map_reduce_chain.invoke(split_docs)
print(result['output_text'])