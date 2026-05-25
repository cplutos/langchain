import os
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.synthetic_data import create_data_generation_chain
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
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
# result = chain( # 给予一些提示词，随机生成一句话
#     {
#         'fields': ['蓝色', '黄色'],
#         "preferences": {}
#     }
# )
#
# print(result)


# 生成数据
result = chain( # 给予一些提示词，随机生成一句话
    {
        'fields': {"颜色":['蓝色', '黄色']},
        "preferences": {"style":"让它像诗歌一样"}
    }
)

print(result)