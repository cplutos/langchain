import os

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langserve import add_routes
from openai import OpenAI

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

# 定义提示模版
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。用{language}尽你所能回答所有问题。'),
    MessagesPlaceholder(variable_name='my_msg'),
])

# 5. 得到链
chain = prompt_template | model

# 保存聊天的历史记录
store = {}  # 所有用户的聊天记录都保存到store  key： sessionId


#
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'  # 每次聊天时候发送msg的key
)

config = {'configurable': {'session_id': 'zs123'}}  # 给当前会话传递一个session_id

# 第一轮
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='你好啊！ 我是DaLing')],
        'language': '中文'
    },
    config=config
)

print(resp.content)

# 第二轮
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='请问：我的名字是什么？')],
        'language': '中文'
    },
    config=config
)

print(resp2.content)

config = {'configurable': {'session_id': 'lisi12312'}}  # 给当前会话传递一个session_id

# 第二轮
for resp in do_message.stream({'my_msg': [HumanMessage(content='请给我将一个笑话')], 'language': 'English'},
                              config=config):
    # 每一个resp都是一个token
    print(resp.content, end='-')

