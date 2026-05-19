import os

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

# 初始化模型
# 注意：文档中提到 gpt-3.5-turbo-instruct，但你用的是 Chat 模型接口
# 如果报错，可以尝试将 model 改为 'gpt-3.5-turbo-instruct' 或查看平台支持的模型列表
model = ChatOpenAI(model='gpt-3.5-turbo')

# 构建消息并调用
msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好，请问你要去哪里？')
]

result = model.invoke(msg)
# print("原始响应:", result)

# 解析与链式调用
parser = StrOutputParser()
# print("解析结果:", parser.invoke(result))

# 定义提示模版
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容 翻译成{language}'),
    ('user','{text}')
])

#5. 直接使用chain来调用
chain = prompt_template | model | parser
# print("链式调用结果:", chain.invoke(msg))
# print(chain.invoke({'language': 'English', 'text': '下午还有一节课，不能去打球了'}))


# 把我们的程序部署成服务
# 创建fastAPI的应用
app = FastAPI(title='我的Langchain服务', version='V1.0', description='使用Langchain翻译任何语句的服务')

add_routes(
    app,
    chain,
    path='/chainDemo'
)

if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=8000)



