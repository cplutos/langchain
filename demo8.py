import os
from typing import Optional, List

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

# pydantic：处理数据，验证数据，定义数据格式，序列化和反序列化，类型转换等等

# 定义一个数据
class Person(BaseModel):
    """
    关于一个人的数据模型
    """
    name: Optional[str] = Field(default=None, description='表示人的名字')

    hair_color: Optional[str] = Field(
        default=None, description="如果知道的话，这个人的头发颜色"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="以米为单位测量的高度"
    )

class ManyPerson(BaseModel):
    """
    数据模型：代表多个人
    """
    persons: List[Person]

# 定义自定义提示以提供指令和任何其他上下文。
# 1) 你可以在提示模板中添加示例以提高提取质量
# 2) 引入额外的参数以考虑上下文（例如，包括有关提取文本的文档的元数据。）
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的提取算法。只从未结构化文本中提取相关信息。如果你不知道要提取的属性的值，返回该属性的值为null。",
        ),
        # 请参阅有关如何使用参考记录消息历史的案例
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

# with_structured_output 模型的输出是一个结构化的数据
# chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=Person)
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=ManyPerson)
# text = '马路上走来一个女生，长长的黑头发披在肩上，大概一米七左右'
# text = "马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右。走在她旁边的是她的男朋友，叫：刘海；比她高10厘米。"
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."

resp = chain.invoke(text)
print(resp)