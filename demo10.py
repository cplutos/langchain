import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
# 创建模型
# 情感属于固定，不需要多变的温度，设置为0
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)


# class Classification(BaseModel):
#     """
#     定义一个Pydantic的数据模型，未来
#     """
#     # 文本的情感倾向，预期为字符串类型
#     sentiment: str = Field(description='文本的情感')
#
#     # 文本的攻击性，预期为1到10的整数
#     aggressiveness: int = Field(
#         description="描述文本的攻击性，数字越大越具攻击性"
#     )
#
#     # 文本使用的语言，预期使用字符串类型
#     language: str = Field(description="文本使用的语言")

class Classification(BaseModel):
    """
    定义一个Pydantic的数据模型，未来
    """
    # 文本的情感倾向，预期为字符串类型
    sentiment: str = Field(..., enum=['happy', 'neutral', 'sad'], description='文本的情感')

    # 文本的攻击性，预期为1到10的整数
    aggressiveness: int = Field(..., enum=[1, 2, 3, 4, 5],
                                description="描述文本的攻击性，数字越大越具攻击性"
                                )

    # 文本使用的语言，预期使用字符串类型
    language: str = Field(..., enum=["spanish", "english", "french", "中文", "italian"],description="文本使用的语言")


tagging_prompt = ChatPromptTemplate.from_template(
    """
        从以下段落中提取所需信息。
        只提取'Classification'类中提到的属性。
        段落：
        {input}
    """
)

chain = tagging_prompt | model.with_structured_output(Classification)

# input_text = "中国人民大学的王教授：师德败坏，做出的事情实在让我生气！"
input_text = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"

result: Classification = chain.invoke({'input': input_text})
print(result)
