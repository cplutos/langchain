import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor

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

# 没有任何代理的情况下
# 回答：content='很抱歉，我无法为您提供实时的天气信息。您可以通过查看天气预报网站或使用天气预报应用程序来获取最新的北京天气信息。祝您在北京度过愉快的时光！如果您有任何其他问题，请随时告诉我。我会尽力帮助您。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 98, 'prompt_tokens': 16, 'total_tokens': 114, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'id': 'chatcmpl-DhFAZIVe92KOsuxLn4zHltAiYAbnO', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019e4083-043e-71a3-9af9-a3de61a5c14b-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 16, 'output_tokens': 98, 'total_tokens': 114, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
# result = model.invoke([HumanMessage(content='北京天气怎么样？')])
# print(result)

# LangChain内置了一个工具，可以轻松的使用Tavily搜索引擎作为工具
# max_results=2 ：只返回两个结果
search = TavilySearchResults(max_results=2)
# print(search.invoke('北京天气怎么样？'))

# 让模型绑定工具
tools = [search]
# model_with_tools = model.bind_tools(tools)

# 模型可以自行推理：是否需要工具去完成用户的答案
# resp = model_with_tools.invoke([HumanMessage(content='中国的首都是哪个城市？')])
#
# print(f'Model_Result_Content：{resp.content}')
# print(f'Tools_Result_Content：{resp.tool_calls}')
#
# resp2 = model_with_tools.invoke([HumanMessage(content='北京天气怎么样？')])
#
# print(f'Model_Result_Content：{resp2.content}')
# print(f'Tools_Result_Content：{resp2.tool_calls}')


# 创建代理

agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

resp = agent_executor.invoke({'messages': [HumanMessage(content='中国的首都是哪个城市？')]})
print(resp['messages'])

resp2 = agent_executor.invoke({'messages': [HumanMessage(content='北京天气怎么样？')]})
print(resp2['messages'])

print(resp2['messages'][2].content)