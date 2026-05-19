from langserve import RemoteRunnable

if __name__=="__main__":
    client = RemoteRunnable('http://0.0.0.0:8000/chainDemo/')
    print(client.invoke({'language': 'italian', 'text': '你好！'}))
