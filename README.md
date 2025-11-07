# mcp-demo

一个基于 python 本地运行 MCP 协议的 server 和 client 样例

1. 安装 uv
```
pip install uv
```
or
```
conda install uv
```

2. 创建工程目录
```
uv init mcp-demo
```

3. 运行 server.py
server中包含了MCP服务的工具，可以与LLM无关。但本项目由于需要分析所以其中包含了LLM调用的工具函数。

4. 运行 client.py
客户端，需要有工具规划和调用的能力，例如cursor和cline或其他agent的调用。
