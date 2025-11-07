import asyncio
import os
import json
from typing import Optional, List
from contextlib import AsyncExitStack
from datetime import datetime
import re
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
load_dotenv()


class MCPClient:

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("未找到 API Key，请在 .env 文件中设置 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str):
        # 对服务器脚本进行判断，只允许是 .py 或 .js
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        # 确定启动命令，.py 用 python，.js 用 node
        command = "python" if is_python else "node"

        # 构造 MCP 所需的服务器参数，包含启动命令、脚本路径参数、环境变量（为 None 表示默认）
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        # 启动 MCP 工具服务进程（并建立 stdio 通信）
        self.stdio, self.write = await self.exit_stack.enter_async_context(stdio_client(server_params))

        # 创建 MCP 客户端会话对象
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # 初始化会话
        await self.session.initialize()

        # 获取工具列表并打印
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        response = await self.session.list_tools()

        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            } for tool in response.tools
        ]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_filename = f"search_{query}_{timestamp}.md"
        md_path = os.path.join("./search_results", md_filename)

        query = query.strip() + f" [md_filename={md_filename}] [md_path={md_path}]"
        messages = [{"role": "user", "content": query}]

        tool_plan = await self.plan_tool_usage(query, available_tools)
        tool_outputs = {}


        for step in tool_plan:
            tool_name = step["name"]
            tool_args = step["arguments"]

            for key, val in tool_args.items():
                if isinstance(val, str) and val.startswith("{{") and val.endswith("}}"):
                    ref_key = val.strip("{} ")
                    resolved_val = tool_outputs.get(ref_key, val)
                    tool_args[key] = resolved_val

            if tool_name == "analyze_sentiment" and "filename" not in tool_args:
                tool_args["filename"] = md_filename
            if tool_name == "send_email_with_attachment" and "attachment_path" not in tool_args:
                tool_args["attachment_path"] = md_path

            result = await self.session.call_tool(tool_name, tool_args)

            tool_outputs[tool_name] = result.content[0].text
            messages.append({
                "role": "tool",
                "tool_call_id": tool_name,
                "content": result.content[0].text
            })

        result_messages = "文档已生成并保存在: " + md_path

        return result_messages

    async def chat_loop(self):
        print("\n🤖 MCP 客户端已启动！输入 'quit' 退出")

        while True:
            try:
                query = input("\nInput: ").strip()
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print(f"\n🤖 AI: {response}")

            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def plan_tool_usage(self, query: str, tools: List[dict]) -> List[dict]:
        print("\n📤 提交给大模型的工具定义:")
        print(json.dumps(tools, ensure_ascii=False, indent=2))
        tool_list_text = "\n".join([
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in tools
        ])
        system_prompt = {
            "role": "system",
            "content": (
                "你是一个智能任务规划助手，用户会给出一句自然语言请求。\n"
                "你只能从以下工具中选择（严格使用工具名称）：\n"
                f"{tool_list_text}\n"
                "如果多个工具需要串联，后续步骤中可以使用 {{上一步工具名}} 占位。\n"
                "返回格式：JSON 数组，每个对象包含 name 和 arguments 字段。\n"
                "不要返回自然语言，不要使用未列出的工具名。"
            )
        }

        planning_messages = [
            system_prompt,
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=planning_messages,
            extra_body={"enable_thinking": False},
            stream=False
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"```(?:json)?\\s*([\s\S]+?)\\s*```", content)
        if match:
            json_text = match.group(1)
        else:
            json_text = content

        try:
            plan = json.loads(json_text)
            return plan if isinstance(plan, list) else []
        except Exception as e:
            print(f"❌ 工具调用链规划失败: {e}\n原始返回: {content}")
            return []

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    server_script_path = "./server.py"
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

