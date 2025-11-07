import os
import json
from datetime import datetime
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

mcp = FastMCP("NewsServer")

@mcp.tool()
async def search_google(keyword: str) -> str:
    """
    使用 Serper API（Google Search 封装）根据关键词搜索内容，返回前5条标题、简单描述和链接。

    参数:
        keyword (str): 关键词，如 "小米汽车"

    返回:
        str: JSON 字符串，包含新闻标题、描述、链接
    """

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "❌ 未配置 SERPER_API_KEY，请在 .env 文件中设置"

    url = "https://google.serper.dev/news"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": keyword}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        data = response.json()

    if "news" not in data:
        return "❌ 未获取到搜索结果"

    articles = [
        {
            "title": item.get("title"),
            "desc": item.get("snippet"),
            "url": item.get("link")
        } for item in data["news"]
    ]

    return (
        f"✅ 已获取与 [{keyword}] 相关的 Google 新闻：\n"
        f"{json.dumps(articles, ensure_ascii=False, indent=2)}\n"
    )

@mcp.tool()
async def analyze_sentiment(text: str, filename: str) -> str:
    """
    对传入的一段文本内容进行情感分析，并保存为指定名称的 Markdown 文件。

    参数:
        text (str): 新闻描述或文本内容
        filename (str): 保存的 Markdown 文件名（不含路径）

    返回:
        str: 完整文件路径
    """

    openai_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("MODEL")
    client = OpenAI(api_key=openai_key, base_url=os.getenv("BASE_URL"))

    prompt = f"请对以下新闻内容进行情绪倾向分析，并说明原因：\n\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
        stream=False
    )
    result = response.choices[0].message.content.strip()

    markdown = f"""# 舆情分析报告

**分析时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📥 原始文本

{text}

---

## 📊 分析结果

{result}
"""

    output_dir = "./sentiment_reports"
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return file_path


async def debug_sentiment_analysis():
    print("--- 🚀 开始调试 analyze_sentiment ---")

    test_text = "小米汽车今天发布了，市场反应非常热烈，订单量远超预期。"
    test_filename = "./sentiment_report_test.md"

    try:
        file_path = await analyze_sentiment(text=test_text, filename=test_filename)
        
        print(f"\n--- ✅ 调试成功 ---")
        print(f"报告已生成在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            print("\n--- 报告内容预览 ---")
            print(f.read())

    except Exception as e:
        print(f"\n--- ❌ 调试失败 ---")
        print(f"错误: {e}")


async def debug_search():
    print("--- 🚀 开始调试 search_google_news ---")

    test_keyword = "小鹏机器人评价"

    try:
        result_message = await search_google(keyword=test_keyword)
        
        print(f"\n--- ✅ 调试成功 ---")
        print(result_message)

    except Exception as e:
        print(f"\n--- ❌ 调试失败 ---")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("===================================================")
    print(f"🚀 服务 [NewsServer] 正在启动...")
    print(f"🛠️  已注册工具:")
    
    for tool_name in mcp._tool_manager._tools.keys():
        print(f"    - {tool_name}")
        
    print("===================================================")
    print(f"✅ [NewsServer] MCP 服务已就绪 (transport='stdio')")
    mcp.run(transport='stdio')


    # import asyncio
    # asyncio.run(debug_search())