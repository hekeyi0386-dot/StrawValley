"""
快速测试脚本 —— 直接运行，验证 API 是否正常工作
用法: python test_api.py
"""

import requests

BASE = "http://localhost:8000"


def test_root():
    r = requests.get(f"{BASE}/")
    print("健康检查:", r.json())


def test_models():
    r = requests.get(f"{BASE}/models")
    print("模型列表:", r.json())


def test_chat():
    payload = {
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手，用中文回答。"},
            {"role": "user",   "content": "用一句话介绍你自己。"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }
    r = requests.post(f"{BASE}/chat", json=payload)
    if r.ok:
        data = r.json()
        print(f"模型: {data['model']}")
        print(f"回复: {data['content']}")
        print(f"用量: {data['usage']}")
    else:
        print("错误:", r.status_code, r.text)


def test_simple():
    r = requests.post(
        f"{BASE}/chat/simple",
        params={"user_input": "1+1等于多少？", "system_prompt": "你是数学老师"}
    )
    if r.ok:
        print("简化接口回复:", r.json()["content"])
    else:
        print("错误:", r.status_code, r.text)


if __name__ == "__main__":
    test_root()
    test_models()
    test_chat()
    test_simple()
