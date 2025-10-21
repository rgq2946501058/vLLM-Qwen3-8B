from vllm import LLM, SamplingParams
from flask import Flask, request, make_response
import argparse
import json

# 初始化Flask应用
app = Flask(__name__)
llm = None  # 全局变量存储模型，避免重复加载

def load_model(model_path):
    """加载模型（只加载一次，减少显存占用）"""
    global llm
    print(f"正在加载模型: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=8192,  # 适配显存的最大序列长度
        gpu_memory_utilization=0.9,  # 控制显存使用率，避免溢出
        tensor_parallel_size=1  # 单GPU部署，无需张量并行
    )
    print("模型加载完成！")

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """模拟OpenAI格式的聊天接口，确保中文正常显示"""
    global llm
    if llm is None:
        # 模型未加载时返回错误
        error_data = {"error": "模型未成功加载，请检查服务启动日志"}
        response = make_response(json.dumps(error_data, ensure_ascii=False))
        response.headers["Content-Type"] = "application/json"
        return response, 500
    
    # 1. 获取请求参数（兼容OpenAI接口格式）
    data = request.json
    model_name = data.get("model", "unknown-model")  # 模型名称（仅用于返回标识）
    messages = data.get("messages", [])  # 对话历史
    temperature = data.get("temperature", 0.7)  # 随机性（0-1，值越高越灵活）
    max_tokens = data.get("max_tokens", 512)  # 最大生成token数

    # 2. 拼接对话历史（适配千问模型的输入格式）
    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role == "user":
            prompt += f"用户：{content}\n"
        elif role == "assistant":
            prompt += f"助手：{content}\n"
    # 追加助手的回答前缀，引导模型生成回复
    prompt += "助手："

    # 3. 设置采样参数（控制生成效果）
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["用户："],  # 遇到“用户：”停止生成，避免续写下一轮用户输入
        skip_special_tokens=True  # 跳过模型中的特殊token（如<s>、</s>）
    )

    # 4. 调用模型生成回答
    outputs = llm.generate([prompt], sampling_params)
    # 提取生成结果（处理单条请求的情况）
    generated_text = outputs[0].outputs[0].text.strip()

    # 5. 构造返回结果（严格遵循OpenAI接口格式，中文不转义）
    response_data = {
        "id": f"chatcmpl-{hash(prompt) % 1000000}",  # 简单生成唯一ID
        "object": "chat.completion",
        "created": 1716230400,  # 固定时间戳（实际场景可替换为time.time()）
        "model": model_name,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": generated_text  # 模型生成的中文回答
                },
                "finish_reason": "stop",  # 停止原因（stop=正常停止，length=达到max_tokens）
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),  # 输入token数（估算值）
            "completion_tokens": len(generated_text),  # 生成token数（估算值）
            "total_tokens": len(prompt) + len(generated_text)  # 总token数
        }
    }

    # 6. 返回响应（关闭ASCII编码，确保中文正常显示）
    response = make_response(json.dumps(response_data, ensure_ascii=False))
    response.headers["Content-Type"] = "application/json"
    return response

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Qwen3-8B模型VLLM部署（Flask接口）")
    parser.add_argument("--model", type=str, default="./Qwen3-8B", help="模型本地路径（如./Qwen3-8B）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址（0.0.0.0允许外部访问）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（如8000）")
    args = parser.parse_args()
    
    # 加载模型并启动服务
    load_model(args.model)
    print(f"\n服务启动成功！可通过以下地址测试：")
    print(f"本地测试：http://localhost:{args.port}/v1/chat/completions")
    print(f"外部测试：http://你的AutoDL公网地址:{args.port}/v1/chat/completions")
    
    # 启动Flask服务（debug=False避免重复加载模型，threaded=True支持多线程）
    app.run(host=args.host, port=args.port, debug=False, threaded=True)