# vLLM-Qwen3-8B
本教程试用vLLM部署Qwen3-8B模型，试用flask封装接口，并试用postman进行接口测试。
# 第一步：准备环境（在 AutoDL 上租 GPU）
打开AutoDL 官网，注册登录后，点击 “租用实例”。  
选一个显存足够的 GPU（8B 模型至少需要 10GB 以上显存，比如 RTX 3090/4090，选 “按量计费” 更灵活）。  
镜像选 “PyTorch 2.0+Python3.9”（自带基础环境，省去配置麻烦），点击 “立即创建”。  
创建后，点击 “JupyterLab” 进入操作界面（类似网页版的代码编辑器）。  

# 第二步：更新软件包
执行以下命令：
sudo apt update && sudo apt upgrade

# 第三步：Python安装
执行以下命令：
sudo apt upgrade python3
## pip安装：
sudo apt install python3-pip
## venv安装：
sudo apt install python3-venv
## pip镜像设置：
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 第四步：创建vLLM项目目录
cd /root/autodl-tmp
mkdir vllm_deploy
cd ./vllm_deploy

# 第五步：虚拟环境设置
## 创建虚拟环境
python3 -m venv .venv
## 激活虚拟环境
source .venv/bin/activate

# 第六步：安装vllm库
pip install "vllm>=0.8.5"

# 第七步：模型下载（使用modelscope）
pip install modelscope
将模型下载到工作目录：
modelscope download --model Qwen/Qwen3-8B  --local_dir ./Qwen3-8B

# 第八步：启动api服务
vllm serve ./Qwen3-8B --port 8000 --host 0.0.0.0 --max-model-len 32768
（根据可用内存，我租用的机器，最大支持的序列长度为 32768。）
也可以试用flask封装一个api接口（可选，也可以直接用上面自带的接口）：代码见app.py，运行指令：python app.py --model ./Qwen3-8B --port 8000

# 第九步：curl测试接口
打开一个新终端：
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{ 
    "model": "./Qwen3-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'

  之后返回：{"id": "chatcmpl-531412", "object": "chat.completion", "created": 1716230400, "model": "./Qwen3-8B", "choices": [{"message": {"role": "assistant", "content": "你好，我是Qwen，很高兴认识你！😊 有什么我可以帮助你的吗？"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 9, "completion_tokens": 31, "total_tokens": 40}}
  说明接口正常运行啦。
# 第十步：将auto-dl的端口让外部访问：
在auto-dl容器示例界面，点击“自定义服务”，然后下载工具。
复制容器实例的ssh指令和密码到工具，然后“代理到本地端口”填8000，“代理到远程端口”可以任意不冲突的。
之后就可以在本地电脑上访问测试接口了。
若想让别人也能测试接口，可以借用ngrok工具。
下载好之后，运行grok。输入指令：ngrok http 8080 ，就会返回一个连接。

# 第十一步：postman测试接口
以POST方式，输入https://localhost:8000/v1/chat/completions（或使用ngrok映射的地址+/v1/chat/completions）
点击Body，输入：
{
  "messages": [
    {"role": "user", "content": "介绍一下LSTM"}
  ],
  "max_tokens": 300,
  "temperature": 0.7
}

输出：{
    "id": "chatcmpl-520012",
    "object": "chat.completion",
    "created": 1716230400,
    "model": "unknown-model",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "好的，我现在要介绍LSTM。首先，我需要确定用户对LSTM的了解程度。可能他们已经对神经网络有一定的基础，但需要更深入的解释。LSTM是循环神经网络的一种，主要用于处理序列数据，比如时间序列或自然语言。我需要解释LSTM的结构，包括输入门、遗忘门和输出门，以及它们的作用。同时，要提到它的优势，比如解决长期依赖问题，以及应用场景，如语音识别、文本生成等。还要注意避免使用过于专业的术语，保持解释清晰易懂。可能用户还想知道LSTM与传统RNN的区别，需要对比一下。此外，可以举一些实际例子来帮助理解。最后，确保回答结构清晰，分点说明，让用户容易跟随。"
            },
            "finish_reason": "stop",
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 274,
        "total_tokens": 289
    }
}
说明接口正确运行了！
