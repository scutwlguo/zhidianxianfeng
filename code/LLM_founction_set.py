import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import Optional
from typing import Annotated, Literal

# 加载环境变量
load_dotenv()


def create_llm(
        platform: Annotated[
            Literal["openai", "deepseek", "aliyun", "dmx"],
            "支持的平台"
        ],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
) -> Optional[ChatOpenAI]:
    """
    创建大语言模型实例

    参数:
        platform: 平台名称，支持 'openai', 'deepseek', 'aliyun', 'dmx'
        model_name: 模型名称
        temperature: 温度参数，控制随机性
        max_tokens: 最大生成token数

    返回:
        ChatOpenAI 实例或 None（如果平台不支持）
    """
    platform = platform.lower()

    if platform == 'openai':
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=temperature,
            max_tokens=max_tokens
        )

    elif platform == 'deepseek':
        return ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model=model_name,
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=temperature,
            max_tokens=max_tokens
        )

    elif platform == 'aliyun':
        return ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=model_name,
            openai_api_base=os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=temperature,
            max_tokens=max_tokens
        )

    elif platform == 'dmx':
        return ChatOpenAI(
            api_key=os.getenv("DMX_API_KEY"),
            model=model_name,  # 可用模型可以参考这个网址 https://www.dmxapi.cn/pricing
            openai_api_base=os.getenv("DMXAPI_URL", "https://www.dmxapi.cn/v1"),
            temperature=temperature,
            max_tokens=max_tokens
        )

    else:
        print(f"不支持的平台: {platform}")
        return None


def get_embedding_model(source: str = "api", model_name: str = "text-embedding-3-small", **kwargs):
    """
    获取嵌入模型

    参数:
        source: "local" 使用本地模型, "api" 使用远程API
        model_name: 模型名称
        kwargs: 其他传递给模型的参数

    返回:
        嵌入模型实例
    """
    if source == "local":
        # 假设有本地模型加载方式
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="E:\\研究生文件\\科研\\大模型\\models\\BAAI\\bge-m3",
            model_kwargs={
                "device": "cuda"  # 如果有GPU可以使用"cuda"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )
    else:
        # 默认用API方式
        embedding_model = OpenAIEmbeddings(
            model=model_name,  # "text-embedding-3-small" 或 "text-embedding-3-large"
            **kwargs
        )
    return embedding_model


def show_workflow_graph(
        rag_chain: Annotated[
            object,
            "编译好的工作流对象，需包含get_graph方法"
        ],
        filename: Annotated[
            str,
            "保存生成图像的文件名，默认为'graph.jpg'"
        ] = "graph.jpg"
) -> None:
    """
    展示工作流的Mermaid图。
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    try:
        mermaid_code = rag_chain.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(mermaid_code)

        img = mpimg.imread(filename)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

    # from IPython.display import Image, display
    #
    # try:
    #     display(Image(rag_chain.get_graph().draw_mermaid_png()))
    # except Exception:
    #     # This requires some extra dependencies and is optional
    #     pass


# 使用示例
if __name__ == "__main__":
    # 创建DMX平台的GLM-4.5-Flash模型
    llm = create_llm("dmx", "GLM-4.5-Flash")
    if llm:
        response = llm.invoke("你好，请问你是什么模型？")
        print(response.content)

    # 创建DeepSeek平台的模型
    deepseek_llm = create_llm("deepseek", "deepseek-chat")
    if deepseek_llm:
        response = deepseek_llm.invoke("请介绍一下你自己")
        print(response.content)

    # 创建OpenAI平台的模型
    openai_llm = create_llm("openai", "gpt-3.5-turbo")
    if openai_llm:
        response = openai_llm.invoke("请用一句话介绍你自己")
        print(response.content)

    # 创建Aliyun平台的模型
    aliyun_llm = create_llm("aliyun", "qwen-turbo")
    if aliyun_llm:
        response = aliyun_llm.invoke("你能做什么？")
        print(response.content)
