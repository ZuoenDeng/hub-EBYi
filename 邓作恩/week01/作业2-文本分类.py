import pandas as pd
import jieba # 中文分词
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN 分类模型
from openai import OpenAI
from typing import Union
from fastapi import FastAPI
app = FastAPI()

# 加载数据集
dataset = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=None)
# print(dataset.head(10))
# print('数据集的样本维度：', dataset.shape)
# print('数据集的样本数量：', dataset[1].count())
# print('数据集的类别数量：', dataset[1].nunique())
# print('数据集的类别名称：', dataset[1].unique())
# print('数据集的类别频次分布：', dataset[1].value_counts())

text = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 该语句执行时，pd.read_csv中不能设置name，否则会报错

vector = CountVectorizer()
vector.fit(text.values)
text = vector.transform(text.values)  # fit用于构建词汇表，transform基于已学到的词汇表将文本转换为向量

model = KNeighborsClassifier()
model.fit(text, dataset[1].values)
# print(model)

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-08f00947070c455dae967448c2e1892b", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@app.get("/text_cls/ml")
def text_classification_by_ml(text: str) -> str:
    """
    使用机器学习模型对文本进行分类
    :param text: 待分类文本
    :return: 分类结果
    """
    test_query = " ".join(jieba.lcut(text))
    test_query = vector.transform([test_query])
    return model.predict(test_query)[0]

@app.get("/text_cls/llm")
def text_classification_by_llm(text: str) -> str:
    """
    使用语言模型对文本进行分类
    :param text: 待分类文本
    :return: 分类结果
    """
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-flash", # 模型的代号

        # 对话列表
        messages=[
            {"role": "user", "content": f"""
            {text}
            帮我对以上的内容进行分类，从以下的类别中进行选择，并且除了类别信息其他内容全部删除，只返回类别名称：{dataset[1].unique()}
            """},  # 用户的提问
        ]


    )
    return completion.choices[0].message.content
