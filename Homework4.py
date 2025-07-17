import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def get_data():
    url = "https://i.cmzj.net/expert/queryExpertById"

    sum_data = []
    ex_id = [1961120, 1724520, 2370927, 1946713, 2371123, 1550802, 1775905, 2439744, 2299072, 2415956, 1968441,
             2388076, 2372346, 1209497, 2203940, 385912, 2370932, 2444138, 2267369, 2370219, 2377923, 1841146]

    for id in ex_id:
        params = {"expertId": f"{id}"}
        response = requests.get(url=url, params=params)
        dic = json.loads(response.text)
        diff = {"age": dic["data"]["age"], "articles": dic["data"]["articles"], "ssqOne": dic["data"]["ssqOne"],
                "ssqTwo": dic["data"]["ssqTwo"], "ssqThree": dic["data"]["ssqThree"]}
        sum_data.append(diff)

    df = pd.DataFrame(sum_data)
    df.to_csv("expert.csv", encoding="utf-8")

def data_analysis():
    # 创建一个包含两个子图的画布
    data=pd.read_csv("expert.csv")
    # 创建彩龄分布直方图画布
    plt.figure(figsize=(10, 6))

    # 计算年龄的均值、中位数和众数
    age_mean = np.mean(data['age'])
    age_median = np.median(data['age'])
    age_mode = pd.Series(data['age']).mode()[0]

    # 绘制年龄的直方图
    plt.hist(data['age'], bins=20, edgecolor='black', alpha=0.7)
    plt.title('专家彩龄分布直方图')
    plt.xlabel('彩龄')
    plt.ylabel('频数')
    plt.axvline(age_mean, color='red', linestyle='dashed', linewidth=1, label=f'均值: {age_mean:.2f}')
    plt.axvline(age_median, color='green', linestyle='dashed', linewidth=1, label=f'中位数: {age_median}')
    plt.axvline(age_mode, color='blue', linestyle='dashed', linewidth=1, label=f'众数: {age_mode}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("年龄分布直方图.png")
    plt.show()

    # 创建发文量分布直方图画布
    plt.figure(figsize=(10, 6))

    # 计算发文量的均值、中位数和众数
    articles_mean = np.mean(data['articles'])
    articles_median = np.median(data['articles'])
    articles_mode = pd.Series(data['articles']).mode()[0]

    # 绘制发文量的直方图
    plt.hist(data['articles'], bins=20, edgecolor='black', alpha=0.7)
    plt.title('专家发文量分布直方图')
    plt.xlabel('发文量')
    plt.ylabel('频数')
    plt.axvline(articles_mean, color='red', linestyle='dashed', linewidth=1, label=f'均值: {articles_mean:.2f}')
    plt.axvline(articles_median, color='green', linestyle='dashed', linewidth=1, label=f'中位数: {articles_median}')
    plt.axvline(articles_mode, color='blue', linestyle='dashed', linewidth=1, label=f'众数: {articles_mode}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("发文量分布直方图.png")
    plt.show()

    # 计算相关系数矩阵
    correlation_matrix = data[['age', 'articles', 'ssqOne', 'ssqTwo', 'ssqThree']].corr()

    # 创建热力图画布
    plt.figure(figsize=(8, 6))

    # 绘制热力图
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('三种奖项与彩龄和发文量的热力图')
    plt.tight_layout()
    plt.savefig("三种奖项与彩龄和发文量相关的热力图.png")
    plt.show()



if __name__ == "__main__":
    get_data()
    data_analysis()











