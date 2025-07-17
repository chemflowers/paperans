import re
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet  # 用于时间序列预测
from collections import Counter

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def get_data():
    merge_data = []
    pageNum = [1, 2, 3, 4]
    tt = [0.4082842005497679, 0.7812417132691014, 0.8036233691454469, 0.5071601937571554]
    last = [3, 4, 5, 6]
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.zhcw.com/",
        "Sec-Fetch-Dest": "script",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\""
    }

    url = "https://jc.zhcw.com/port/client_json.php"

    for pN,tN,lN in zip(pageNum,tt,last):
        params = {
            "callback": "jQuery112208823471430548924_1751607466638",
            "transactionType": "10001001",
            "lotteryId": "281",
            "issueCount": "0",
            "startIssue": "24126",
            "endIssue": "25073",
            "startDate": "",
            "endDate": "",
            "type": "1",
            "pageNum": f"{pN}",
            "pageSize": "30",
            "tt": f"{tN}",
            "_": f"175160746664{lN}"
        }
        response = requests.get(url, headers=headers, params=params)
        data=response.text
        obj=re.compile(r"jQuery112208823471430548924_1751607466638\((?P<json>.*?)\)")
        content=obj.search(data).group("json")
        dic=json.loads(content)
        diff=dic["data"]
        merge_data+=diff
    df = pd.DataFrame(merge_data)
    df.to_csv("大乐透.csv")

def first_question():
    data = pd.read_csv("大乐透.csv")
    data["openTime"] = pd.to_datetime(data["openTime"])
    df = data[["openTime", "saleMoney"]]
    df=df.sort_values("openTime").reset_index(drop=True)
    df["weekday"] = df["openTime"].dt.dayofweek

    # 按日期排序
    data = data.sort_values("openTime").reset_index(drop=True)
    # 添加“周几”字段（开奖固定为周一、周三、周六，验证周期规律）
    data["weekday"] = data["openTime"].dt.dayofweek  # 0=周一，2=周三，5=周六

    plt.plot(data["openTime"], data["saleMoney"] / 1e6, "-g", label="实际销售额（万元）")  # 转换为万元便于显示

    # 格式优化
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # 按月显示刻度
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.ylabel("销售额（万元）")
    plt.title("大乐透销售额随开奖日期变化趋势（2024.10-2025.06）")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("销售额趋势图.png", bbox_inches="tight")
    plt.show()

    #为建立Prophet模型做数据准备
    prophet_data = df.rename(columns={"openTime": "ds", "saleMoney": "y"})

    #建立模型
    m=Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False,
              changepoint_prior_scale=0.53)
    m.add_regressor('weekday')
    m.fit(prophet_data)

    future = m.make_future_dataframe(periods=5)
    future['weekday'] = future['ds'].dt.dayofweek

    forecast = m.predict(future)

    next_prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-4]
    weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    weekday_name = weekday_map[next_prediction['ds'].dayofweek]

    print(f"\n下一次预测日期：{next_prediction['ds'].strftime('%Y-%m-%d')} ({weekday_name})")
    print(f"预测销售额：{next_prediction['yhat'] / 1e6:.2f} 万元")
    print(f"预测区间：{next_prediction['yhat_lower'] / 1e6:.2f} 万元 ~ {next_prediction['yhat_upper'] / 1e6:.2f} 万元")

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1=m.plot(forecast)
    plt.title("大乐透销售额预测结果")
    plt.xlabel("日期")
    plt.ylabel("销售额（元）")
    plt.tight_layout()
    fig1.savefig("销售额预测结果图.png", bbox_inches="tight")

    fig2=m.plot_components(forecast)
    plt.tight_layout()
    fig2.savefig("预测组件分析图.png", bbox_inches="tight")

    plt.show()

def second_question():
    data = pd.read_csv("大乐透.csv")
    front_all_numbers = []
    for row in data['frontWinningNum']:
        # 假设号码是空格分隔的字符串，如"01 02 03"，拆分后存入列表
        front_all_numbers.extend(row.split())
    front_counts = Counter(front_all_numbers)
    total_rows = len(data)
    # 按号码升序排序，确保x轴顺序正确
    sorted_front_numbers = sorted(front_counts.keys(), key=lambda x: int(x))
    front_frequencies = [front_counts[num]/(total_rows*5) for num in sorted_front_numbers]

    # 绘制前区频率直方图
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_front_numbers, front_frequencies, color='blue')
    plt.xticks(rotation=0)  # x轴标签不旋转，与示例格式一致
    plt.xlabel("号码")
    plt.ylabel("频率")
    plt.title("前区号码频率分布直方图")
    plt.grid(alpha=0.3)  # 网格透明度0.3，增强可读性
    plt.tight_layout()  # 自动调整布局，避免元素重叠
    plt.savefig("前区号码频率分布直方图.png", bbox_inches="tight")
    plt.show()

    # ------------------------ 后区号码频率统计与绘图 ------------------------
    back_all_numbers = []
    for row in data['backWinningNum']:
        back_all_numbers.extend(row.split())  # 同样拆分后区号码
    back_counts = Counter(back_all_numbers)
    # 按号码升序排序
    sorted_back_numbers = sorted(back_counts.keys(), key=lambda x: int(x))
    back_frequencies = [back_counts[num] / (total_rows*2) for num in sorted_back_numbers]

    # 绘制后区频率直方图
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_back_numbers, back_frequencies, color='blue')
    plt.xticks(rotation=0)
    plt.xlabel("号码")
    plt.ylabel("频率")
    plt.title("后区号码频率分布直方图")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("后区号码频率分布直方图.png", bbox_inches="tight")

    # 前区每个位置的号码分布（前区共5个位置）
    front_position_counts = [Counter() for _ in range(5)]
    for row in data['frontWinningNum']:
        numbers = row.split()
        for i in range(5):
            if i < len(numbers):
                front_position_counts[i][numbers[i]] += 1

    # 后区每个位置的号码分布（后区共2个位置）
    back_position_counts = [Counter() for _ in range(2)]
    for row in data['backWinningNum']:
        numbers = row.split()
        for i in range(2):
            if i < len(numbers):
                back_position_counts[i][numbers[i]] += 1

    # 绘制前区各位置号码分布
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    fig.suptitle("前区各位置号码出现次数分布", fontsize=16)

    for i in range(5):
        position = i + 1
        counts = front_position_counts[i]
        # 确保所有可能的号码都显示，即使次数为0
        all_possible_numbers = [f"{num:02d}" for num in range(1, 36)]  # 前区号码范围1-35
        number_list = []
        count_list = []
        for num in all_possible_numbers:
            number_list.append(num)
            count_list.append(counts.get(num, 0))

        axes[i].bar(number_list, count_list, color='skyblue')
        axes[i].set_title(f"第{position}位")
        axes[i].set_xlabel("号码")
        axes[i].set_ylabel("出现次数")
        axes[i].tick_params(axis='x', rotation=90, labelsize=8)
        axes[i].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，避免标题重叠
    plt.savefig("前区各位置号码出现次数.png", bbox_inches="tight")
    plt.show()

    # 绘制后区各位置号码分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("后区各位置号码出现次数分布", fontsize=16)

    for i in range(2):
        position = i + 1
        counts = back_position_counts[i]
        # 确保所有可能的号码都显示，即使次数为0
        all_possible_numbers = [f"{num:02d}" for num in range(1, 13)]  # 后区号码范围1-12
        number_list = []
        count_list = []
        for num in all_possible_numbers:
            number_list.append(num)
            count_list.append(counts.get(num, 0))

        axes[i].bar(number_list, count_list, color='lightgreen')
        axes[i].set_title(f"第{position}位")
        axes[i].set_xlabel("号码")
        axes[i].set_ylabel("出现次数")
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("后区各位置号码出现次数.png", bbox_inches="tight")
    plt.show()

    front_most_common_numbers = [counts.most_common(1)[0][0] if counts else "" for counts in front_position_counts]
    back_most_common_numbers = [counts.most_common(1)[0][0] if counts else "" for counts in back_position_counts]
    final_numbers = front_most_common_numbers + back_most_common_numbers
    print("最终预测的号码为：", final_numbers)



def third_question():
    df=pd.read_csv("大乐透.csv")
    weeks = df['week'].unique()

    # 提取每个开奖日对应的销售额数据
    sales_data = [df[df['week'] == week]['saleMoney'].values for week in weeks]

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

    # 绘制销售额随不同开奖日的分布图
    plt.figure(figsize=(10, 6))
    plt.boxplot(sales_data, tick_labels=weeks)
    plt.title('销售额随不同开奖日的分布图')
    plt.xlabel('开奖日')
    plt.xticks(rotation=45)
    plt.ylabel('销售额')
    plt.savefig("销售额随不同开奖日的分布图.png")

    # 计算不同开奖日销售额的平均值
    average_sales = df.groupby('week')['saleMoney'].mean()

    # 绘制不同开奖日销售额平均值的图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(average_sales.index, average_sales.values)
    plt.title('不同开奖日销售额的平均值')
    plt.xlabel('开奖日')
    plt.xticks(rotation=45)
    plt.ylabel('平均销售额')
    plt.savefig("平均销售额随不同开奖日的分布图.png")

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    # 提取前区和后区中奖号码，并拆分为单个号码
    df['frontNumbers'] = df['frontWinningNum'].str.split(' ')
    df['backNumbers'] = df['backWinningNum'].str.split(' ')

    # 按开奖日分组
    monday_data = df[df['week'] == '星期一']
    wednesday_data = df[df['week'] == '星期三']
    saturday_data = df[df['week'] == '星期六']
    # 统计周一前区和后区号码出现次数

    def count_numbers(data, column, num_range):
        all_numbers = []
        for numbers in data[column]:
            all_numbers.extend(numbers)
        count_dict = {str(i).zfill(2): 0 for i in num_range}
        for num in all_numbers:
            if num in count_dict:
                count_dict[num] += 1
        return pd.Series(count_dict)

    monday_front_count = count_numbers(monday_data, 'frontNumbers', range(1, 36))
    monday_back_count = count_numbers(monday_data, 'backNumbers', range(1, 13))

    # 统计周三前区和后区号码出现次数
    wednesday_front_count = count_numbers(wednesday_data, 'frontNumbers', range(1, 36))
    wednesday_back_count = count_numbers(wednesday_data, 'backNumbers', range(1, 13))

    # 统计周六前区和后区号码出现次数
    saturday_front_count = count_numbers(saturday_data, 'frontNumbers', range(1, 36))
    saturday_back_count = count_numbers(saturday_data, 'backNumbers', range(1, 13))

    # 创建画布，包含3个子图
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    def plot_group(ax, front_ser, back_ser, title):
        # 前区坐标：1~35（方便后区接着排）
        front_x = range(1, len(front_ser) + 1)
        # 后区坐标：紧接前区，从 len(front_x)+1 开始
        back_x = range(len(front_x) + 1, len(front_x) + len(back_ser) + 1)

        # 绘制柱状图
        ax.bar(front_x, front_ser, label='前区号码', color='tab:blue')
        ax.bar(back_x, back_ser, label='后区号码', color='tab:orange')

        # 构造 X 轴标签：前区用原始号码（01~35），后区用 01~12
        front_labels = front_ser.index.tolist()
        back_labels = [f"{i:02d}" for i in range(1, len(back_ser) + 1)]
        all_labels = front_labels + back_labels

        # 关键：坐标是连续的（front_x + back_x），但标签手动替换后区为 01~12
        ax.set_xticks(list(front_x) + list(back_x))
        ax.set_xticklabels(all_labels, rotation=90, fontsize=8)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('号码', fontsize=10)
        ax.set_ylabel('出现次数', fontsize=10)
        ax.legend()

    plot_group(axes[0], monday_front_count, monday_back_count, '周一号码出现次数')
    plot_group(axes[1], wednesday_front_count, wednesday_back_count, '周三号码出现次数')
    plot_group(axes[2], saturday_front_count, saturday_back_count, '周六号码出现次数')
    plt.savefig("不同组别的号码分布直方图.png", bbox_inches="tight")

    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    get_data()
    first_question()
    second_question()
    third_question()






