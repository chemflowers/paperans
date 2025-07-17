# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 19:39:51 2025

@author: ww229
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import re

def get_weather_data(year_month):
    """
    获取指定年月的大连天气数据
    :param year_month: 格式为YYYYMM，如202201
    :return: 该月每天的天气数据列表
    """
    url = f"https://www.tianqihoubao.com/lishi/dalian/month/{year_month}.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
       # response.encoding = 'utf_8_sig'  
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        
        if not table:
            print(f"未找到表格数据: {year_month}")
            return []
        
        rows = table.find_all('tr')[1:]  # 跳过表头
        
        monthly_data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                # 检查1：是否有列 | 检查2：列是否全为空
                if not cols or all(col.get_text(strip=True) == '' for col in cols):
                    continue  # 跳过空行
                date = cols[0].get_text(strip=True)
                weather = cols[1].get_text(strip=True).replace('\n', '').replace('\r', '').replace(' ', '')
                temp = cols[2].get_text(strip=True)
                wind = cols[3].get_text(strip=True).replace('\n', '').replace('\r', '').replace(' ', '')
                
                # 分离白天和夜晚的天气
                day_night_weather = weather.split('/')
                day_weather = day_night_weather[0] if len(day_night_weather) > 0 else ''
                night_weather = day_night_weather[1] if len(day_night_weather) > 1 else ''
                
                # 分离最高温度和最低温度
                #temp_match = re.search(r'(\d+)℃/(\d+)℃', temp)
                temp_match = re.search(r'([-−]?\d+)℃/([-−]?\d+)℃', temp)
                max_temp = temp_match.group(1) if temp_match else ''
                min_temp = temp_match.group(2) if temp_match else ''
                
                # 分离白天和夜晚的风力
                day_night_wind = wind.split('/')
                day_wind = day_night_wind[0] if len(day_night_wind) > 0 else ''
                night_wind = day_night_wind[1] if len(day_night_wind) > 1 else ''
                
                # 提取风力等级
                day_wind_level = re.search(r'(\d+)-?(\d+)?级', day_wind)
                day_wind_min = day_wind_level.group(1) if day_wind_level else ''
                day_wind_max = day_wind_level.group(2) if day_wind_level and day_wind_level.group(2) else day_wind_min
                
                night_wind_level = re.search(r'(\d+)-?(\d+)?级', night_wind)
                night_wind_min = night_wind_level.group(1) if night_wind_level else ''
                night_wind_max = night_wind_level.group(2) if night_wind_level and night_wind_level.group(2) else night_wind_min
                
                monthly_data.append({
                    '日期': date,
                    '白天天气': day_weather,
                    '夜晚天气': night_weather,
                    '最高温度': max_temp,
                    '最低温度': min_temp,
                    '白天风力': day_wind,
                    '白天风力最小值': day_wind_min,
                    '白天风力最大值': day_wind_max,
                    '夜晚风力': night_wind,
                    '夜晚风力最小值': night_wind_min,
                    '夜晚风力最大值': night_wind_max
                })
        
        return monthly_data
    
    except Exception as e:
        print(f"获取{year_month}数据时出错: {e}")
        return []

def crawl_dalian_weather(start_year=2022, end_year=2024):
    """
    爬取指定年份范围的大连天气数据
    :param start_year: 开始年份
    :param end_year: 结束年份
    :return: 包含所有数据的DataFrame
    """
    all_data = []
    
    for year in range(start_year, end_year + 1):
        for month in tqdm(range(1, 13), desc=f"爬取{year}年数据"):
            year_month = f"{year}{month:02d}"
            monthly_data = get_weather_data(year_month)
            all_data.extend(monthly_data)
            time.sleep(1)  # 礼貌爬取，避免被封
    
    df = pd.DataFrame(all_data)
    
    return df

# 执行爬取
weather_df = crawl_dalian_weather(2022, 2024)
# 保存数据
weather_df.to_csv('dalian_weather_2022_2024-1.csv', index=False, encoding='utf_8_sig')

#df = pd.read_csv('dalian_weather_2022_2022-2.csv')
#df.dropna(how='all', inplace=True)  # 删除所有列均为空的行
#df.to_csv('cleaned_weather_1.csv', index=False, encoding='utf_8_sig')

# 加载数据
df = pd.read_csv('dalian_weather_2022_2024-1.csv')

# 数据清洗
def clean_data(df):
    # 转换日期格式
    # 假设日期在字符串开头，格式为 "2022年01月01日"
    df['日期'] = pd.to_datetime(
        df['日期'].str.extract(r'^(\d{4}年\d{2}月\d{2}日)')[0],
        format='%Y年%m月%d日',
        errors='coerce'  # 匹配失败时设为NaT而非报错
        )
    
    # 转换温度为数值
    df['最高温度'] = pd.to_numeric(df['最高温度'], errors='coerce')
    df['最低温度'] = pd.to_numeric(df['最低温度'], errors='coerce')
    
    # 转换风力为数值
    df['白天风力最小值'] = pd.to_numeric(df['白天风力最小值'], errors='coerce')
    df['白天风力最大值'] = pd.to_numeric(df['白天风力最大值'], errors='coerce')
    df['夜晚风力最小值'] = pd.to_numeric(df['夜晚风力最小值'], errors='coerce')
    df['夜晚风力最大值'] = pd.to_numeric(df['夜晚风力最大值'], errors='coerce')
    
    # 计算平均风力
    df['白天平均风力'] = (df['白天风力最小值'] + df['白天风力最大值']) / 2
    df['夜晚平均风力'] = (df['夜晚风力最小值'] + df['夜晚风力最大值']) / 2
    
    # 添加月份和年份信息
    df['月份'] = df['日期'].dt.month
    df['年份'] = df['日期'].dt.year
    
    # 简化天气分类
    weather_mapping = {
        '晴': '晴天',
        '多云': '多云',
        '阴': '阴天',
        '雨': '雨天','小雨': '雨天','中雨': '雨天','大雨': '雨天',
        '暴雨': '雨天','雷阵雨': '雨天','阵雨': '雨天',
        '小到中雨': '雨天','中到大雨': '雨天','大到暴雨': '雨天',
        '雪': '雪天','小雪': '雪天','中雪': '雪天','大雪': '雪天',
        '阵雪': '雪天','小到中雪':'雪天','中到大雪':'雪天','雨夹雪':'雪天',
        '雾': '雾天',
        '霾': '雾霾'
    }
    
    df['白天天气分类'] = df['白天天气'].map(lambda x: next((v for k, v in weather_mapping.items() if k in x), '其他'))
    df['夜晚天气分类'] = df['夜晚天气'].map(lambda x: next((v for k, v in weather_mapping.items() if k in x), '其他'))
    return df

cleaned_df = clean_data(df)
cleaned_df.to_csv('cleaned_dalian_weather_2022_2024-1.csv', index=False, encoding='utf_8_sig')


import matplotlib.pyplot as plt

# 设置中文字体（以微软雅黑为例）
#plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows系统常用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']        # 或使用 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示问题

# 计算每月平均最高温度和平均最低温度
monthly_avg =cleaned_df.groupby('月份').agg({
    '最高温度': 'mean',
    '最低温度': 'mean'
}).reset_index().round(1)
x=monthly_avg['月份']
y1=monthly_avg['最高温度']
y2=monthly_avg['最低温度']
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y1, label='平均最高温度', marker='o', color='r')

# 添加数值标签（可自定义偏移量）
for xi, yi in zip(x, y1):
    ax.annotate(
        f"{yi}", 
        xy=(xi, yi),          # 标签指向的坐标
        xytext=(0, 5),        # 文本位置偏移量（像素）
        textcoords='offset points',
        ha='center',          # 水平对齐
        va='bottom'           # 垂直对齐
    )

ax.plot(x, y2, label='平均最低温度', marker='o', color='b')
for xi, yi in zip(x, y2):
    ax.annotate(
        f"{yi}", 
        xy=(xi, yi),          # 标签指向的坐标
        xytext=(0, 5),        # 文本位置偏移量（像素）
        textcoords='offset points',
        ha='center',          # 水平对齐
        va='bottom'           # 垂直对齐
    )
plt.title('大连市2022-2024年月平均气温变化')
plt.xlabel('月份')
plt.ylabel('平均温度(℃)')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

  
# 风力等级分类
def classify_wind(wind):
    if wind   < 1: return '0-1级'
    elif wind < 2: return '1-2级'
    elif wind < 3: return '2-3级'
    elif wind < 4: return '3-4级'
    elif wind < 5: return '4-5级'
    elif wind < 6: return '5-6级'
    elif wind < 7: return '6-7级'
    elif wind < 8: return '7-8级'
    else: return '9级及以上'

cleaned_df['白天风力等级'] = cleaned_df['白天平均风力'].apply(classify_wind)
cleaned_df['夜晚风力等级'] = cleaned_df['夜晚平均风力'].apply(classify_wind)

# 统计每月不同风力等级天数
wind_day =(cleaned_df.groupby(['年份','月份', '白天风力等级'])
                     .size()
                     .unstack(fill_value=0)
                     .groupby('月份').mean().round(1).fillna(0))
fig, ax = plt.subplots(figsize=(12, 6))
wind_day.plot(kind='bar', stacked=True, ax=ax)
plt.title('大连市各月白天风力等级分布（2022-2024年月平均天数）')
plt.xlabel('月份')
plt.ylabel('平均天数')
plt.xticks(rotation=0)
plt.legend(title='风力等级',bbox_to_anchor=(1, 1))
# 添加数值标签
for rect in ax.patches:
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()
    if height>0:
        ax.text(x + width/2., y + height/2., 
                f"{height:.1f}", 
                ha='center', va='center',
                fontsize=8,
                ) 
plt.tight_layout()  # 自动调整布局
plt.show()

wind_night =(cleaned_df.groupby(['年份','月份', '夜晚风力等级'])
                     .size()
                     .unstack(fill_value=0)
                     .groupby('月份').mean().round(1).fillna(0))
fig, ax = plt.subplots(figsize=(12, 6))
wind_night.plot(kind='bar', stacked=True, ax=ax)
plt.title('大连市各月夜晚风力等级分布（2022-2024年月平均天数）')
plt.xlabel('月份')
plt.ylabel('平均天数')
plt.xticks(rotation=0)
plt.legend(title='风力等级',bbox_to_anchor=(1, 1))
# 添加数值标签
for rect in ax.patches:
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()
    if height>0 :
        ax.text(x + width/2., 
                y + height/2., 
                f"{height:.1f}", 
                ha='center', 
                va='center',
                fontsize=8,
                ) 
plt.tight_layout()  # 自动调整布局
plt.show()


'''
# 绘制饼图（以1月为例）
month = 1
current_data = wind_day.loc[month]

# 过滤0值数据
plot_data = current_data[current_data > 0]

plt.figure(figsize=(12, 10))

# 自定义百分比和天数显示
def autopct_format(pct):
    total = plot_data.sum()
    days = pct/100 * total
    return f'{days:.1f}天\n({pct:.1f}%)' 

wedges, texts, autotexts = plt.pie(
    plot_data,
    labels=plot_data.index,
    autopct=lambda p: f'{plot_data.sum()*p/100:.1f}天\n({p:.1f}%)',
    startangle=90,
    colors=plt.cm.Pastel1(np.linspace(0, 1, len(plot_data))),
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    pctdistance=0.85
)

# 添加图例
legend_labels = [f'{l}: {v:.1f}天' for l,v in plot_data.items()]
plt.legend(
    wedges,
    legend_labels,
    title='风力等级',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    fontsize=11
)

plt.title(f'大连市{month}月白天风力分布（2022-2024年平均天数)',fontsize=15, pad=20)
plt.tight_layout()
plt.show()

from matplotlib.patches import Patch

wind_dist = weather_df.groupby(['年份','月份','白天风力等级']).size().unstack(fill_value=0)
monthly_avg_wind = wind_dist.groupby('月份').mean().round(1)
all_wind_levels = [ '1-2级','2-3级','3-4级','4-5级','5-6级','6-7级','7-8级']

color_map = {
    '0-1级': '#a6cee3',
    '1-2级': '#66c2a5',
    '2-3级': '#fc8d62',
    '3-4级': '#8da0cb',
    '4-5级': '#e78ac3',
    '5-6级': '#a6d854',
    '6-7级': '#ffd92f',
    '7-8级': '#ff7f00',
}
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('大连市各月风力等级分布（2022-2024年平均）', fontsize=16, y=1.0)

for month in range(1, 13):
    ax = axes[(month-1)//4, (month-1)%4]
    month_data = monthly_avg_wind.loc[month]
    plot_data = month_data[month_data > 0]
    
    if len(plot_data) > 0:
        
        # 获取对应的颜色
        colors = [color_map[level] for level in plot_data.index]
        
        # 绘制饼图
        wedges, texts = ax.pie(
            plot_data,
            colors=colors,
            startangle=90,
            wedgeprops={'width':0.8, 'edgecolor':'white'},
            pctdistance=0.75
        )
        
        # 添加百分比标签
        for i, (level, days) in enumerate(plot_data.items()):
            pct = days/plot_data.sum()*100
            angle = (wedges[i].theta2 - wedges[i].theta1)/2 + wedges[i].theta1
            x = 0.8 * np.cos(np.deg2rad(angle))
            y = 0.8 * np.sin(np.deg2rad(angle))
            ax.text(x, y, f'{days:.1f}天\n({pct:.1f}%)', 
                    ha='center', va='center', fontsize=9)
    # 添加图例 - 只显示当前月份存在的风力等级
    legend_patches = [
        Patch(facecolor=color_map[level], 
              label=f'{level}: {plot_data.get(level, 0):.1f}天')
        for level in plot_data.index
    ]
    
    ax.legend(
        handles=legend_patches,
        title=f'{month}月',
        loc='upper left',
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(1, 1)
    )  
    ax.set_title(f'{month}月', fontsize=12, pad=10)
    ax.set_aspect('equal')

# 创建统一的图例 (放在图形底部)
legend_patches = [Patch(facecolor=color_map[level], label=f'{level}') 
                 for level in all_wind_levels]

# 调整图例位置和样式
fig.legend(
    handles=legend_patches,
    title='风力等级',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=len(all_wind_levels),
    fontsize=10,
    title_fontsize=12,
    frameon=True,
    framealpha=0.8
)
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.4, hspace=0.5)
plt.show()
'''

# 正确统计：先按年份+月份分组，再计算三年同月平均
day_monthly_avg = (
    cleaned_df.groupby(['年份', '月份', '白天天气分类'])
    .size()
    .unstack(fill_value=0)
    .groupby('月份')  # 对同月份求三年平均
    .mean()
    .round(1)        # 保留1位小数
    .fillna(0)
)

# 方法二：使用pivot_table
    #monthly_avg = (
    #    weather_df.pivot_table(
    #        index=['年份', '月份'],
    #       columns='白天天气分类',
    #        aggfunc='size',
    #        fill_value=0
    #    )
    #    .groupby('月份')
    #    .mean()
    #    .round(1)
    #)
fig, ax = plt.subplots(figsize=(12, 6))
day_monthly_avg.plot(kind='bar', stacked=True, ax=ax,colormap='tab20')

# 优化标签
plt.title('大连市各月白天天气状况分布（2022-2024年月平均天数）', pad=20)
plt.xlabel('月份', labelpad=10)
plt.ylabel('平均天数', labelpad=10)
plt.xticks(rotation=0)
plt.legend(title='天气类型', bbox_to_anchor=(1, 1))

# 添加数据标签
for rect in ax.patches:
    height = rect.get_height()
    # 确保height是标量值
    if pd.api.types.is_scalar(height) and height > 0:
        ax.text(
            rect.get_x() + rect.get_width()/2,
            rect.get_y() + rect.get_height()/2,
            f"{rect.get_height():.1f}",
            ha='center',
            va='center',
            fontsize=8,
            color='black'
        )

plt.tight_layout()
plt.show()


night_monthly_avg = (
    cleaned_df.pivot_table(
        index=['年份', '月份'],
        columns='白天天气分类',
        aggfunc='size',
        fill_value=0
    )
    .groupby('月份')
    .mean()
    .round(1)
)
fig, ax = plt.subplots(figsize=(12, 6))
night_monthly_avg.plot(kind='bar', stacked=True, ax=ax,colormap='tab20')

plt.title('大连市各月夜晚天气状况分布（2022-2024年月平均天数）', pad=20)
plt.xlabel('月份', labelpad=10)
plt.ylabel('平均天数', labelpad=10)
plt.xticks(rotation=0)
plt.legend(title='天气类型', bbox_to_anchor=(1, 1))

# 添加数据标签
for rect in ax.patches:
    height = rect.get_height()
    # 确保height是标量值
    if pd.api.types.is_scalar(height) and height > 0:
        ax.text(
            rect.get_x() + rect.get_width()/2,
            rect.get_y() + rect.get_height()/2,
            f"{rect.get_height():.1f}",
            ha='center',
            va='center',
            fontsize=8,
            color='black'
        )
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. 准备训练数据（2022-2024年数据）
def prepare_simple_features(df):
    # 按月聚合数据
    monthly_data = df.groupby(['年份', '月份'], as_index=False).agg({
        '最高温度': 'mean',
        '白天平均风力': 'mean',
        '夜晚平均风力': 'mean'
    })
    
    # 添加月份周期性特征
    monthly_data['月份_sin'] = np.sin(2 * np.pi * monthly_data['月份']/12)
    monthly_data['月份_cos'] = np.cos(2 * np.pi * monthly_data['月份']/12)
    
    # 添加季节特征
    monthly_data['季节'] = (monthly_data['月份'] - 1) // 3 + 1
    
    return monthly_data

train_data = prepare_simple_features(cleaned_df)

# 2. 定义特征和目标
features = ['月份', '白天平均风力', '夜晚平均风力', '季节', '月份_sin', '月份_cos']
target = '最高温度'

X = train_data[features]
y = train_data[target]

# 3. 创建简化模型
model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(n_estimators=100, random_state=42)
)

# 4. 训练模型
model.fit(X, y)

# 评估模型
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
print(f"模型在训练集上的MAE: {mae:.2f}℃")

# 5. 预测2025年数据
# 准备预测数据
df_2025 = pd.concat([pd.DataFrame(get_weather_data(f"2025{month:02d}")) for month in range(1, 7)])
df_2025 = clean_data(df_2025)
predict_data = df_2025.groupby('月份', as_index=False).agg({
    '白天平均风力': 'mean',
    '夜晚平均风力': 'mean'
})

# 添加特征
predict_data['月份_sin'] = np.sin(2 * np.pi * predict_data['月份']/12)
predict_data['月份_cos'] = np.cos(2 * np.pi * predict_data['月份']/12)
predict_data['季节'] = (predict_data['月份'] - 1) // 3 + 1

# 确保特征顺序一致
predict_data = predict_data[features]

# 预测
predict_data['预测最高温度'] = model.predict(predict_data)

# 获取实际值
actual_avg = df_2025.groupby('月份')['最高温度'].mean().reset_index()

# 合并结果
result = pd.merge(predict_data, actual_avg, on='月份', how='left')
result.rename(columns={'最高温度': '实际最高温度'}, inplace=True)

# 6. 可视化
plt.figure(figsize=(12, 6))
plt.plot(result['月份'], result['预测最高温度'], label='预测最高温度', marker='o', color='r')
plt.plot(result['月份'], result['实际最高温度'], label='实际最高温度', marker='o', color='b')
plt.title('大连市2025年1-6月预测与实际最高温度对比')
plt.xlabel('月份')
plt.ylabel('温度(℃)')
plt.xticks(range(1, 7))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 添加误差标注
for i, row in result.iterrows():
    error = abs(row['预测最高温度'] - row['实际最高温度'])
    plt.annotate(f'误差: {error:.1f}℃', 
                 (row['月份'], (row['预测最高温度'] + row['实际最高温度'])/2),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()






















