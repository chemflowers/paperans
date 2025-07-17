import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
font = {'family' : 'MicroSoft YaHei',
        'weight' : 'bold',
        'size' : 8}
matplotlib.rc('font', **font)

# 数据爬取部分

child_resp = requests.get('https://www.hurun.net/zh-CN/Rank/HsRankDetailsList?num=ODBYW2BI&search=&offset=0&limit=1094')
#print(child_resp.text)

obj3 = re.compile(r'"hs_Character_Gender":"(?P<gender>.*?)".*?"hs_Character_Age":"(?P<age>.*?)".*?"hs_Character_Fullname_Cn":"(?P<name>.*?)".*?"hs_Character_BirthPlace_Cn":"(?P<birthplace>.*?)".*?"hs_Rank_Rich_Wealth":(?P<wealth>.*?)".*?"hs_Rank_Rich_ComName_Cn":"(?P<comname>.*?)".*?"hs_Rank_Rich_Industry_Cn":"(?P<industry>.*?)"',re.S)
result3 = obj3.finditer(child_resp.text)

data = []

for it in result3:
    name = it.group('name')
    age = it.group('age')
    birthplace = it.group('birthplace')
    gender = it.group('gender')
    wealth = it.group('wealth')
    comname = it.group('comname')
    industry = it.group('industry')
    s = [name,age,gender,wealth,birthplace,comname,industry]
    data.append(s)

df = pd.DataFrame(data,columns=['姓名','年龄','性别','财富值','出生地','企业','行业'])
print(df.to_string())

# 转换年龄为数值
df['年龄'] = pd.to_numeric(df['年龄'], errors='coerce')

# 处理性别字段
def clean_gender(gender):
    return '先生' if '先生' in gender or 'male' in gender.lower() else '女士'

df['性别'] = df['性别'].apply(clean_gender)

# 处理出生地字段
df['出生地'] = df['出生地'].replace('', '未知').replace('None', '未知')

# 年龄分组
def age_group(age):
    if pd.isna(age):
        return '未知'
    elif age < 40:
        return '40岁以下'
    elif age < 50:
        return '40-49岁'
    elif age < 60:
        return '50-59岁'
    elif age < 70:
        return '60-69岁'
    else:
        return '70岁以上'

df['年龄段'] = df['年龄'].apply(age_group)

# 统计出生地
birthplace_distribution = df['出生地'].value_counts().reset_index()
birthplace_distribution.columns = ['出生地', '人数']
birthplace_distribution = birthplace_distribution.head(10)

# 计算前10名总和
top_10_total = birthplace_distribution['人数'].sum()
other_count = len(df) - top_10_total
birthplace_distribution = pd.concat([
    birthplace_distribution,
    pd.DataFrame({'出生地': ['其他'], '人数': [other_count]})
])

# 统计性别分布
gender_distribution = df['性别'].value_counts().reset_index()
gender_distribution.columns = ['性别', '人数']

# 统计年龄分布
age_distribution = df['年龄段'].value_counts().reset_index()
age_distribution.columns = ['年龄段', '人数']
age_distribution = age_distribution.sort_values('年龄段')

# 处理行业拆分
df_industry = df.copy()
df_industry['行业'] = df_industry['行业'].str.split('、')
df_industry = df_industry.explode('行业')
df_industry['行业'] = df_industry['行业'].str.strip()

# 处理财富值
def parse_wealth(value):
    value = re.sub(r'[^\d.]', '', str(value))
    return float(value)
df_industry['财富值'] = df_industry['财富值'].apply(parse_wealth)

# 分析各行业数据
industry_wealth = df_industry.groupby('行业')['财富值'].sum().reset_index()
industry_wealth.columns = ['行业', '总财富值']
industry_wealth = industry_wealth.sort_values('总财富值', ascending=False)

industry_counts = df_industry.groupby('行业')['姓名'].nunique().reset_index()
industry_counts.columns = ['行业', '富豪数量']
industry_counts = industry_counts.sort_values('富豪数量', ascending=False)

industry_avg_wealth = df_industry.groupby('行业')['财富值'].mean().reset_index()
industry_avg_wealth.columns = ['行业', '平均财富值']
industry_avg_wealth = industry_avg_wealth.sort_values('平均财富值', ascending=False)
'''
# 输出分析结果
print("\n各行业总财富值排名：")
print(industry_wealth)

print("\n各行业富豪数量分布：")
print(industry_counts)

print("\n各行业平均财富值排名：")
print(industry_avg_wealth)

print("\n富豪年龄分布：")
print(age_distribution)

print("\n富豪性别分布：")
print(gender_distribution)
print("性别统计总人数:", gender_distribution['人数'].sum())

print("\n富豪出生地分布（前10+其他）：")
print(birthplace_distribution)
print("出生地统计总人数:", birthplace_distribution['人数'].sum())
'''
# 可视化分析结果
plt.figure(figsize=(22, 18))

# 行业富豪数量柱状图
plt.subplot(3, 1, 1)
top_10_counts = industry_counts.head(10)
bars1 = plt.bar(top_10_counts['行业'], top_10_counts['富豪数量'], color='skyblue')
plt.title('各行业富豪数量排名（前10）')
plt.xlabel('行业')
plt.ylabel('富豪数量')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 行业总财富值柱状图
plt.subplot(3, 1, 2)
top_10_wealth = industry_wealth.head(10)
bars2 = plt.bar(top_10_wealth['行业'], top_10_wealth['总财富值'], color='salmon')
plt.title('各行业总财富值排名（前10）', fontsize=14)
plt.xlabel('行业', fontsize=12)
plt.ylabel('总财富值（亿美元）', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 50,
             f'{height:.1f}', ha='center', va='bottom', fontsize=10)

# 行业平均财富值柱状图
plt.subplot(3, 1, 3)
top_10_avg_wealth = industry_avg_wealth.head(10)
bars3 = plt.bar(top_10_avg_wealth['行业'], top_10_avg_wealth['平均财富值'], color='lightgreen')
plt.title('各行业平均财富值排名（前10）', fontsize=14)
plt.xlabel('行业', fontsize=12)
plt.ylabel('平均财富值（亿美元）', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.1f}', ha='center', va='bottom', fontsize=10)

# 年龄分布柱状图
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
bars4 = plt.bar(age_distribution['年龄段'], age_distribution['人数'], color='plum')
plt.title('富豪年龄分布')
plt.xlabel('年龄段')
plt.ylabel('人数')
plt.xticks(rotation=0)
plt.tight_layout()

for bar in bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 性别分布饼图
plt.subplot(2, 2, 2)
plt.pie(gender_distribution['人数'], labels=gender_distribution['性别'], autopct='%1.1f%%',
        startangle=90, colors=['lightblue', 'pink', 'lightgreen'])
plt.title('富豪性别分布')
plt.axis('equal')
plt.tight_layout()

# 出生地分布饼图
plt.subplot(2, 1, 2)
plt.pie(birthplace_distribution['人数'], labels=birthplace_distribution['出生地'],
        autopct=lambda p: f'{p:.1f}%',
        startangle=90, shadow=True, pctdistance=0.8)
plt.title('富豪出生地分布（前10+其他）')
plt.axis('equal')
plt.tight_layout()

plt.show()

child_resp.close()

