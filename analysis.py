import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


sns.set(style="whitegrid")


df = pd.read_csv("cvpr_titles.csv")


year_counts = df['year'].value_counts().sort_index()


plt.figure(figsize=(10, 6))
plt.plot(year_counts.index, year_counts.values, marker='o', color='#2c7fb8', linewidth=2)


for x, y in zip(year_counts.index, year_counts.values):
    plt.text(x, y + 5, str(y), ha='center', fontsize=10, color='black')


plt.title("Number of Papers Published at CVPR (2020–2023)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Papers", fontsize=12)


plt.xticks(year_counts.index, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.show()

