from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("cvpr_titles.csv")


year = 2024
df_year = df[df['year'] == year]


text = " ".join(df_year['title'].astype(str).tolist())


wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='Set2',
    max_words=100,
    max_font_size=90,
    min_font_size=10,
    font_path=None,
    collocations=True
).generate(text)


plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title(f"CVPR {year} Paper Title Word Cloud", fontsize=20)
plt.tight_layout(pad=2)


output_path = f"cvpr_wordcloud_{year}.png"
plt.savefig(output_path, dpi=300)

plt.show()

