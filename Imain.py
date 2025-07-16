import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud



def fetch_icml_papers(start_year=2020, end_year=2024):
    base_url = "https://dblp.org/db/conf/icml/"
    papers = []

    for year in range(start_year, end_year + 1):
        url = f"{base_url}icml{year}.html"
        print(f"Fetching ICML {year} from {url}")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        entries = soup.find_all('li', class_='entry inproceedings')

        for entry in entries:
            title_tag = entry.find('span', class_='title')
            author_tags = entry.find_all('span', itemprop='author')
            authors = [a.text.strip() for a in author_tags]
            link_tag = entry.find('nav')
            paper_url = link_tag.a['href'] if link_tag and link_tag.a else None

            papers.append({
                'title': title_tag.text.strip() if title_tag else None,
                'authors': ", ".join(authors),
                'year': year,
                'conference': "ICML",
                'url': paper_url
            })

        time.sleep(1)

    df = pd.DataFrame(papers)
    df.to_csv("icml_papers.csv", index=False, encoding='utf-8-sig')
    print(f"Saved icml_papers.csv with {len(df)} entries")
    return df


def predict_2025(df):
    yearly = df['year'].value_counts().sort_index()
    years = list(yearly.index)
    paper_counts = list(yearly.values)
    if len(paper_counts) < 3:
        print("Not enough data to perform prediction.")
        return None
    diff = [paper_counts[i] - paper_counts[i - 1] for i in range(1, len(paper_counts))]
    x = diff[:-1]
    y = diff[1:]
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum([xi ** 2 for xi in x])
    sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    delta_next = a * diff[-1] + b
    predicted = int(paper_counts[-1] + delta_next)
    print(f"Predicted ICML 2025 paper count: {predicted}")
    return predicted


def plot_trend(df, prediction=None, predict_year=None, conference_name="ICML"):
    counts = df['year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    hist_years = [y for y in counts.index if y != predict_year]
    hist_counts = [counts[y] for y in hist_years]
    plt.plot(hist_years, hist_counts, marker='o', color='darkgreen', linewidth=2, label='Actual')
    for x, y in zip(hist_years, hist_counts):
        plt.text(x, y + 10, str(y), ha='center', fontsize=10, color='black')

    # 添加预测点（单独绘制）
    if prediction and predict_year:
        plt.scatter(predict_year, prediction, color='red', s=80, label='Prediction')
        plt.text(predict_year, prediction + 10, f"{prediction} (pred)", ha='center', fontsize=10, color='red')

    # 图设置
    plt.title(f"{conference_name} Paper Count Trend (2020–{predict_year})", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{conference_name.lower()}_trend_predicted.png", dpi=300)
    plt.show()




def generate_wordclouds(df):
    out_dir = "icml_wordclouds"
    os.makedirs(out_dir, exist_ok=True)
    for year in range(2020, 2025):
        titles = df[df['year'] == year]['title'].dropna().astype(str).tolist()
        if not titles:
            continue
        text = " ".join(titles)
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='Set2',
            max_words=100,
            collocations=True
        ).generate(text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"ICML {year} Title Word Cloud", fontsize=20)
        plt.tight_layout()
        file_path = os.path.join(out_dir, f"icml_wordcloud_{year}.png")
        plt.savefig(file_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    df_icml = fetch_icml_papers()
    prediction = predict_2025(df_icml)
    plot_trend(df_icml, prediction=prediction, predict_year=2025, conference_name="ICML")
    generate_wordclouds(df_icml)
