import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_dblp_papers(conference: str, start_year: int, end_year: int):
    base_url = f"https://dblp.org/db/conf/{conference}/"
    all_papers = []

    for year in range(start_year, end_year + 1):
        url = f"{base_url}{conference}{year}.html"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        entries = soup.find_all('li', class_='entry inproceedings')

        for entry in entries:
            title_tag = entry.find('span', class_='title')
            author_tags = entry.find_all('span', itemprop='author')
            authors = [a.text.strip() for a in author_tags]
            paper_url = None
            if entry.find('nav'):
                link = entry.find('nav').find('a')
                paper_url = link['href'] if link else None

            paper = {
                "title": title_tag.text.strip() if title_tag else None,
                "authors": ", ".join(authors),
                "year": year,
                "conference": conference.upper(),
                "url": paper_url
            }
            all_papers.append(paper)

        time.sleep(1)

    return all_papers

papers = fetch_dblp_papers("cvpr", 2020, 2024)

df = pd.DataFrame(papers)
df.to_csv("cvpr_papers_full.csv", index=False, encoding='utf-8-sig')

