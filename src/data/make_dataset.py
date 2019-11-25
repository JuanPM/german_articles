from bs4 import BeautifulSoup
import requests
import os
import re
import pandas as pd


def fetch_url_to_txt(url):
    """Scrapes data from provided url. Writes contents to txt file in data/raw folder"""
    page = requests.get(url)

    soup = BeautifulSoup(page.text, 'html.parser')

    rel_path = '/Users/juanpablomejia/Desktop/german_articles/data/raw'

    completeName = os.path.join(rel_path, "raw.txt")

    with open(completeName, 'w', encoding='utf-8') as f_out:
        f_out.write(soup.prettify())


def txt_to_feature_df():
    file = open('/Users/juanpablomejia/Desktop/german_articles/data/raw/raw.txt')
    soup = BeautifulSoup(file, 'html.parser')
    hits = soup.findAll('span')

    index = []
    eng_words = []
    articles = []
    ger_words = []

    for hit in hits:
        index_match = re.search(r'\d+', hit.text)
        eng_match = re.search(r'\s(\w+)\s', hit.text)
        art_match = re.search(r'D[eia][res]\s', hit.text)
        ger_match = re.search(r'D[eia][res]\s(\w+)', hit.text)

        if (not index_match) or (not art_match):
            continue
        else:
            index.append(index_match.group())
            eng_words.append(eng_match.group())
            articles.append(art_match.group().strip())
            ger_words.append(ger_match.group(1))

    dic = {'index': index, 'eng_words': eng_words,
           'ger_words': ger_words, 'articles': articles}

    df = pd.DataFrame(dic)
    df = df.set_index('index')
    df['articles'] = df['articles'].astype('category')
    df['articles_cat'] = df['articles'].cat.codes

    df = df.apply(last_k_letters, axis=1)
    path = '/Users/juanpablomejia/Desktop/german_articles/data/processed/df.csv'
    df.to_csv(path)


def last_k_letters(row):
    word = row['ger_words']
    k = len(word)
    string = ''
    for i in range(k, 0, -1):
        string += ' ' + word[-i:]
        string = string.lstrip()
    row['k_letters'] = string
    return row
