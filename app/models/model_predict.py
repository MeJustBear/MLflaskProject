import itertools
import requests
from bs4 import BeautifulSoup

import numpy as np


def predict_by_url(modelE, vgm_url, tokenizer, encoder):
    html_text = requests.get(vgm_url).text
    print(html_text)
    soup = BeautifulSoup(html_text, 'html.parser')

    lines = []

    page_head = soup.find("h1", {"class": "article__heading"})
    page_summary = soup.find("div", {"class": "article__summary"})

    page_article = soup.find("div", {"class": "article__text"})
    if not page_article == None:
        page_article = page_article.find_all("p")

    lines = [page_head.text]
    lines.append(page_summary.text)
    if not page_article == None:
        for par in page_article:
            lines.append(par.text)

    linesSeq = np.array(tokenizer.texts_to_sequences(lines))
    linesSeq = np.array(list(itertools.chain.from_iterable(linesSeq)))

    maxlen = modelE.layers[0].input_shape[1]
    nClasses = modelE.layers[-1].output_shape[1]
    c = int(np.ceil(len(linesSeq) / maxlen))
    linesSeq = np.asarray(linesSeq, dtype=np.int32)

    linesSeq = np.resize(linesSeq, (c, int(maxlen)))

    layers = modelE.predict(linesSeq)
    layers = np.reshape(layers, (c, nClasses))
    layers = np.sum(layers, axis=0) / float(c)

    probabilities = []
    for i in range(len(layers)):
        probabilities.append(str(encoder.classes_[i] + str(np.clip(np.round(layers[i] * 100.0, 4), 0, 100)) + "%\n"))

    return probabilities
