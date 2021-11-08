import itertools
import requests
from bs4 import BeautifulSoup

import numpy as np


def predict_by_url(model, vgm_url, tokenizer, encoder):
    html_text = requests.get(vgm_url).text
    print(html_text)
    soup = BeautifulSoup(html_text, 'html.parser')

    page_head = soup.find("h1", {"class": "article__heading"})
    page_summary = soup.find("div", {"class": "article__summary"})

    page_article = soup.find("div", {"class": "article__text"})
    if not (page_article is None):
        page_article = page_article.find_all("p")

    lines = [page_head.text, page_summary.text]
    if not (page_article is None):
        for par in page_article:
            lines.append(par.text)

    layers = predict_text(lines, tokenizer, model)
    probabilities = [[], []]
    for i in range(len(layers)):
        probabilities[0].append(encoder.classes_[i])
        probabilities[1].append("".join(np.clip(np.round(layers[i] * 100.0, 4), 0, 100)).join("%"))

    return probabilities

def predict_text(lines, tokenizer,model):
    linesSeq = np.array(tokenizer.texts_to_sequences(lines))
    linesSeq = np.array(list(itertools.chain.from_iterable(linesSeq)))

    maxlen = model.layers[0].input_shape[1]
    nClasses = model.layers[-1].output_shape[1]
    c = int(np.ceil(len(linesSeq) / maxlen))
    linesSeq = np.asarray(linesSeq, dtype=np.int32)

    linesSeq = np.resize(linesSeq, (c, int(maxlen)))

    layers = model.predict(linesSeq)
    layers = np.reshape(layers, (c, nClasses))
    layers = np.sum(layers, axis=0) / float(c)
    return layers
