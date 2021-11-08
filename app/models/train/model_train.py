import re
import io
import json

import numpy as np

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.models import train

def train_model(save_path):
    data = train.csv_data[['topics']]
    data['text'] = train.csv_data['title'] + '. ' + train.csv_data['text']

    texts = data['text'].astype(str).values
    topics = list(data['topics'].values)
    regex_cite = re.compile(r'[n](?P<word>[^n]+)[n]')
    regex_par = re.compile(r'(?P<sign>[.;])[n](?P<word>[\S]+)')
    for i in range(len(texts)):
      texts[i] = re.sub(regex_cite, r' \g<word> ', re.sub(regex_par, r'\g<sign> \g<word> ', texts[i]))
    encoder = LabelEncoder()  # метод кодирования тестовых лейблов
    encoder.fit(topics)
    topicsEncoded = encoder.transform(topics)  # one-hot

    step_to_categorical = 1000
    vec_topics = np.ndarray(shape=(0, len(encoder.classes_)))
    for i in range(0, len(topics), step_to_categorical):
        vec_topics = np.concatenate(
            (vec_topics, utils.to_categorical(topicsEncoded[i:i + step_to_categorical], len(encoder.classes_))), axis=0)

    maxWordsCount = 100000
    tokenizer = Tokenizer(num_words=maxWordsCount, filters='«»!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                          split=' ', oov_token='unknown', char_level=False)

    tokenizer.fit_on_texts(
        texts)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности

    # для Эмбэддинга
    Sequences = tokenizer.texts_to_sequences(texts)  # последовательность индексов
    npSequences = np.array(Sequences)

    xTrainE, xValE, yTrainE, yValE = train_test_split(npSequences, vec_topics, test_size=0.2)

    maxlen = 500

    xTrainE = pad_sequences(xTrainE, maxlen=maxlen)
    xValE = pad_sequences(xValE, maxlen=maxlen)

    # Полносвязная сеть
    modelE = Sequential()
    # Cлой представления всего словаря слов в векторном представлении
    modelE.add(Embedding(maxWordsCount, 200, input_length=maxlen))
    # Слой регуляризации Dropout
    modelE.add(SpatialDropout1D(0.2))
    # Cлой преобразования двумерных данных в одномерные
    modelE.add(Flatten())
    # Слой пакетной нормализации
    modelE.add(BatchNormalization())
    # Полносвязный слой
    modelE.add(Dense(1000, activation="relu"))
    # Слой регуляризации Dropout
    modelE.add(Dropout(0.2))
    # Слой пакетной нормализации
    modelE.add(BatchNormalization())
    # Выходной полносвязный слой
    modelE.add(Dense(len(encoder.classes_), activation='softmax'))

    modelE.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    modelE.summary()

    nClasses = len(encoder.classes_)
    statistics = np.zeros([nClasses, nClasses], dtype=np.int64)

    validation_predicts = modelE.predict(xValE)

    predict_ind = np.argmax(validation_predicts, axis=1)
    test_ind = np.argmax(yValE, axis=1)

    for i in range(len(predict_ind)):
        statistics[test_ind[i], predict_ind[i]] += 1

    print("Статистика: ")
    classes = encoder.classes_
    percentage = 0
    for i in range(nClasses):
        percent = np.round((statistics[i, i] / np.sum(statistics[i]) * 100.0), 3)
        percentage += percent
        print("Тему:", classes[i], "верно предсказывали в ", percent, "% случаев")
    print("Средняя точность составила: ", percentage / 10.0, "%")

    modelE.save("".join(save_path).join('/model'))
    tokenizer_json = tokenizer.to_json()
    with io.open("".join(save_path).join("tokenizer.json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    np.save("".join(save_path).join("encoder_classes.npy"), encoder.classes_)


