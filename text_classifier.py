import spacy
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle

nlp = spacy.load("uk_core_news_sm")
punctuations = string.punctuation
stop_words = spacy.lang.uk.stop_words.STOP_WORDS


# пасинг навчальних даних
def get_headlines_exel(File_name):
    sample_data = pd.read_excel(File_name, dtype=str)
    return sample_data


# Функція токенізації
def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)

    # леманізація
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # видалення стоп-слів
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    return mytokens


# Очистка тексту та приведення до нижнього регістру
def clean_text(text):
    return text.strip().lower()


# трансформер навчальних даних
class predictors(sklearn.base.TransformerMixin):
    def transform(self, X, **transform_params):
        # очистка тексту
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# створення NumPy масиву з цифр value розміром size
def createArrey(value, size):
    b = np.zeros(size, dtype=int)
    b += value
    return b

# розмітка даних та створення тестової і навчальної вибірок
def get_train_test_split(df, labels):
    X = pd.Series()
    y = pd.Series()
    for i in labels.index:
        X = pd.concat([X, df.iloc[:, i]], ignore_index=True)
        ser_tmp = pd.Series(createArrey(i, len(df.iloc[:, i])))
        y = pd.concat([y, ser_tmp], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test

# зберігаємо модель та мітки
def save_model_label(model, labels):
    filename = "my_model.pickle"

    pickle.dump(model, open(filename, "wb"))

    filename_labels = "my_labels.pickle"

    pickle.dump(labels, open(filename_labels, "wb"))

# завантажуємо модель та мітки
def load_model_label():
    loaded_model = pickle.load(open("my_model.pickle", "rb"))
    loaded_lables = pickle.load(open("my_labels.pickle", "rb"))
    return loaded_model, loaded_lables

def create_model():
    train_file = 'categories_data.xlsx'
    hedlines_train_df = get_headlines_exel(train_file)
    hedlines_train_df.drop('Unnamed: 0', axis=1, inplace=True)

    print(hedlines_train_df.columns)
    print(hedlines_train_df[:5])
    print('------------------------')
    print('Number of stop words:', len(stop_words))
    print('First ten stop words:', stop_words)
    # векторизатор методом мішок слів із використанням кастомного токенізатора
    bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
    labels = pd.Series(hedlines_train_df.columns)
    print(labels)
    X_train, X_test, y_train, y_test = get_train_test_split(hedlines_train_df, labels)

    classifier = LogisticRegression()

    # створюємо пайплайн для обробки даних і навчання моделі
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', bow_vector),
                     ('classifier', classifier)])

    # створюємо модель

    pipe.fit(X_train, y_train)

    predicted = pipe.predict(X_test)

    # Оцінимо модель
    print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
    save_model_label(pipe, labels)