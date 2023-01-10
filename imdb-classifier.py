import os, glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

punctuation = punctuation + "«»·”…“‘’。、，—！()▲（）《》②「」）（"


def create_df(infiles, pol):
    """create a pandas dataframe from files in dir"""

    # neg = 0 / pos = 1
    if pol == "neg":
        df = pd.DataFrame(
            {"text": [""] * len(os.listdir(infiles)), "pol": [0] * len(os.listdir(neg))}
        )
    elif pol == "pos":
        df = pd.DataFrame(
            {"text": [""] * len(os.listdir(infiles)), "pol": [1] * len(os.listdir(neg))}
        )
    else:
        raise TypeError("'pol' value should be either 'neg' or 'pos'")

    file_list = []

    for filename in glob.glob(infiles + "*.txt"):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            file_list.append(f.readlines())

    for i in range(0, len(os.listdir(infiles))):
        df["text"][i] = "".join(file_list[i])

    return df


def shuffle_df(df):
    """shuffle columns in df"""

    df = df.sample(frac=1, ignore_index=True, random_state=0)

    df.index.names = ["index"]

    return df


def preprocessor(X):
    """preprocess text"""

    X = [x.lower() for x in X if x not in punctuation]

    for i, e in enumerate(X):
        e = e.replace("<br /><br />", "")
        e = e.replace("\n", "")
        X[i] = e

    return X


def train_model(X, y, outfile):
    """train model & get eval stats"""

    """ divide dataset into test/train """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=64
    )

    """ train model"""
    model = MultinomialNB()
    model.fit(X_train, y_train)

    """ get accuracy """
    accuracy = model.score(X_test, y_test)

    if not os.path.exists('./model/'):
        os.system('mkdir model')

    with open("model/model.pickle", "wb") as file:
        print(f"Saving model with {accuracy} accuracy...")
        pickle.dump(model, file)

    file_in = open("model/model.pickle", "rb")
    model = pickle.load(file_in)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    if not os.path.exists('./metrics/'):
        os.system('mkdir metrics')

    """get results of classification report in outfile"""
    with open('./metrics/'+outfile, "w") as file:
        file.write(
            f"##### Results on test set :\n\n{metrics.classification_report(y_test, y_pred)}"
        )
        file.close()

    """plot confusion matrix"""
    plot = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(plot, annot=True)
    plt.savefig("./metrics/confusion_matrix.png")

    return model


def predict(model, sentence):
    """predicts polarity based on sentence passed to the function"""

    print("Review |", sentence)

    sentence = preprocessor([sentence])
    x = vectorizer.transform(sentence)
    pred = model.predict(x)

    if pred[0] == 0:
        print("This review is negative.")
        return 0
    elif pred[0] == 1:
        print("This review is positive.")
        return 1


if __name__ == "__main__":

    neg = "./data/imdb_smol/neg/"
    pos = "./data/imdb_smol/pos/"

    neg_df = create_df(neg, pol="neg")
    pos_df = create_df(pos, pol="pos")

    pos_neg_df = shuffle_df(pd.concat([neg_df, pos_df]))

    X, y = pos_neg_df.text, pos_neg_df.pol

    X = preprocessor(X)

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(X)

    """send dataset to csv file"""
    pos_neg_df.to_csv('./data/dataset.tsv', sep='\t', columns=['text', 'pol'], mode='wb', encoding='utf-8')

    """train model"""
    model = train_model(X, y, "classification-report.txt")

    """make prediction"""
    prediction = predict(
        model,
        "Extremely entertaining throwback-appearing (think old Hollywood cinematography and score) blood-soaked origin story of a girl with big dreams. After hints of malevolence, a chain of circumstances and events leads Pearl down a dark path. The key here is Mia Goth's performance. One minute she comes across as a sympathetic character, the next she is beyond creepy, and there are still times in this film with laughter interspersed between moments of horrific violence. Perhaps the best moment in the film is a monologue that is best described as otherworldly. Really good watch, tremendous performance.",
    )
    prediction = predict(
        model,
        "Pearl is that prequel story that no one asked for. There was nothing new in this movie. It was just a boring piece of drama. You can't feel any intensity or any thrill in the story. There was not even a single scene where you would think what's gonna happen next. The performances by the lead actress was good but to be honest, I wasn't able to connect with her. Overall, it was an average drama and a wasted opportunity. Now, it's sequel MaxXxine is also coming. No one asked either for sequel or for prequel and I don't think anyone was interested in knowing psycho's origin story or is interested in knowing what happened after X.",
    )
