import numpy as n
from lda import lda
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import re, string

def parse_doc_list(docs, vocab):
    D = len(docs)
    V = len(vocab)
    wordcts = n.zeros((D, V))
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                wordcts[d, wordtoken] += 1
    return wordcts


def parse_labels(labels):
    ys = [int(i) for i in labels]
    return n.array(ys)

def parse_vocab(vocab):
    ddict = dict()
    for word in vocab:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        ddict[word] = len(ddict)
    return ddict


def main():
    vocab = parse_vocab(file('dictnostops.txt').readlines())
    docs_all = parse_doc_list(file('dataset/docs_newsgroups.txt').readlines(), vocab)
    labels_all = parse_labels(file('dataset/labels_newsgroups.txt').readlines())
    train_size = int(len(docs_all) * 2 / 3)
    labels_train = labels_all[:train_size]
    labels_test = labels_all[train_size:]

    model = lda.LDA(n_topics=10)
    model.fit(docs_all)
    gammas_all = model.doc_topic_
    gammas_train = gammas_all[:train_size, :]
    gammas_test = gammas_all[train_size:, :]
    clf = LogisticRegression()
    clf.fit(gammas_train, labels_train)
    preds = clf.predict(gammas_test)
    report = classification_report(labels_test, preds)
    confusion = confusion_matrix(labels_test, preds)
    print("LDA SVM")
    print(report)
    print(confusion)

if __name__ == "__main__":    
    main()
    
    
