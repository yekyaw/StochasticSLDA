import onlineldavb
import categorical
import numpy as n
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

def parse_labels(labels):
    ys = [int(i) for i in labels]
    return n.array(ys)

def main():
    vocab = file('dictnostops.txt').readlines()
    docs_all = file('dataset/docs_all.txt').readlines()[:2000]
    labels_all = parse_labels(file('dataset/labels_stars.txt').readlines())[:2000]
    train_size = int(len(docs_all) * 2 / 3)
    docs_train = docs_all[:train_size]
    docs_test = docs_all[train_size:]
    labels_train = labels_all[:train_size]
    labels_test = labels_all[train_size:]
    D = len(docs_train)
    K = 8
    olda = categorical.OnlineLDA(vocab, 2, K, D, 1./K, 1./K, 64., 0.7)
#    olda.update_lambda_all(docs_train, labels_train, 2)
    olda.train(docs_train, labels_train, 20)
    
    preds1 = olda.predict(docs_test)
    report = classification_report(labels_test, preds1)
    confusion = confusion_matrix(labels_test, preds1)
    print("SLDA")
    print(report)
    print(confusion)

    gammas_train = olda.estimate_gamma(docs_train)
    gammas_test = olda.estimate_gamma(docs_test)
    clf = svm.LinearSVC()
    clf.fit(gammas_train, labels_train)
    preds = clf.predict(gammas_test)
    report = classification_report(labels_test, preds)
    confusion = confusion_matrix(labels_test, preds)
    print("SVM")
    print(report)
    print(confusion)

if __name__ == "__main__":    
    main()
    
    
