import categorical, onlineldavb
import numpy as n
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def parse_labels(labels):
    ys = [int(i) for i in labels]
    return n.array(ys)

def main():
    vocab = open('dataset/yelp-vocab.txt').read().splitlines()
    docs_all = file('dataset/yelp-stemmed.txt').readlines()
    labels_all = parse_labels(open('dataset/labels_stars.txt').read().splitlines())
    docs_train, docs_test, labels_train, labels_test = train_test_split(docs_all, labels_all, train_size=50000)

    vectorizer = CountVectorizer(vocabulary=vocab)
    linear_svm = svm.LinearSVC(penalty='l1', dual=False)
    linear_svm.fit(vectorizer.transform(docs_train), labels_train)
    preds = linear_svm.predict(vectorizer.transform(docs_test))
    report = classification_report(labels_test, preds)
    confusion = confusion_matrix(labels_test, preds)
    print(report)
    print(confusion)
    
    D = len(docs_train)
    K = 30
    olda = onlineldavb.OnlineLDA(vocab, K, D, alpha=1./K, zeta=1./K, sigma=1)
#    olda.update_lambda_all(docs_train, labels_train)
    olda.train(docs_train, labels_train, 40)
    n.savetxt("yelp_model/lambda.dat", olda._lambda)
    n.savetxt("yelp_model/eta.dat", olda._eta)
    
    preds = olda.predict(docs_test)
    report = classification_report(labels_test, preds)
    confusion = confusion_matrix(labels_test, preds)
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
    
    
