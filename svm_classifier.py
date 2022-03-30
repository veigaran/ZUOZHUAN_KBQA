"""
1、文本向量化
2、构建分类模型
3、模型测试及预测
"""
import pickle

import jieba
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

vocab_path = r'F:\A文档\python学习\Competition\Medication\代码\data\vocab.txt'
stopwords_path = r'F:\A文档\python学习\Competition\Medication\代码\data\stop_words.utf8'
stopwords = [w.strip() for w in open(stopwords_path, 'r', encoding='utf8') if w.strip()]


def readData(path):
    data1 = pd.read_excel(path, names=None, dtype=object, columns=["query", "label"])
    data = shuffle(data1)
    dataMat = [i for i in data["query"]]
    labelMat = [i for i in data["label"]]

    trainData = dataMat[601:]
    trainLabel = labelMat[601:]

    testData = dataMat[:400]
    testLabel = labelMat[:400]
    print(len(dataMat), dataMat[0:10], '\n', labelMat[0:10])
    train = [trainData, trainLabel]
    test = [testData, testLabel]
    return train, test


def cutWords(data_list):
    cutString = []
    for i in data_list:
        res = ''
        textcut = jieba.cut(i)
        for word in textcut:
            res += word + ' '
        cutString.append(res)
    return cutString

    # 模型预测


# def model_predict(text, model, tf):
#     """
#     :param text: 单个文本
#     :param model: 朴素贝叶斯模型
#     :param tf: 向量器
#     :return: 返回预测概率和预测类别
#     """
#     text1 = [" ".join(jieba.cut(text))]
#     # 进行tfidf特征抽取
#     text2 = tf.transform(text1)
#     predict_type = model.predict(text2)[0]
#     return predict_type


def svm(trainFile):
    train, test = readData(trainFile)
    # test = readData(testFile)

    train_dataMat = cutWords(train[0])
    train_labelMat = train[1]
    test_dataMat = cutWords(test[0])
    test_labelMat = test[1]

    stop_words = open(r'./data/stopword.txt', 'r', encoding='utf-8').read()
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig')  # 列表头部\ufeff 处理
    stop_words = stop_words.split('\n')

    #  计算单词权重
    # tfidf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2))
    train_features = tfidf.fit_transform(train_dataMat)
    print(train_features.shape)

    test_features = tfidf.transform(test_dataMat)
    print(test_features.shape)

    path1 = './model/tf.pkl'
    with open(path1, 'wb') as fw:
        pickle.dump(tfidf, fw)
    joblib.dump(tfidf, "./model/tf_idf.m")
    print("tf is done")

    clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                    verbose=0)

    # 模型训练
    clf.fit(train_features, train_labelMat)

    joblib.dump(clf, "./model/SVM.m")
    print("svm is done")

    print(clf.score(train_features, train_labelMat))
    print('训练集准确率：', metrics.accuracy_score(test_labelMat, clf.predict(test_features)))
    print('训练集召回率：', metrics.recall_score(test_labelMat, clf.predict(test_features), average='micro'))
    print('训练集准确率：', metrics.precision_score(test_labelMat, clf.predict(test_features), average='micro'))
    print('训练集F1值：', metrics.f1_score(test_labelMat, clf.predict(test_features), average='micro'))

    # Naive bayes模型测试
    # 加载停用词
    # stop_words = open(r'F:\A文档\python学习\Competition\Medication\代码\data\stopword.txt', 'r',
    #                   encoding='utf-8').read()
    # stop_words = stop_words.encode('utf-8').decode('utf-8-sig')  # 列表头部\ufeff 处理
    # stop_words = stop_words.split('\n')
    #
    # #  计算单词权重
    # tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    # train_features = tf.fit_transform(train_dataMat)
    # print(train_features.shape)
    # test_features = tf.transform(test_dataMat)
    # print(test_features.shape)
    #
    # # 生成 朴素贝叶斯分类器
    # clf_nb = MultinomialNB(alpha=0.001).fit(train_features, train_labelMat)
    # print("done")
    # print('训练集准确率：', metrics.accuracy_score(test_labelMat, clf_nb.predict(test_features)))
    # print('训练集F1值：', metrics.f1_score(test_labelMat, clf_nb.predict(test_features), average='micro'))
    # predict_type = model_predict(question, clf, tf)
    return clf, tfidf


def tfidf_features(text, vectorizer):
    """
    提取问题的TF-IDF特征
    :param text:
    :param vectorizer:
    :return:
    """
    jieba.load_userdict(vocab_path)
    words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in stopwords]
    sents = [' '.join(words)]

    tfidf = vectorizer.transform(sents).toarray()
    return tfidf


def model_predict(x, model):
    """
    预测意图
    :param x:
    :param model:
    :return:
    """
    pred = model.predict(x)
    return pred


if __name__ == '__main__':
    # trainFile = r'F:\A文档\python学习\Competition\Medication\Code\data\query6_8.xlsx'
    # clf_svm, tfidf = svm(trainFile)

    tfidf_path = r'./model/tf.pkl'  # os.path.join(cur_dir, 'model/tfidf_model.m')
    nb_test_path = r'./model/SVM.m'  # 测试nb模型

    tfidf_model = pickle.load(open(tfidf_path, "rb"))
    nb_model = joblib.load(nb_test_path)

    print('开始测试')
    s = "鲁桓公的子女有哪些？"
    tfidf_feature = tfidf_features(s, tfidf_model)
    predicted = model_predict(tfidf_feature, nb_model)

    # predict_type = model_predict(s, clf_svm, tfidf)
    # # p = model_predict(s, clf_nb, tfidf)
    print(predicted)
