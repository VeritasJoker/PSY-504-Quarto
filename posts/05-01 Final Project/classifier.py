import numpy as np
import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE


def argparse_args():
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument(
        "--mode",
        type=str,
        default="review",
        help="Data for embedding (title/review)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="log",
        help="Type of classification model (log/elastic/adaboost/randomforest)",
    )
    return parser.parse_args()


def Scale(data):
    # Scales data
    scaler = preprocessing.StandardScaler().fit(data)  # scaler
    data = scaler.transform(data)  # scale training data
    return data


def LogReg(train_x, train_y, test_x, test_y, iter):
    # logistic
    train_x = Scale(train_x)  # scale training data
    test_x = Scale(test_x)  # scale training data
    logreg = LogisticRegression(penalty=None, solver="saga", max_iter=iter).fit(
        train_x, train_y
    )
    return logreg.score(test_x, test_y)  # return accuracy


def Ridge(train_x, train_y, test_x, test_y, iter):
    # ridge
    ridge = RidgeClassifier(max_iter=iter).fit(train_x, train_y)  # ridge
    return ridge.score(test_x, test_y)  # return accuracy


def ElasticNet(train_x, train_y, test_x, test_y, iter):
    # logistic elastic net
    train_x = Scale(train_x)  # scale training data
    test_x = Scale(test_x)  # scale training data
    logreg = LogisticRegression(
        penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=iter
    ).fit(train_x, train_y)
    return logreg.score(test_x, test_y)  # return accuracy


def AdaBoost(train_x, train_y, test_x, test_y, n_est):
    # adaboost
    ada = AdaBoostClassifier(n_estimators=n_est).fit(train_x, train_y)  # adaboost
    return ada.score(test_x, test_y)  # return accuracy


def RandForest(train_x, train_y, test_x, test_y, n_est):
    # random forest
    forest = RandomForestClassifier(n_estimators=n_est).fit(
        train_x, train_y
    )  # build forest
    return forest.score(test_x, test_y)  # return accuracy


def main():

    args = argparse_args()

    train_df = pd.read_csv("train_s.csv")
    test_df = pd.read_csv("test_s.csv")
    train_review_emb = np.load("train_review_embs.npy")
    train_title_emb = np.load("train_title_embs.npy")
    test_review_emb = np.load("test_review_embs.npy")
    test_title_emb = np.load("test_title_embs.npy")

    train_y = np.array(train_df.rating)
    test_y = np.array(test_df.rating)

    print(train_review_emb.shape, train_title_emb.shape, train_y.shape)
    print(test_review_emb.shape, test_title_emb.shape, test_y.shape)

    if args.mode == "review":
        train_x = train_review_emb
        test_x = test_review_emb
    elif args.mode == "title":
        train_x = train_title_emb
        test_x = test_title_emb

    match args.model:
        case "log":
            score = LogReg(train_x, train_y, test_x, test_y, 500)
        case "elastic":
            score = ElasticNet(train_x, train_y, test_x, test_y, 500)
        case "ridge":
            score = Ridge(train_x, train_y, test_x, test_y, 500)
        case "ada":
            score = AdaBoost(train_x, train_y, test_x, test_y, 500)
        case "randomforest":
            score = RandForest(train_x, train_y, test_x, test_y, 500)
    print(args.mode, args.model, "Score:", score)


if __name__ == "__main__":
    main()
