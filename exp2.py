import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as p
from sklearn.decomposition import FastICA, PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from scipy import linalg
import matplotlib as mpl
import itertools

import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

from sklearn import random_projection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
from scipy.stats import kurtosis 

m_depth = list(range(1, 40))
m_depth.append(None)

n_estimator = list(range(100, 2100, 100))
n_estimator.append(50)

cv = 10

oh = preprocessing.OneHotEncoder(sparse=False)

normalizer = preprocessing.MinMaxScaler()

scaler = preprocessing.StandardScaler()

idTransformer = preprocessing.FunctionTransformer(None)

# assumed ordinals
education = ["unknown", "primary", "secondary", "tertiary"]
credit_default = ["unknown", "yes", "no"]
bm_answer_ = ["no","yes"]
# job = ['unknown','unemployed',]

education_encoder = preprocessing.OrdinalEncoder(categories=[education])
credit_default_encoder = preprocessing.OrdinalEncoder(categories=[credit_default])
bm_answer_encoder = preprocessing.OrdinalEncoder(categories=[bm_answer_])

liv_train, liv_test, liv_ans_train, liv_ans_test = None, None, None, None

bm_train, bm_test, bm_ans_train, bm_ans_test = None, None, None, None

liv_full, liv_ans_full = None, None

bm_full, bm_ans_full = None, None

np.random.seed(65)

# https://chrisalbon.com/machine_learning/feature_engineering/select_best_number_of_components_in_tsvd/
# Create a function
def sel_n_comp(vr, gv: float) -> int:
    tv = 0.0
    
    nc = 0
    
    for ev in vr:
        
        tv += ev
        
        nc += 1
        
        if tv >= gv:
            break
            
    return nc


def create_intake():
    global liv_train, liv_test, liv_ans_train, liv_ans_test, bm_train, bm_test, bm_ans_train, bm_ans_test, liv_full, liv_ans_full, bm_full, bm_ans_full

    liv_dataset = pd.read_csv("indian_liver_patient_dataset.csv")
    liv_answer = pd.DataFrame(data=liv_dataset["class"])#liv_dataset["class"]
    liv_ans_transformer = make_column_transformer(
        (idTransformer, ["class"]),
    )
    liv_answer = liv_ans_transformer.fit_transform(liv_answer)

    liv_dataset = liv_dataset.drop("class", axis=1)

    liv_transformer = make_column_transformer(
        (oh, ["gender"]),
        (
            idTransformer,
            ["age", "TB", "DB", "alkphos", "sgpt", "sgot", "TP", "ALB", "A_G"],
        ),
    )

    liv_full_set = liv_transformer.fit_transform(liv_dataset)

    liv_train, liv_test, liv_ans_train, liv_ans_test = train_test_split(
        liv_full_set, liv_answer, test_size=0.2, random_state=30
    )


    liv_train = scaler.fit_transform(liv_train)
    liv_test = scaler.fit_transform(liv_test)

    #liv_full = scaler.fit_transform(liv_full_set)
    liv_full = liv_full_set
    liv_ans_full = liv_answer

    bm_dataset = pd.read_csv("bank_marketing_dataset.csv")
    bm_answer = pd.DataFrame(data=bm_dataset["y"])

    

    bm_transformer = make_column_transformer(
        (bm_answer_encoder, ["y"]),
    )

    bm_answer = bm_transformer.fit_transform(bm_answer)

    bm_dataset = bm_dataset.drop("y", axis=1)
    # drop predicted outcome
    bm_dataset = bm_dataset.drop("poutcome", axis=1)
    # drop day of month as this is assumed to be a noisy feature
    bm_dataset = bm_dataset.drop("day", axis=1)

    bm_transformer = make_column_transformer(
        (education_encoder, ["education"]),
        (credit_default_encoder, ["default"]),
        (oh, ["job", "marital", "housing", "loan", "contact", "month"]),
        (
            idTransformer,
            ["age", "balance", "duration", "campaign", "pdays", "previous"],
        ),
    )

    bm_full_set = bm_transformer.fit_transform(bm_dataset)

    bm_train, bm_test, bm_ans_train, bm_ans_test = train_test_split(
        bm_full_set, bm_answer, test_size=0.2, random_state=30
    )

    #print(bm_ans_train[6])


    bm_train = scaler.fit_transform(bm_train)
    bm_test = scaler.fit_transform(bm_test)

    #bm_full = scaler.fit_transform(bm_full_set)
    bm_full = bm_full_set
    bm_ans_full = bm_answer

    #print(liv_full)
    #print(liv_ans_full)
    #print()
    #print(bm_full)
    #print(bm_ans_full)

    print("loaded data")

def write_to_csv_pca(
    dataset_name, orig_shape, new_shape, num_best_components,cv_score, acc_score, kurtosis
):
    fname = "files/part2/{}_PCA_metrics.csv".format(dataset_name)
    try:
        f = open(fname)
    except IOError:
        f = open(fname, "a+")
        f.write(
            "Orig Shape,New Shape,Num Best Components,CV Score,Test Acc Score, Kurtosis for Norm Dist\n"
        )
    finally:
        f.close()
    with open(fname, "a+") as f:
        f.write(
            "{},{},{},{},{},{}\n".format(
                '"{}"'.format(orig_shape),
                '"{}"'.format(new_shape),
                num_best_components,
                cv_score,
                acc_score,
                kurtosis
            )
        )

def runPCA(X, y, dataname):
    plt.clf()
    ress = PCA().fit(liv_full)
    plt.plot(range(1, len(ress.explained_variance_ratio_) + 1), ress.explained_variance_ratio_)
    plt.plot(range(1, len(ress.explained_variance_ratio_) + 1), np.cumsum(ress.explained_variance_ratio_))
    plt.title("Cumulative Explained Variance")
    plt.xlabel('Num components')
    plt.ylabel('Cumulative Explained Variance')
    plt.savefig("files/part2/"+ dataname + "_PCA_cumulative_exp_variance_" + "plot.png")
    plt.clf()

    nc = 2

    ress2 = PCA(n_components= nc).fit(X)
    origshape = X.shape
    ress3 = ress2.transform(X)
    newshape = ress3.shape
    
    X_tr, X_te, y_tr, y_te = train_test_split(ress3, y, test_size=0.2)
    classifier = DecisionTreeClassifier(max_depth = 26, min_samples_leaf = 3)
    cvscore = cross_val_score(classifier, X_tr, y_tr, cv=20).mean()

    classifier.fit(X_tr, y_tr)
    y_pr = classifier.predict(X_te)
    accscore = accuracy_score(y_te, y_pr)

    write_to_csv_pca(dataname, origshape,newshape, nc, cvscore,accscore, kurtosis(y))

    plt.figure(figsize=(6,4))
    plt.title("PCA Components plot")
    for o in range(0,nc):
        for j in range(0,nc):
            if o == j:
                continue
            plt.scatter(ress3[:,o], ress3[:,j])
    plt.savefig("files/part2/"+ dataname + "_PCA_components_scatter_" + "plot.png")

def write_to_csv_ica(
    dataset_name, best_num_components, cv_score, acc_score
):
    fname = "files/part2/{}_ICA_metrics.csv".format(dataset_name)
    try:
        f = open(fname)
    except IOError:
        f = open(fname, "a+")
        f.write(
            "Best Num Components,CV Score,Test Acc Score\n"
        )
    finally:
        f.close()
    with open(fname, "a+") as f:
        f.write(
            "{},{},{}\n".format(
                best_num_components,
                cv_score,
                acc_score,
            )
        )

def runICA(X, y, dataname):
    plt.clf()
    bn = (0, 0)
    for i in range(2, 15):
        ica = FastICA(n_components=i, max_iter=10000, tol=0.1).fit(X)
        ress = ica.fit_transform(X)  
        X_tr, X_te, y_tr, y_te = train_test_split(ress, y, test_size=0.2)
        classifier = DecisionTreeClassifier(max_depth = 26, min_samples_leaf = 3)
        cv_score = cross_val_score(classifier, X_tr, y_tr, cv=20).mean()
        if (cv_score > bn[1]):
            bn = (i, cv_score)

    classifier = DecisionTreeClassifier(max_depth = 26, min_samples_leaf = 3)
    print("B comp: " + str(bn[0]))
    ica = FastICA(n_components=int(bn[0]), max_iter=10000, tol=0.1).fit(X)
    ress = ica.fit_transform(X) 
    X_tr, X_te, y_tr, y_te = train_test_split(ress, y, test_size=0.2)
    classifier.fit(X_tr, y_tr)
    y_pr = classifier.predict(X_te)
    accscore = accuracy_score(y_te, y_pr)
    print("Test Acc score: " + str(accscore))

    nc = min(bn[0],ress.shape[1])
    write_to_csv_ica(dataname, nc,bn[1],accscore)

    plt.figure(figsize=(6,4))
    plt.title("ICA Components plot")
    for o in range(0,nc):
        for j in range(0,nc):
            if o == j:
                continue
            plt.scatter(ress[:,o], ress[:,j])
    plt.savefig("files/part2/"+ dataname + "_ICA_components_scatter_" + "plot.png")

def write_to_csv_rp(
    dataset_name, best_num_components, cv_score, acc_score
):
    fname = "files/part2/{}_RP_metrics.csv".format(dataset_name)
    try:
        f = open(fname)
    except IOError:
        f = open(fname, "a+")
        f.write(
            "Best Num Components,CV Score,Test Acc Score\n"
        )
    finally:
        f.close()
    with open(fname, "a+") as f:
        f.write(
            "{},{},{}\n".format(
                best_num_components,
                cv_score,
                acc_score,
            )
        )


def runRP(X, y, dataname):
    plt.clf()
    bn = (0, 0)
    for i in range(2, 15):
        rp = random_projection.SparseRandomProjection(n_components=i, random_state=65).fit(X)
        ress = rp.fit_transform(X)  
        X_tr, X_te, y_tr, y_te = train_test_split(ress, y, test_size=0.2)
        classifier = DecisionTreeClassifier(max_depth = 26, min_samples_leaf = 3)
        cv_score = cross_val_score(classifier, X_tr, y_tr, cv=20).mean()
        if (cv_score > bn[1]):
            bn = (i, cv_score)

    classifier = DecisionTreeClassifier(max_depth = 26, min_samples_leaf = 3)
    print("B comp: " + str(bn[0]))
    rp = random_projection.SparseRandomProjection(n_components=bn[0], random_state=65).fit(X)
    ress = rp.fit_transform(X) 
    X_tr, X_te, y_tr, y_te = train_test_split(ress, y, test_size=0.2)
    classifier.fit(X_tr, y_tr)
    y_pr = classifier.predict(X_te)
    accscore = accuracy_score(y_te, y_pr)
    print("Test Acc score: " + str(accscore))

    nc = min(bn[0],ress.shape[1])
    write_to_csv_rp(dataname, nc,bn[1],accscore)

    plt.figure(figsize=(6,4))
    plt.title("RP Components plot")
    for o in range(0,nc):
        for j in range(0,nc):
            if o == j:
                continue
            plt.scatter(ress[:,o], ress[:,j])
    plt.savefig("files/part2/"+ dataname + "_RP_components_scatter_" + "plot.png")


def write_to_csv_svd(
    dataset_name, best_num_components, cv_score, acc_score
):
    fname = "files/part2/{}_SVD_metrics.csv".format(dataset_name)
    try:
        f = open(fname)
    except IOError:
        f = open(fname, "a+")
        f.write(
            "Best Num Components,CV Score,Test Acc Score\n"
        )
    finally:
        f.close()
    with open(fname, "a+") as f:
        f.write(
            "{},{},{}\n".format(
                best_num_components,
                cv_score,
                acc_score,
            )
        )


def runSVD(X, y, dataname):
    plt.clf()

    tsvd = TruncatedSVD(n_components=X.shape[1]-1)
    ress = tsvd.fit(X)

    tvr = ress.explained_variance_ratio_
    plt.plot(range(1, len(tvr) + 1), tvr)
    plt.plot(range(1, len(tvr) + 1), np.cumsum(tvr))
    plt.title("Cumulative Explained Variance")
    plt.xlabel('Num components')
    plt.ylabel('Cumulative Explained Variance')
    plt.savefig("files/part2/"+ dataname + "_SVD_cumulative_exp_variance_" + "plot.png")

    nc = sel_n_comp(tvr, 0.99)

    classifier = DecisionTreeClassifier(max_depth = 26, min_samples_leaf = 3)
    tsvd = TruncatedSVD(n_components=nc).fit(X)
    ress2 = tsvd.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(ress2, y, test_size=0.2)
    cvscore = cross_val_score(classifier, X_tr, y_tr, cv=20).mean()
    print("Cross validation score: " + str(cvscore))

    classifier.fit(X_tr, y_tr)
    y_pred =classifier.predict(X_te)
    accscore = accuracy_score(y_te, y_pred)
    print("Test Accuracy: " + str(accscore))

    write_to_csv_svd(dataname, nc,cvscore,accscore)

    plt.figure(figsize=(6,4))
    plt.title("SVD Components plot")
    for o in range(0,nc):
        for j in range(0,nc):
            if o == j:
                continue
            plt.scatter(ress2[:,o], ress2[:,j])
    plt.savefig("files/part2/"+ dataname + "_SVD_components_scatter_" + "plot.png")



def run():
    create_intake()
    ##PCA
    #liver patients
    runPCA(liv_full, liv_ans_full, "liver_patients")
    #bm
    runPCA(bm_full, bm_ans_full, "bm")
    print("done w PCA")

    ##ICA
    #liver patients
    runICA(liv_full, liv_ans_full, "liver_patients")
    #bm
    runICA(bm_full, bm_ans_full, "bm")
    print("done w ICA")

    ##Rand Projections
    #liver patients
    runRP(liv_full, liv_ans_full, "liver_patients")
    #bm
    runRP(bm_full, bm_ans_full, "bm")
    print("done w RP")

    ##SVD
    #liver patients
    runSVD(liv_full, liv_ans_full, "liver_patients")
    #bm
    runSVD(bm_full, bm_ans_full, "bm")
    print("done w SVD")




def main():
    run()
    return


if __name__ == "__main__":
    main()