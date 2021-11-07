import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from sklearn.decomposition import PCA
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
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

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

kmeans_figsize = [12, 20]

def gmmj(gmmp, gmmq, ns=10**5):
    # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4
    
    X = gmmp.sample(ns)[0]
    log_p_X = gmmp.score_samples(X)
    log_q_X = gmmq.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmmq.sample(ns)[0]
    log_p_Y = gmmp.score_samples(Y)
    log_q_Y = gmmq.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


def getB(arr, X):
    return arr[np.argsort(arr)[:X]]

def run_kmeans(X, k, dataname):
    plt.clf()
    pln = len(k)
    figure, axes = plt.subplots(pln, 2, figsize=kmeans_figsize)
    col = 0
    for nc in k:
        kmeans = KMeans(n_clusters=nc, algorithm="full")
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=axes[col][0])
        visualizer.fit(X)
        visualizer.finalize()
        
        kmeans = KMeans(n_clusters=nc, algorithm="full")
        visualizer = InterclusterDistance(kmeans, ax=axes[col][1])
        visualizer.fit(X)
        visualizer.finalize()
        col =col+1
        print("finished kmeans for clusters : " + str(nc))
        #plt.savefig("files/part1/"+ dataname  + "_" + str(nc) + "_kmeans_plot.png")
        #plt.clf()
        #figure, axes = plt.subplots(1, 2)
    plt.savefig("files/part1/"+ dataname  + "_full_kmeans_plot.png")
        #plt.clf()


def silhoutte_plt(X, k, m, dataname, modelname, mm):
    plt.clf()
    n_cs=np.arange(2, k)
    s=[]
    s_err=[]
    iters=k
    for n in n_cs:
        tmp_s=[]
        for _ in range(iters):
            clf=m(n).fit(X) 
            labels=clf.predict(X)
            ss=metrics.silhouette_score(X, labels, metric=mm)
            tmp_s.append(ss)
        val=np.mean(getB(np.array(tmp_s), int(iters/5)))
        err=np.std(tmp_s)
        s.append(val)
        s_err.append(err)
    plt.errorbar(n_cs, s, yerr=s_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_cs)
    plt.xlabel("Num clusters ({})".format("Tested"))
    plt.ylabel("Score")
    plt.savefig("files/part1/"+ dataname + "_" + modelname + "_sil_plot.png")

def dist_bw_gmmplots(X, n, dataname):
    plt.clf()
    n_cs=np.arange(2, n)
    iters=n
    results=[]
    res_sigs=[]
    for n in n_cs:
        dist=[]

        for iteration in range(iters):
            train, test=train_test_split(X, test_size=0.5)

            gmm_train=GaussianMixture(n, n_init=2).fit(train) 
            gmm_test=GaussianMixture(n, n_init=2).fit(test) 
            dist.append(gmmj(gmm_train, gmm_test))
        sc=getB(np.array(dist), int(iters/5))
        result=np.mean(sc)
        rs=np.std(sc)
        results.append(result)
        res_sigs.append(rs)


    plt.errorbar(n_cs, results, yerr=res_sigs)
    plt.title("Distance b/w Train and Test Gaussian MMs", fontsize=18)
    plt.xticks(n_cs)
    plt.xlabel("Num components")
    plt.ylabel("Dist")
    plt.savefig("files/part1/"+ dataname + "_Gaussian_M_" + "plot.png")
    #plt.show()

def bics(X, n, dataname):
    plt.clf()
    n_cs=np.arange(2, n)
    b=[]
    b_err=[]
    iterations=n
    for n in n_cs:
        tmp_bic=[]
        for _ in range(iterations):
            gmm=GaussianMixture(n, n_init=2).fit(X) 

            tmp_bic.append(gmm.bic(X))
        val=np.mean(getB(np.array(tmp_bic), int(iterations/5)))
        err=np.std(tmp_bic)
        b.append(val)
        b_err.append(err)

    plt.errorbar(n_cs,b, yerr=b_err, label='BIC')
    plt.title("BIC Scores", fontsize=20)
    plt.xticks(n_cs)
    plt.xlabel("Num components")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("files/part1/"+ dataname + "_BIC_scr_" + "plot.png")
    #plt.show()
    plt.clf()
    
    plt.errorbar(n_cs, np.gradient(b), yerr=b_err, label='BIC')
    plt.title("Gradient of BIC Scores", fontsize=20)
    plt.xticks(n_cs)
    plt.xlabel("Num components")
    plt.ylabel("gradient")
    plt.legend()
    plt.savefig("files/part1/"+ dataname + "_BIC_Gradient_" + "plot.png")
    #plt.show()
    plt.clf()

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

def run():
    create_intake()
    
    ##KMeans
    k = list(range(2,5))
    #liver patients
    silhoutte_plt(liv_full, 10, KMeans, "liver_patients", "KMeans",  "euclidean")
    print("sil plot done for liver kmeans")
    run_kmeans(liv_full, k, "liver_patients")
    print("kmeans plot done for liver")



    print("finished first data set")


    #bank marketing
    silhoutte_plt(bm_full, 10, KMeans, "bm", "KMeans", "manhattan")
    print("sil plot done for bm kmeans")
    run_kmeans(bm_full, k, "bm")
    print("kmeans plot done for bm")

    ##Gaussian Mixture

    k = list(range(2,5))
    #liver patients
    silhoutte_plt(liv_full, 10, GaussianMixture, "liver_patients", "Gaussiam_M", "euclidean")
    print("sil plot done for liver gaussian mix")
    dist_bw_gmmplots(liv_full, 10, "liver_patients")
    print("gaussian plots done for liver gaussian mix")
    bics(liv_full, 10, "liver_patients")
    print("bic plots done for liver patients gaussian mix")

    silhoutte_plt(bm_full, 10, GaussianMixture, "bm", "Gaussiam_M", "manhattan")
    print("sil plot done for bm gaussian mix")
    dist_bw_gmmplots(bm_full, 10, "bm")
    print("gaussian plots done for bm gaussian mix")
    bics(bm_full, 10, "bm")
    print("bic plots done for bm gaussian mix")


def main():
    run()
    return


if __name__ == "__main__":
    main()
