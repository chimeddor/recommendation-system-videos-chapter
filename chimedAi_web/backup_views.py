# startrecommendation
import assemblyai as aai
# import youtube_dl
import pafy
import io
import re
import os
import ast
import sys
import nltk
import time
import json
import random
import pickle
import requests
import markdown
import statistics
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from io import StringIO
import tensorflow as tf
from pathlib import Path
from pytube import YouTube
from pytube import Playlist
from tensorflow import keras
import gensim.downloader as api
from collections import Counter

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from itertools import combinations
from tensorflow.keras import layers
from scipy.sparse import csr_matrix
from matplotlib import font_manager, rc
from sklearn.feature_extraction.text import CountVectorizer

import pprint
import gensim
import seaborn as sns
from time import sleep
from pandas import DataFrame
from sklearn.manifold import TSNE

from sklearn import cluster
from sklearn import metrics
from collections import Counter
from gensim.models import Word2Vec
from keras.utils import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.cluster import KMeansClusterer

from sklearn.model_selection import KFold

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import string
from matplotlib import cm

downloads = {
    "tokenizers/punkt":"punkt",
    "corpora/stopwords":"stopwords",
    "corpora/omw-1.4.zip":"omw-1.4",
    "corpora/wordnet.zip":"wordnet"
}

for resource_path, resource_name in downloads.items():
    if not nltk.data.find(resource_path):
        nltk.download(resource_name)

# stop_words_l=stopwords.words('english')
stop_words =stopwords.words('english')

import time
from sklearn.feature_extraction.text import CountVectorizer

# endrecommendation

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from keras.preprocessing.text import Tokenizer


# django
from django.shortcuts import render,redirect,HttpResponse, get_object_or_404
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Users, Video
from django.contrib import messages
from django.contrib.auth import get_user_model, logout
from django.contrib.auth.hashers import make_password
from urllib.parse import parse_qs, urlparse
# from chimedAi_web.models import Users

from datetime import datetime, timezone

W2V_PATH="/var/www/chimedAi/word2vec/GoogleNews-vectors-negative300.bin"    


saved_category = ['creating machine learning model scratch using kaggle',
                    'create text embedding python using',
                    'big data analytics spark part',
                    'getting started cpu',
                    'big data hadoop',
                    'recommendation systems using tensorflow',
                    'ai vs machine learning deep',
                    'kubernetes crash course',
                    'google cloud github',
                    'word embeddings knowledge graphs',
                    'deep learning recommendations',
                    'software architecture practical way prem cloud']
#search_recommendation 안에 있는 코드를 밖으로 꺼내기... 
#검색 버팅을 클릭 전에 실행되어 있어야 함. 
#버팅 클릭마다 새로 클러스터링하고 gnn도 새로 구현되면 안 돼...
# def concact_data(new_video_name):

#     mainData=pd.read_csv('/var/www/chimedAi/dataset/uurchilj_ashiglasan_chapter.csv')
#     tst_id = []
#     tst_VideoName = []
#     tst_headline = []
#     tst_summary = []
#     tst_gist = []
#     tst_num = 0

#     result_dic = {}

#     #zuwhun unuudrin videonii ner bolon chapter shalgah heseg
#     files2 = open(f'/var/www/chimedAi/chapters/{new_video_name}.json')
#     loadedFile = json.load(files2)

#     for chapters in loadedFile['chapters']:
    
#         #auto로 동영상 이름을 가질 수 있도록 해 줘야 한다.
#         tst_VideoName.append(new_video_name)
#         tst_headline.append(chapters['headline'])
#         tst_gist.append(chapters['gist'])
#         tst_summary.append(chapters['summary'])


#     test_data = {
#         'videoname': tst_VideoName,
#         'gist(chapter)': tst_gist,
#         'headline': tst_headline,
#         'text(summary)': tst_summary,
#         'category': tst_VideoName,
#         'type': 'lecture'
#     }

#     concData = pd.DataFrame(test_data)
#     #mash chuhal data
#     lastData = pd.concat([mainData,concData],ignore_index=True)
#     ldata = lastData.loc[:, ~lastData.columns.str.contains('^Unnamed')]
#     ldata.to_csv('/var/www/chimedAi/dataset/uurchilj_ashiglasan_chapter.csv', index=False)

#     print("--- created uurchilj_ashiglasan_chapter.csv ---")
#     return ldata

dataset = pd.read_csv("/var/www/chimedAi/dataset/uurchilj_ashiglasan_chapter.csv")
#stopwords zone
article_chapter = dataset['text(summary)']

#niit chapteruudiig neg listend oruulaw
chapter_list = [chapter for chapter in article_chapter]
#neg listend baigaa chapteruudiing '' neg string bolgow
big_chapter_string = ' '.join(chapter_list)

# nltk.download('punkt')
# nltk.download('stopwords')

tokens = word_tokenize(big_chapter_string)

# 다 문자인지 확인 다음 소문자로 변경
words = [word.lower() for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))

# end stopwords zone

print("model_w2v loading...")
# print("model_w2v commented....")
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)

#load data
with open("/var/www/chimedAi/dataset/class_idx.json", 'r') as json_file:
    class_idx = json.load(json_file)

# with open("/var/www/chimedAi/dataset/*.json", "r") as json_file2:
#     x = json.load(json_file2)

choosed_chapters = pd.read_csv("/var/www/chimedAi/dataset/choosed_cluster.csv")
matrix_data = pd.read_csv('/var/www/chimedAi/dataset/matrix_data.csv')
citations = pd.read_csv("/var/www/chimedAi/dataset/citations.csv")
mainData=pd.read_csv('/var/www/chimedAi/dataset/uurchilj_ashiglasan_chapter.csv')
matrix_data_gnn_last_line = pd.read_csv("/var/www/chimedAi/dataset/matrix_data_gnn_last_line.csv")

with open("/var/www/chimedAi/dataset/clusterWithTopFeatures.json", "r") as clusterwith:
    clusterWithTopFeatures = json.load(clusterwith)

with open("/var/www/chimedAi/dataset/cluster_details.json", "r") as cluster_details_jsn:
    cluster_detailsJson = json.load(cluster_details_jsn)

with open("/var/www/chimedAi/dataset/test_gnn_result_display_class.json", "r") as file1:
  test_gnn_result_display_class = json.load(file1)

top_features2 = pd.read_csv("/var/www/chimedAi/dataset/top_features2.csv")

#load data end

# class_values
class_values = sorted(saved_category)

#groups--g sort hiij gargaj ireed indexiig ni awna
class_idx = {name:id for id, name in enumerate(class_values)}

coun_vect_2 = CountVectorizer()
class CountArray:
    def __init__(self, topFeatures2):
        self.topFeatures2 = topFeatures2
        
    def max_df(self):
        # max_df 이상의 나타나는 단어를 무시
        # max_df=0.85
        count_matrix = coun_vect_2.fit_transform(self.topFeatures2)
        count_array = count_matrix.toarray()
        return count_array

#자연어처리로 질문과 동영상 이름을 
class RemovePunctuation:
    def __init__(self, text):
        self.text = text

    #instance
    def remove(self):
        #+ 여러개의 character을 한 공백으로 하기
        cleaned_sentence = re.sub(f'[{re.escape(string.punctuation)}]+', ' ', self.text)
        #문자열에서 양 끝에 있는 공백을 제거하는 작업
        cleaned_sentence = cleaned_sentence.strip()
        return cleaned_sentence

# ENE bichig barimt bvgdiig barimt heweer ni vldeeh heregtei
def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    # doc.sort()
    return doc

class EuclideanDistances:
    def __init__(self, feature, max_features):
        self.feature = feature
        self.max_features = max_features

    def euclidean(self):
        tfidfvectoriser=TfidfVectorizer(max_features=self.max_features)
        tfidfvectoriser.fit(self.feature)
        tfidf_vectors=tfidfvectoriser.transform(self.feature)

        tfidf_vectors=tfidf_vectors.toarray()
        pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
        pairwise_differences=euclidean_distances(tfidf_vectors)
        return pairwise_differences

def clustering():

    print("loading GoogleNews-vectors-negative300...")
    dataset = pd.read_csv("/var/www/chimedAi/dataset/uurchilj_ashiglasan_chapter.csv")

    print("+++++++++++++++++++++++++")
    print("dataset: ",dataset.shape)
    print("+++++++++++++++++++++++++")

    information_dict = {}

    dataset = dataset[['videoname','text(summary)','gist(chapter)']]
    dataset = dataset.replace(' | ',' ')

    article_chapter = dataset['text(summary)']

    #niit chapteruudiig neg listend oruulaw
    chapter_list = [chapter for chapter in article_chapter]
    category_list = [category for category in dataset['gist(chapter)']]

    #neg listend baigaa chapteruudiing '' neg string bolgow
    big_chapter_string = ' '.join(chapter_list)

    # nltk.download('punkt')
    # nltk.download('stopwords')

    tokens = word_tokenize(big_chapter_string)

    # 다 문자인지 확인 다음 소문자로 변경
    words = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    # words = [word for word in words if not word in stop_words]

    # # ENE bichig barimt bvgdiig barimt heweer ni vldeeh heregtei
    # def preprocess(text):
    #     text = text.lower()
    #     doc = word_tokenize(text)
    #     doc = [word for word in doc if word not in stop_words]
    #     doc = [word for word in doc if word.isalpha()]
    #     # doc.sort()
    #     return doc

    # corpusiig uridchilan bolowsruulna
    #자연어 처리 chapter tus buriig tus tusiin listend hiigeed 자연어 처리
    corpus = [preprocess(chapter) for chapter in chapter_list]
    category_corpus = [preprocess(cate) for cate in category_list]

    stop_words2 = list(stopwords.words('english'))
    dataset["chapter"] = [' '.join(i) for i in corpus]
    dataset["category"] = [' '.join(i) for i in category_corpus]
    #dataset을 Vectorizer
    tfidf_vect = TfidfVectorizer(tokenizer=preprocess,
                            stop_words='english', ngram_range=(1,2),
                            min_df=0.005, max_df=0.95)
    # fit_transform으로 위에서 구축한 도구로 텍스트 벡터화

    ftr_vect = tfidf_vect.fit_transform(dataset['chapter'])

    sil_score_df = pd.DataFrame()
    sil_score_list = []

    # information_dict[k] = {}
    # n_clusters: 군집화할 수, 즉 군집 중심점의 개수를 의미합니다.
    # init: 초기에 군집 중심점의 좌표를 설정할 방식을 말하며 보통은 임의로 중심을 설정하지 않고 일반적으로 k-means++방식으로 최초 설정합니다.
    # n_init: 서로 다른 군집 중심점(centroid)을 최초 셋팅한다.
    # max_iter: 최대 반복 횟수, 이 횟수 이전 모든 데이터의 중심점 이동이 없으면 종료
    k = 50
    # k = 10
    kclusterer = KMeans(n_clusters=k, init='k-means++', n_init=k, copy_x=True, max_iter=300,tol=0.0001,algorithm='auto')
    cluster_prepict = kclusterer.fit_predict(ftr_vect)
    cluster_labels = np.unique(cluster_prepict)
    n_clusters = cluster_labels.shape[0]
    cluster_centers = kclusterer.cluster_centers_
    sil_samples = silhouette_samples(ftr_vect, cluster_prepict, metric='euclidean')
    sil_score = silhouette_score(ftr_vect, cluster_prepict, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    #centroid_diameter_distance arga
    def centroid_diameter_distance(K):
        center = np.mean(K, axis=0)
        res = 2*np.mean([np.linalg.norm(x-center) for x in K])
        return res

    cluster_means = {}
    for i, cluster_n in enumerate(cluster_labels):
        c_silhouette_value = sil_samples[cluster_prepict == cluster_n]
        cluster_means[cluster_n] = centroid_diameter_distance(c_silhouette_value)
        print(cluster_n, centroid_diameter_distance(c_silhouette_value))
        c_silhouette_value.sort()
        silhouette_avg = np.mean(c_silhouette_value)

    # sil_score_df = pd.DataFrame({'cluster':cluster_label, 'sil_score': sil_score_list})
    dataset['cluster_label'] = cluster_prepict
    dataset.sort_values(by=['cluster_label'])

    #threshoold use... 임계값을 활용하여 클러스터를 호출하기
    choosed_cluster = []
    del_cluster = []

    for cluster_n, similarity in cluster_means.items():
        if similarity > 0.03:
            print("choosed clusters: ",cluster_n)
            choosed_cluster.append(cluster_n)
        else:
            # print("deleted clusters: ", cluster_n)
            del_cluster.append(cluster_n)

    #임계값보다 높은 클러스터의 챕터들만 추출
    choosed_chapters = dataset[dataset['cluster_label'].isin(choosed_cluster)]

    choosed_chapters.to_csv("/var/www/chimedAi/dataset/choosed_cluster.csv",index=False)
    print("finished clustering!")
    return choosed_chapters, tfidf_vect, kclusterer , choosed_cluster, saved_category



#추천 챕터 선정을 위한 코사인 유사도 부분- gnn과 클러스터링 둘 같이 같이 사용 중.

adjacency_list, source_2, target_2, weights_2 = {}, [], [], []
class CosineSimilarity:
    def __init__(self, features_df, category):
        self.category = category
        self.features_df = features_df
    
    #work
    def search(self):
        pd.set_option('display.max_colwidth',0)
        pd.set_option('display.max_columns', 0)

        if self.category == "first":
            features = self.features_df["features"]
            other = self.features_df["clusters"].name
        else:
            features = self.features_df["chapter"]
            other = self.features_df["text(summary)"].name

        tfidfvectoriser=TfidfVectorizer(max_features=250)
        tfidfvectoriser.fit(features)

        tfidf_vectors=tfidfvectoriser.transform(features)
        tfidf_vectors=tfidf_vectors.toarray()

        try:
            words=tfidfvectoriser.get_feature_names()
        except:
            words = tfidfvectoriser.vocabulary_
            words = sorted(words.keys(), key = words.get)

        pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
        # pairwise_differences=euclidean_distances(tfidf_vectors)
        return pairwise_similarities, features, other

    def most_similar(self, doc_id, similarity_matrix, features, other):
        # if self.features is None or self.other is None:
        #     raise ValueError("Feaures and other must be set. run search() method first.")

        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
        chapter_ls, similarity_ls, cluster_of_numbers, id_lst, videoname_list = [], [], [], [], []

        for ix in similar_ix:
            if ix==doc_id:
                continue
            id_lst.append(ix)
            chapter_ls.append(self.features_df.iloc[ix][features.name])
            similarity_ls.append(similarity_matrix[doc_id][ix])
            cluster_of_numbers.append(self.features_df.iloc[ix][other])
            # videoname_list.append(documents_df.iloc[ix][videoname_s.name])
        
        output = pd.DataFrame({'id':id_lst, 'chapters:':chapter_ls,'similarity': similarity_ls,f'{other}': cluster_of_numbers})
        return output

def chapter_similarity_check(cluster_number):

    hhe = choosed_chapters[choosed_chapters["cluster_label"].isin(cluster_number)]
    send_chapter = hhe[["cluster_label","chapter","text(summary)"]]
    send_chapter.loc[-1] = list(features_df.loc[0])+["nan"]

    send_chapter.index = send_chapter.index + 1
    send_chapter = send_chapter.sort_index()
    
    #work을 실행시키는 부분
    cosineSimilaritySearch = CosineSimilarity(send_chapter,"chapter")
    pairwise_similarities, features_val, other_val = cosineSimilaritySearch.search()
    
    #work 속에 있는 most_similar을 실행. work 속의 search mothod의 값을 사용.
    work_chapter = cosineSimilaritySearch.most_similar(0, pairwise_similarities, features_val, other_val)

    recommendation_chapters = pd.DataFrame(work_chapter.loc[work_chapter["similarity"]>0,["text(summary)","similarity"]])
    recommendation_chapters = recommendation_chapters.rename(columns={"text(summary)":"chapters"})

    #챕터 중복을 없애기
    recommendation_chapters_unique = recommendation_chapters.drop_duplicates(subset="chapters")

    #각 그룹에서 similarity의 최대 값을 선택
    recommendation_chapters_final = recommendation_chapters_unique.groupby("chapters").max().reset_index()

    #similarity를 정렬 최대에서 최소값으로
    recommendation_chapters_sort = recommendation_chapters_final.sort_values(by = ["similarity"], ascending=False)

    # print([type(int(i)) for i in id])
    return recommendation_chapters_sort

    

#clusteriin search heseg... gnn ashiglaagvi vy
def search_cluster(ques):
    global source_2
    global target_2
    global weights_2
    top_features2 = pd.read_csv("/var/www/chimedAi/dataset/top_features2.csv")

    # clusterWithTopFeatures.keys()
    # ques = "what is machine learning and big data"
    question = preprocess(ques)

    cluster_with_chapters = [{clusters:cluster_detailsJson[clusters]["chapter"]} for clusters in list(cluster_detailsJson.keys())]

    cluster_with_chapters = pd.DataFrame([(k, ",".join(v)) for d in cluster_with_chapters for k, v in d.items()], columns=["cluster", "chapter"])
    top_features2 = list(top_features2.top_features2)

    concted_features_question = question+top_features2

    # max_df 이상의 나타나는 단어를 무시
    # max_df=0.85
    # coun_vect_2 = CountVectorizer()
    # count_matrix_2 = coun_vect_2.fit_transform(concted_features_question)
    # count_array_2 = count_matrix_2.toarray()

    count_array_2 = CountArray(concted_features_question)
    count_array_2 = count_array_2.max_df()
    data_matrix_2 = csr_matrix(count_array_2)
    data_dict_2 = data_matrix_2.todok()

    for edge, weight in data_dict_2.items():
        #row, neg ugeer clusteruud bairshij bui indx
        source_2.append(edge[0])
        #feaures
        target_2.append(edge[1])
        weights_2.append(weight)


    citations_2 = pd.DataFrame({'source':source_2,'target':target_2,'we':weights_2})
    citations_2.to_csv("/var/www/chimedAi/dataset/citations_2.csv",index=False)


    #not question /GNN-d ashiglah asuult baihgui dataframe uusgej bui heseg/
    # noquestion_coun_vect = CountVectorizer()
    # noquestion_count_matrix_2 = noquestion_coun_vect.fit_transform(top_features2)
    # noquestion_count_array_2 = noquestion_count_matrix_2.toarray()
#START
    # noquestion_count_array_2 = CountArray(top_features2)
    # noquestion_count_array_2 = noquestion_count_array_2.max_df()
    # noquestion_data_matrix_2 = csr_matrix(noquestion_count_array_2)
    # no_question_data_dict_2 = noquestion_data_matrix_2.todok()

    # noquestion_source_2 = []
    # noquestion_target_2 = []
    # noquestion_weights_2 = []

    # for edge, weight in no_question_data_dict_2.items():
    #     #row, neg ugeer clusteruud bairshij bui indx
    #     noquestion_source_2.append(edge[0])

    #     #feaures
    #     noquestion_target_2.append(edge[1])
    #     noquestion_weights_2.append(weight)

    # noquestion_citations = pd.DataFrame({"source":noquestion_source_2, "target":noquestion_target_2, "we":noquestion_weights_2})
    # noquestion_citations.to_csv("/var/www/chimedAi/dataset/noquestion_citations.csv",index=False)

#END

    #해당 단어가 클러스터 내 있을 경우 1 없을 경우 0
    # matrix_df_1 = pd.DataFrame(data=count_array_2, columns=coun_vect_2.get_feature_names())

    try:
        matrix_df_2 = pd.DataFrame(data=count_array_2, columns=coun_vect_2.get_feature_names())
        # matrix_df_noquestion = pd.DataFrame(data=noquestion_count_array_2, columns=noquestion_coun_vect.get_feature_names())

    except:
        vocabulary = coun_vect_2.vocabulary_
        feature_names = sorted(vocabulary.keys(), key=vocabulary.get)
        matrix_df_2 = pd.DataFrame(data=count_array_2, columns=feature_names)

#start
        # noqustion_vocabulary = noquestion_coun_vect.vocabulary_
        # no_question_feature_names = sorted(noqustion_vocabulary.keys(), key = noqustion_vocabulary.get)
        # matrix_df_noquestion = pd.DataFrame(data=noquestion_count_array_2, columns=no_question_feature_names)
#end

    question_len = len(question)
    #클러스터 number을 해당 클러스터의 대표적인 단어만큼 추출
    clusteriin_dugaar = [cluster for cluster, features in clusterWithTopFeatures.items() for feature in features['top_features']]
    #asuult oroogvi matrix_df_noquestion dataframe-d clusteriin dugaariig nemej uguw
    
#start
    # noquestion_columnii_urt = len(matrix_df_noquestion.columns)
    # matrix_df_noquestion.insert(loc=noquestion_columnii_urt, value=clusteriin_dugaar, column="cluster_number")
#end
    #Category를 해당 클러스터에 속하는 대표적인 단어만큼 추출
    try:
        category_copy = [other["chapter_category"] for cluster, other in clusterWithTopFeatures.items() for feature in other["top_features"]]
        #asuult oroogvi matrix_df_noquestion dataframe-d category-g nemej uguw
        
        #start
        # noquestion_category_urt = len(matrix_df_noquestion.columns)
        # matrix_df_noquestion.insert(loc=noquestion_category_urt, value=category_copy, column="category")
        # matrix_df_noquestion.to_csv("/var/www/chimedAi/dataset/matrix_df_noquestion.csv",index=False)
        #end
    except:

        # clusteriin_dugaar = matrix_df.cluster_number.tolist()
        clusteriin_dugaar = [int(cluster) for cluster, features in clusterWithTopFeatures.items() for feature in features['top_features']]
        category_cluster = {cluster:random.choice(saved_category) for cluster in set(clusteriin_dugaar)}
        matrix_df_cluster = pd.DataFrame({"cluster_number": clusteriin_dugaar})
        category_copy = list(matrix_df_cluster["cluster_number"].apply(lambda key: category_cluster[key]))

        #start
        #asuult oroogvi matrix_df_noquestion dataframe-d category-g nemej uguw
        # noquestion_category_urt = len(matrix_df_noquestion.columns)
        # matrix_df_noquestion.insert(loc=noquestion_category_urt, value=category_copy, column="category")
        # matrix_df_noquestion.to_csv("/var/www/chimedAi/dataset/matrix_df_noquestion.csv",index=False)
        #end

    clusteriin_dugaar_max = int(max(clusteriin_dugaar))
    clusteriin_dugaar_ = [clusteriin_dugaar_max + 10 for _ in range(question_len)]
    clusteriin_dugaar = clusteriin_dugaar_ + clusteriin_dugaar

    category_copy = ["question" for _ in range(question_len)] + category_copy

    #feature word-iin daraagiin column nemej uguw /xamgiin ard/
    columnii_urt = len(matrix_df_2.columns)
    matrix_df_2.insert(loc=columnii_urt,value=clusteriin_dugaar, column='cluster_number')
    matrix_df_2.insert(loc=columnii_urt,value=category_copy, column='category')
    matrix_df_2.to_csv("/var/www/chimedAi/dataset/matrix_df_2.csv",index=False)

    #search deer xereglex dataframe /feature word bolon cluster number-ees burdsen/
    features_df = pd.DataFrame()
    features_df['clusters'] = list(matrix_df_2.cluster_number.unique())
    features_df['features'] = [' '.join(question)] + [' '.join(word["top_features"]) for word in clusterWithTopFeatures.values()]

    #consine similarity start

    #cosine similarity end

#start
    #test
#     def work(features_df,category):

#         documents_df = features_df.copy()

#         pd.set_option('display.max_colwidth', 0)
#         pd.set_option('display.max_columns', 0)
#         if category == "first":
#             features = documents_df['features']
#             other = documents_df['clusters'].name
#         elif category == "second":
#             features = documents_df['chapters']
#             other = documents_df['videoname'].name
#         elif category == "chapter":
#             features = documents_df["chapter"]
#             other = documents_df["text(summary)"].name
#             # videoname_s = documents_df["videoname"]
#             # other = documents_df["cluster_label"].name

#         tfidfvectoriser=TfidfVectorizer(max_features=250)
#         tfidfvectoriser.fit(features)

#         tfidf_vectors=tfidfvectoriser.transform(features)

#         tfidf_vectors=tfidf_vectors.toarray()

#         # pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
#         # pairwise_differences=euclidean_distances(tfidf_vectors)

#         # pairwise_differences = EuclideanDistances(features, 80)

#         # tokenize and pad every document to make them of the same size
#  #START
#         # tokenizer=Tokenizer()
#         # tokenizer.fit_on_texts(features)
#         # tokenized_documents=tokenizer.texts_to_sequences(features)
#         # tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=80,padding='post')
#         # vocab_size=len(tokenizer.word_index)+1

#         #     # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index.
#         # embedding_matrix=np.zeros((vocab_size,300))
#         # for word,i in tokenizer.word_index.items():
#         #     if word in model_w2v:
#         #         embedding_matrix[i]=model_w2v[word]
#         # embedding_matrix[0]
#         # # creating document-word embeddings
#         # document_word_embeddings=np.zeros((len(tokenized_paded_documents),80, 300))

#         # for i in range(len(tokenized_paded_documents)):
#         #     for j in range(len(tokenized_paded_documents[0])):
#         #         document_word_embeddings[i][j]=embedding_matrix[tokenized_paded_documents[i][j]]

#         # document_word_embeddings.shape

#         # tf-idf vectors do not keep the original sequence of words, converting them into actual word sequences from the documents

#         # document_embeddings=np.zeros((len(tokenized_paded_documents),300))
# #END       
#         try:
#             words=tfidfvectoriser.get_feature_names()
#         except:
#             words = tfidfvectoriser.vocabulary_
#             words = sorted(words.keys(), key = words.get)
# #START
#         # for i in range(len(document_word_embeddings)):
#         #     for j in range(len(words)):
#         #         document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]

#         # document_embeddings=document_embeddings/np.sum(tfidf_vectors,axis=1).reshape(-1,1)
#         # document_embeddings[np.isnan(document_embeddings)] = 0
# #END
#         pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
#         pairwise_differences=euclidean_distances(tfidf_vectors)
#end
#start
        # def most_similar(doc_id,similarity_matrix,matrix):
        #     # print (f'Search: {documents_df.iloc[doc_id][features.name]}')
        #     # print ('\n')
        #     if matrix=='Cosine Similarity':
        #         similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
        #     elif matrix=='Euclidean Distance':
        #         similar_ix=np.argsort(similarity_matrix[doc_id])
        #     chapter_ls = []
        #     similarity_ls = []
        #     cluster_of_numbers = []
        #     id_lst = []
        #     videoname_list = []
        #     for ix in similar_ix:
        #         if ix==doc_id:
        #             continue
        #         id_lst.append(ix)
        #         chapter_ls.append(documents_df.iloc[ix][features.name])
        #         similarity_ls.append(similarity_matrix[doc_id][ix])
        #         cluster_of_numbers.append(documents_df.iloc[ix][other])
        #         # videoname_list.append(documents_df.iloc[ix][videoname_s.name])
        #     if category != "chapter":
        #         output = pd.DataFrame({'id':id_lst,'similar and recommend chapters:':chapter_ls,'similarity': similarity_ls,f'{other}': cluster_of_numbers})
        #     else:
        #         output = pd.DataFrame({'videoname':id_lst,'similar and recommend chapters:':chapter_ls,'similarity': similarity_ls,f'{other}': cluster_of_numbers})

        #     # output.to_csv(f'/content/drive/MyDrive/searched_chapters/{number_}_{similarity_n}_{disnace_number}_{inp}_searched_chapters.csv',index=False)
        #     return output

        # most_similars = most_similar(0,pairwise_similarities,'Cosine Similarity',)
        # return most_similars

#end
    #recommendation cluster use feature words
    # print("질문: ", ques)

    #pairwise_similarities가 필요함으로 cosinesimilarity class에 속하는 search을 실행
    print("features_df------>",list(features_df.loc[0]))
    cosinesimilarity = CosineSimilarity(features_df, "first")
    pairwise_similarities, features_val, other_val = cosinesimilarity.search()

    most_similars = cosinesimilarity.most_similar(0, pairwise_similarities, features_val, other_val)

    linked_clusters = most_similars.loc[most_similars['similarity']>0,['chapters:','clusters']]
    clustersN = list(linked_clusters['clusters'])

    # print("관련된이 있는 클러스터의 번호: ", clustersN)
    # print("-"*120)

    # recommended_clusters = most_similars.loc[most_similars['similarity']>0,['similar and recommend chapters:','clusters']]
    # recm_clustersN = list(recommended_clusters['clusters'])

    # print("추천된 클러스터 번호: ",recm_clustersN)
    # print("\n")
    # print(most_similars)

    #여기에 chapter_similarity_check을 쓴다.
    # def chapter_similarity_check(cluster_number):
    #     hhe = choosed_chapters[choosed_chapters["cluster_label"].isin(cluster_number)]
    #     send_chapter = hhe[["cluster_label","chapter","text(summary)"]]
    #     send_chapter.loc[-1] = list(features_df.loc[0])+["nan"]
    #     send_chapter.index = send_chapter.index + 1
    #     send_chapter =send_chapter.sort_index()
        
    #     #work을 실행시키는 부분
    #     cosineSimilaritySearch = CosineSimilarity(send_chapter,"chapter")
    #     cosineSimilaritySearch = cosineSimilaritySearch.search()
        
    #     #work 속에 있는 most_similar을 실행. work 속의 search mothod의 값을 사용.
    #     cosineSimilarityMostSimilar = CosineSimilarity(0, cosineSimilaritySearch)
    #     work_chapter = cosineSimilarityMostSimilar.most_similar()

    #     recommendation_chapters = pd.DataFrame(work_chapter.loc[work_chapter["similarity"]>0,["text(summary)","similarity"]])
    #     recommendation_chapters = recommendation_chapters.rename(columns={"text(summary)":"recommended chapters"})

    #     #챕터 중복을 없애기
    #     recommendation_chapters_unique = recommendation_chapters.drop_duplicates(subset="recommended chapters")

    #     #각 그룹에서 similarity의 최대 값을 선택
    #     recommendation_chapters_final = recommendation_chapters_unique.groupby("recommended chapters").max().reset_index()

    #     #similarity를 정렬 최대에서 최소값으로
    #     recommendation_chapters_sort = recommendation_chapters_final.sort_values(by = ["similarity"], ascending=False)

    #     # print([type(int(i)) for i in id])
    #     return recommendation_chapters_sort
    #     # choosed_chapters[choosed_chapters["id"].isin(id)]

    clustersN = [int(i) for i in clustersN]
    #recommendat chapter
    #필요없는 것 같음.
    # cluster_chapter_similarity = chapter_similarity_check(clustersN)

    #원래
    # chapter_similarity = chapter_similarity_check(clustersN)
    
    #새로
    chapter_similarity = chapter_similarity_check(clustersN)
    content = []
    if chapter_similarity.similarity.empty:
        value = "클러스터가 추천할 챕터 없음."
    else:
        recommended_chapters = chapter_similarity.loc[chapter_similarity["similarity"] > 0.25]
        #ihees bagaruu sort
        recommended_chapters.sort_values(by="chapters", ascending=False)

        for chapter in list(recommended_chapters["chapters"]):
            #cluster argiin sanal bolgoson chapter
            recc_chapter = {}
            recc_chapter["chapter"] = chapter
            recc_chapter["youtube_id"] = "nKW8Ndu7Mjw"
            recc_chapter["category"] = "cluster"
            content.append(recc_chapter)


    # matrix_df_2_columns = list(set(matrix_df_2.columns) - {"cateogry"})
    # matrix_groupby = matrix_df_2[matrix_df_2_columns].groupby("cluster_number").sum()

    # test_df = matrix_groupby.T
    # global adjacency_list

    # # for i, row in test_df.iterrows():
    # #     for j, value in row.iterrows():
    # #         print(j,value)
    # #         if value >= 1:
    # #             if i not in adjacency_list:
    # #                 adjacency_list[i] = []
    # #             adjacency_list[i].append(j)
    
    #choosed_chapters.loc[choosed_chapters["text(summary)"].isin(list(recommended_chapters["recommended chapters"])), "videoname"]
    return content

#start
#gnn work을 위에서 class CosineSimilarity에서 cluster과 같이 사용한다.
# #gnn_cosine_search
# def gnn_work(features_df,category):

#   documents_df = features_df.copy()

#   pd.set_option('display.max_colwidth', 0)
#   pd.set_option('display.max_columns', 0)
#   if category == "first":
#     features = documents_df['features']
#     other = documents_df['clusters'].name
#   elif category == "second":
#     features = documents_df['chapters']
#     other = documents_df['videoname'].name
#   elif category == "chapter":
#     features = documents_df["chapter"]
#     other = documents_df["text(summary)"].name
#     # videoname_s = documents_df["videoname"]
#     # other = documents_df["cluster_label"].name

#   tfidfvectoriser=TfidfVectorizer(max_features=250)
#   tfidfvectoriser.fit(features)
#   tfidf_vectors=tfidfvectoriser.transform(features)

#   tfidf_vectors=tfidf_vectors.toarray()

# #   pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
# #   pairwise_differences=euclidean_distances(tfidf_vectors)

# #START
# #   # tokenize and pad every document to make them of the same size
# #   tokenizer=Tokenizer()
# #   tokenizer.fit_on_texts(features)
# #   tokenized_documents=tokenizer.texts_to_sequences(features)
# #   tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=250,padding='post')
# #   vocab_size=len(tokenizer.word_index)+1

# #     # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index.
# #   embedding_matrix=np.zeros((vocab_size,300))
# #   for word,i in tokenizer.word_index.items():
# #       if word in model_w2v:
# #           embedding_matrix[i]=model_w2v[word]
# #   embedding_matrix[0]
# #   # creating document-word embeddings
# #   document_word_embeddings=np.zeros((len(tokenized_paded_documents),250,300))

# #   for i in range(len(tokenized_paded_documents)):
# #       for j in range(len(tokenized_paded_documents[0])):
# #           document_word_embeddings[i][j]=embedding_matrix[tokenized_paded_documents[i][j]]

# #   document_word_embeddings.shape

# #   # tf-idf vectors do not keep the original sequence of words, converting them into actual word sequences from the documents

# #   document_embeddings=np.zeros((len(tokenized_paded_documents),300))

# #END
#   try:
#     words=tfidfvectoriser.get_feature_names()
#   except:
#     words = tfidfvectoriser.vocabulary_
#     words = sorted(words.keys(), key = words.get)

# #START
# #   for i in range(len(document_word_embeddings)):
# #       for j in range(len(words)):
# #           document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]

# #   document_embeddings=document_embeddings/np.sum(tfidf_vectors,axis=1).reshape(-1,1)
# #   document_embeddings[np.isnan(document_embeddings)] = 0
# #END

#   pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
#   pairwise_differences=euclidean_distances(tfidf_vectors)

#   def most_similar(doc_id,similarity_matrix, matrix):
#       # print (f'Search: {documents_df.iloc[doc_id][features.name]}')
#       # print ('\n')
#       if matrix=='Cosine Similarity':
#           similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
#       elif matrix=='Euclidean Distance':
#           similar_ix=np.argsort(similarity_matrix[doc_id])
#       chapter_ls = []
#       similarity_ls = []
#       cluster_of_numbers = []
#       id_lst = []
#       videoname_list = []
#       for ix in similar_ix:
#           if ix==doc_id:
#             continue
#           id_lst.append(ix)
#           chapter_ls.append(documents_df.iloc[ix][features.name])
#           similarity_ls.append(similarity_matrix[doc_id][ix])
#           cluster_of_numbers.append(documents_df.iloc[ix][other])
#           # videoname_list.append(documents_df.iloc[ix][videoname_s.name])
#       if category != "chapter":
#         output = pd.DataFrame({'id':id_lst,'similar and recommend chapters:':chapter_ls, 'similarity': similarity_ls,f'{other}': cluster_of_numbers})
#       else:
#         output = pd.DataFrame({'videoname':id_lst,'similar and recommend chapters:':chapter_ls, 'similarity': similarity_ls,f'{other}': cluster_of_numbers})

#       # output.to_csv(f'/content/drive/MyDrive/searched_chapters/{number_}_{similarity_n}_{disnace_number}_{inp}_searched_chapters.csv',index=False)
#       return output
#   most_similars = most_similar(0, pairwise_similarities,'Cosine Similarity',)
#   return most_similars

#End

#start
#gnn_cosine_search의 gnn_work을 실행
#여기에 gnn_chapter_similarity_check을 쓴다.
# def gnn_chapter_similarity_check(cluster_number, ques):
#     question_list = []
#     question_list.append(ques)

#     hhe = choosed_chapters[choosed_chapters["cluster_label"].isin(cluster_number)]
#     send_chapter = hhe[["cluster_label","chapter","text(summary)"]]
#     print("send_chapter.columns: ", send_chapter.columns)
#     #   send_chapter.loc[-1] = ["question"]+question_list+["nan"]
#     send_chapter.loc[-1] = ["question"] + question_list + ["nan"]

#     send_chapter.index = send_chapter.index + 1
#     send_chapter =send_chapter.sort_index()

#     #gnn_work
#     work_chapter = gnn_work(send_chapter,"chapter")
#     recommendation_chapters = pd.DataFrame(work_chapter.loc[work_chapter["similarity"]>0,["text(summary)","similarity"]])
#     recommendation_chapters = recommendation_chapters.rename(columns={"text(summary)":"recommended chapters"})

#     #recommended chapters 열에서 중복 제거
#     recommendation_chapters_unique = recommendation_chapters.drop_duplicates(subset = 'recommended chapters')
#     # 각 그룹에서 similarity 열의 최대값 유지
#     recommendation_chapters_final = recommendation_chapters_unique.groupby("recommended chapters").max().reset_index()
#     recommendation_chapters_sort = recommendation_chapters_final.sort_values(by=["similarity"],ascending=False)

#     return recommendation_chapters_sort
#end

def get_cluster_details(cluster_model, cluster_data, feature_names,cluster_num,top_n_features=20):

  cluster_details = {}
  center_feature_idx = cluster_model.cluster_centers_.argsort()[:,::-1]
  #zuwhun choosed xiigdsen clusteruudiin feature, videoname, cluster number 추출
  for cluster_num in cluster_num:
    cluster_details[cluster_num] = {}
    cluster_details[cluster_num]["cluster"] = cluster_num
    #top feature-uudiin id group bolgonoos 10 feature-iin id

    #sugalaj awax features-iig :10
    top_ftr_idx = center_feature_idx[cluster_num, :top_n_features]
    #id-gaar ni damjuulan feature-uudiig oloh
    top_ftr = [feature_names[idx] for idx in top_ftr_idx]

    #array-r garch ireh uchir list-ruu shiljvvleh
    top_ftr_val = cluster_model.cluster_centers_[cluster_num, top_ftr_idx].tolist()

    cluster_details[cluster_num]["top_features"] = top_ftr
    cluster_details[cluster_num]["top_featrues_value"] = top_ftr_val

    videonames = cluster_data[cluster_data["cluster_label"]== cluster_num]["videoname"]
    chapters = cluster_data[cluster_data["cluster_label"] == cluster_num]["chapter"]
    category = cluster_data[cluster_data["cluster_label"] == cluster_num]["category"]
    cluster_details[cluster_num]["chapter"] = list(chapters)

    #category
    #category bolon videonames -iin index-iig ustgaj zuwhun value-g awch uldehiin tuls listend oruulaw
    category = list(dict.fromkeys(category.values.tolist()))
    cluster_details[cluster_num]["chapter_category"] = category
    videonames = list(dict.fromkeys(videonames.values.tolist()))
    cluster_details[cluster_num]["videoname"] = videonames
  return cluster_details

clusterOfeatures = {}
clusterOfChapters = {}
def print_cluster_details(cluster_details):
  for cluster_num, cluster_detail in cluster_details.items():
    # print("cluster_detail",cluster_detail)
    print(f"클러스터 번호: {cluster_num}")
    print(f'클러스터 {cluster_num}으로 본류된 강의 동영상: \n {cluster_detail["videoname"][:5]}')
    print("deeguur bair ezelj bui 10 shirheg feature VGS: \n",cluster_detail["top_features"])
    print("categorys: ",cluster_detail["chapter_category"])
    # print(f"클러스터 {cluster_num }의 챕터: \n",cluster_detail["chapter"])
    print("\n")
    print("-"*20)

    #features
    clusterOfeatures[cluster_num] = {}
    clusterOfeatures[cluster_num]["cluster_number"] = cluster_num
    clusterOfeatures[cluster_num]["top_features"] = cluster_detail["top_features"]
    clusterOfeatures[cluster_num]["videoname"] = cluster_detail["videoname"]
    clusterOfeatures[cluster_num]["top_features_value"] = cluster_detail["top_featrues_value"]
    clusterOfeatures[cluster_num]["chapter_category"] = cluster_detail["chapter_category"]

    #chapters
    clusterOfChapters[cluster_num] = {}
    clusterOfChapters[cluster_num]["cluster_number"] = cluster_num
    clusterOfChapters[cluster_num]["chapter"] = list(cluster_detail["chapter"])

    print("print_cluster_details complete!")

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

def create_ffn(hidden_units, dropout_rate, name=None):
    print("this is create_ffn...")
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers, name=name)


def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=200, restore_best_weights=True)
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history

class GraphConvLayer(layers.Layer):
    print("this is GraphConvLayer...")
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,):
        # super(GraphConvLayer, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        print("+++++GraphConvLayer++++++")
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    #Prepare
    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    #Aggreate
    #집계
    def aggregate(self, node_indices, neighbour_messages, node_repesentations):

        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim].
        print("node_repesentations <=======>", node_repesentations)
        num_nodes = node_repesentations.shape[0]
        print("num_nodes", num_nodes)
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        print("edges[0]<==========>", node_indices)
        print("edges[1]<==========>", neighbour_indices)
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)
        print("neighbour_repesentations<=============> ",neighbour_repesentations)
        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        print("neighbour_messages:------>", neighbour_messages)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)
    #-----------------------------------------------------------------------------------------------

#0.5 0.25 0.2
class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)
        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")

        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )

        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        print("postprocess:----->", self.postprocess)
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # print("def call x: (Preprocess the node_features to produce node representations.) ----> ", x)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # print("def call x1+x: skip connection---> ", x)
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        print(input_node_indices)
        # print("node_embeddings: ------>", node_embeddings)
        # Compute logits
        return self.compute_logits(node_embeddings)

#@title 기본 제목 텍스트
#0.25, 0.5
train_data, test_data = [], []
cpy_matrix_data = matrix_data.copy()
unique_values = cpy_matrix_data.drop_duplicates(subset=["cluster_number"])["cluster_number"].tolist()
plus_matrix_data = pd.DataFrame({"cluster_number": unique_values})

for val in unique_values:
    temp = cpy_matrix_data[cpy_matrix_data["cluster_number"] == val]["category"].tolist()
    plus_matrix_data.loc[plus_matrix_data["cluster_number"] == val, 'category'] = ''.join(temp[0])


def implementing_gnn():
    global matrix_data
    global citationas
    global cpy_matrix_data

    print("loading implementing gnn...")

    if "question" in matrix_data.category.unique().tolist():
        question_index = list(matrix_data[matrix_data["category"]=="question"].index)
        #question을 matrix_data_questino에 저장
        matrix_data_question = matrix_data.loc[question_index,:]
        #question을 빼기
        matrix_data = matrix_data.loc[~matrix_data.index.isin(question_index)]

        question_category_num = matrix_data_question["category"]
        question_category_num = question_category_num.unique()[0]

    #--------------------------------------------------------------------------------------
    # class_values
    # class_values = sorted(matrix_data['category'].unique())
    class_values_pd = pd.DataFrame(data = class_values, columns=['class_values'])
    class_values_pd = class_values_pd.to_csv("/var/www/chimedAi/dataset/class_values.csv", index=False) 

    #groups--g sort hiij gargaj ireed indexiig ni awna
    class_idx =  {name: id for id, name in enumerate(class_values)}
    matrix_data_idx = {idx: name for idx, name in enumerate(sorted(matrix_data["cluster_number"]))}
    # matrix_data["cluster_number"] = matrix_data["cluster_number"].apply(lambda name: matrix_data_idx[name])

    #citationas-d source -ni id-ruu baisan bol tuhain id-deer baigaa cluster numberiig source-d oruulj uguw
    citations["source"] = citations["source"].apply(lambda value: matrix_data_idx[value])
    #groups-iig class_idx-eer solij ugnu
    matrix_data["category"] = matrix_data["category"].apply(lambda value: class_idx[value])
    
    print("shape----> ",matrix_data.shape)
    #graph
    # s = matrix_data.to_csv("/var/www/chimedAi/dataset/matrix_data2.csv",index=False)

  
    global train_data
    global test_data
    for _, cluster_data in matrix_data.groupby('category'):

        # #cluster-datag 1: hoish gej ugwul groups column garch ireh uchir 1: eer uguh yostoi
        random_selection = np.random.rand(len(cluster_data.index)) < 0.50

        # train_data.append(matrix_data_question)
        train_data.append(cluster_data[random_selection])

        #cluster-datag 1: hoish gej ugwul groups column garch ireh uchir 1: eer uguh yostoi
        test_data.append(cluster_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    print(f'train data {train_data.shape}')
    print(f'test data {test_data.shape}')

    # class_idx
    # choosed_cluster = pd.read_csv("/var/www/chimedAi/dataset/choosed_cluster.csv")

    #@title Default title text
    #column name-ees id, groups features-vvdiig hasna
    feature_names = set(matrix_data.columns) - {'cluster_number','category'}
    num_features = len(feature_names)
    num_classes = len(class_idx)

    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_data)):

        x_train = train_data.iloc[train_indices][list(feature_names)].to_numpy()
        x_train_cluster_number = train_data.iloc[train_indices]["cluster_number"]

        y_train = train_data.iloc[train_indices]["category"]
        x_val = train_data.iloc[val_indices][list(feature_names)]
        y_val = train_data.iloc[val_indices]["category"]

        x_test = test_data[list(feature_names)].to_numpy()
        x_test_indx = test_data[list(feature_names)].index
        x_test_cluster_number = test_data["cluster_number"]
        y_test = test_data["category"]
    y_test = y_test.to_numpy()

    y_train_pd = pd.DataFrame({"y_train":y_train})
    y_train_pd = y_train_pd.to_csv("/var/www/chimedAi/dataset/y_train.csv", index=False)
    x_train_cluster_number_pd = pd.DataFrame({"x_train_cluster_number": x_train_cluster_number})
    x_train_cluster_number_pd = x_train_cluster_number_pd.to_csv("/var/www/chimedAi/x_train_cluster_number.csv", index=False)
    y_test_npy = np.save("/var/www/chimedAi/dataset/y_test.npy", y_test, allow_pickle=True)

    #0.25, 0.5
    def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
        inputs = layers.Input(shape=(num_features,), name="input_features")
        x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
        print(f"this is first x {x}")
        print("\n")
        for block_idx in range(4):
            # Create an FFN block.
            x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
            print("\n")
            print(f"this is x1 {x1}")
            # Add skip connection.
            x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
            print(f"\n")
            print(f"this is x {x}")
        # Compute logits.
        logits = layers.Dense(num_classes, name="logits")(x)
        print(f"logits {logits}")
        # Create the model.
        return keras.Model(inputs=inputs, outputs=logits, name="baseline")
    baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
    baseline_model.summary()

    history2 = run_experiment(baseline_model, x_train, y_train)

    # display_learning_curves(history2)

    # verbose: 정수. 0: 자동, 1: 최신화 메시지.
    _, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
    test_accuracy = round(test_accuracy * 100, 2)
    print(f"Test accuracy: {test_accuracy}%")

    # with open("/var/www/chimedAi/dataset/clusterWithTopFeatures.json", "r") as clusterwith:
    #     clusterWithTopFeatures = json.load(clusterwith)

    node_word_list2 = list(train_data[list(feature_names)].columns)

    #new_instances에서 사용
    node_word_list2_pd = pd.DataFrame({"node_word":node_word_list2})
    node_word_list2_pd = node_word_list2_pd.to_csv("/var/www/chimedAi/dataset/instace_node_word_list2.csv", index=False)

    def compute_word_similarities(word_list1, question_list):
        similarities = []
        for question in question_list:
            try:
                if question in model_w2v.vocab and word_list1 in model_w2v.vocab:
                    similarity = model_w2v.similarity(question, word_list1)
                    similarities.append(similarity)
            except:
                if question in model_w2v.index_to_key and word_list1 in model_w2v.index_to_key:
                    similarity = model_w2v.similarity(question, word_list1)
                    similarities.append(similarity)

        return np.mean(similarities)

    def generate_instance(node_word_list, question):
        instances = []
        # for _ in range(num_classes):
        # print(node_word_list, question)
        token_similarity = [compute_word_similarities(node_words, question_words) for node_words in node_word_list]
        token_similarity = np.nan_to_num(token_similarity, nan=0)
        stat = statistics.median(token_similarity)
        # instance = (token_similarity >= np.median(token_similarity)).astype(int)
        instance = (token_similarity >= 0.3).astype(int)
        if 1 not in instance:
            instance = (token_similarity > stat).astype(int)
            ins_counter = Counter(instance)
            if ins_counter.get(1, 0) > 5:
              instance = (token_similarity >= 0.2).astype(int)

        instances.append(instance)
        return np.array(instances)

    #잠깐 commend 함
    # new_instances = generate_instance(node_word_list2, question_words)


    # logits = baseline_model.predict(new_instances)
    # probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
    # display_class_probabilities(probabilities)

    # "--------------------------------gnn___________________--v--"

    # Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
    edges = citations[["source", "target"]].to_numpy().T
    # Create an edge weights array of ones.
    edge_weights = tf.ones(shape=edges.shape[1])
    # Create a node features array of shape [num_nodes, num_features].
    node_features = tf.cast(
        #changed id to groups
        matrix_data.sort_values("cluster_number")[list(feature_names)].to_numpy(), dtype=tf.dtypes.float32
    )

    
    #new_instances ni cluster_id bolon cluster_numbers -iig xassan array

    matrix_data_gnn_last_line = matrix_data.to_csv("/var/www/chimedAi/dataset/matrix_data_gnn_last_line.csv", index=False)

    # 배열을 파일에 저장
    with open('/var/www/chimedAi/dataset/node_features.pkl', 'wb') as f:
        pickle.dump(node_features, f)

    num_classes_pd = pd.DataFrame(data=[num_classes], columns = ["num_classes"])
    num_classes_pd = num_classes_pd.to_csv("/var/www/chimedAi/dataset/num_classes.csv",index=False)


    # feature_index_pd = pd.DataFrame({"feature_index": feature_index})
    # feature_index_pd = feature_index_pd.to_csv("/var/www/chimedAi/dataset/feature_index.csv",index=False)

    # cluster_index_pd = pd.DataFrame({"cluster_index": cluster_index})
    # cluster_index_pd = cluster_index_pd.to_csv("/var/www/chimedAi/dataset/cluster_index.csv", index=False)
    # print("+"*100) 
    # print("this is num_nodes, ",num_nodes, type(num_nodes))
    # num_nodes_pd = pd.DataFrame(data=[num_nodes],columns = ["num_nodes"])
    # num_nodes_pd = num_nodes_pd.to_csv("/var/www/chimedAi/dataset/num_nodes.csv")

    edge_weights = np.save("/var/www/chimedAi/dataset/edge_model_weight.npy", edge_weights, allow_pickle=True)

    edges_npy = np.save("/var/www/chimedAi/dataset/edges.npy", edges, allow_pickle=True)

    # with open('/var/www/chimedAi/dataset/gnn_model.pkl', 'wb') as f:
    #  pickle.dump([gnn_model.edge_weights, gnn_model.edges, gnn_model.node_features], f)
    with open("/var/www/chimedAi/dataset/class_idx.json", 'w') as json_file:
        json.dump(class_idx, json_file)

    test_data["index"] = test_data.index
    test_data_save = test_data.to_csv("/var/www/chimedAi/dataset/test_data.csv",index=False)
    x_test_indx_pd = pd.DataFrame({"x_test_index": x_test_indx})
    x_test_indx_pd = x_test_indx_pd.to_csv("/var/www/chimedAi/dataset/x_test_indx.csv", index=False)
    print("finish implement gnn!")

#gnn_with_cosine_search과 cluster_with_cosine_searchiig ajillulahiin tuld comment bolgow
#20231114_starbucks

# test_gnn_result_display_class = {}
# def predict(ques):
#     question_words = preprocess(ques)
#     print("question_words: ", question_words)

#     matrix_data = pd.read_csv("/var/www/chimedAi/dataset/matrix_data_gnn_last_line.csv")

#     with open('/var/www/chimedAi/dataset/gnn_model.pkl', 'rb') as f:
#         gnn_mo = pickle.load(f)
#     edges = gnn_mo[1]

#     node_word_list2 = pd.read_csv("/var/www/chimedAi/dataset/instace_node_word_list2.csv")

#     def compute_word_similarities(word_list1, question_list):
#         similarities = []
#         for question in question_list:
#             try:
#                 if question in model_w2v.vocab and word_list1 in model_w2v.vocab:
#                     similarity = model_w2v.similarity(question, word_list1)
#                     similarities.append(similarity)
#             except:
#                 if question in model_w2v.index_to_key and word_list1 in model_w2v.index_to_key:
#                     similarity = model_w2v.similarity(question, word_list1)
#                     similarities.append(similarity)

#         return np.mean(similarities)

#     def generate_instance(node_word_list, question):
#         instances = []
#         # for _ in range(num_classes):
#         # print(node_word_list, question)
#         token_similarity = [compute_word_similarities(node_words, question_words) for node_words in node_word_list]
#         token_similarity = np.nan_to_num(token_similarity, nan=0)
#         stat = statistics.median(token_similarity)
#         # instance = (token_similarity >= np.median(token_similarity)).astype(int)
#         instance = (token_similarity >= 0.3).astype(int)
#         if 1 not in instance:
#             instance = (token_similarity > stat).astype(int)
#             ins_counter = Counter(instance)
#             if ins_counter.get(1, 0) > 5:
#                 instance = (token_similarity >= 0.2).astype(int)

#         instances.append(instance)
#         return np.array(instances)

#     new_instances = generate_instance(node_word_list2["node_word"], question_words)

#     # 배열을 파일에서 불러오기
#     with open('/var/www/chimedAi/dataset/node_features.pkl', 'rb') as f:
#         node_features = pickle.load(f)

#     num_classes = pd.read_csv("/var/www/chimedAi/dataset/num_classes.csv")
#     num_classes = list(num_classes["num_classes"])[0]


#     # feature_index = list(pd.read_csv("/var/www/chimedAi/dataset/feature_index.csv")['feature_index'])

#     # cluster_index = list(pd.read_csv("/var/www/chimedAi/dataset/cluster_index.csv")['cluster_index'])

#     num_nodes = list(pd.read_csv("/var/www/chimedAi/dataset/num_nodes.csv")['num_nodes'])[0]
#     class_values = pd.read_csv("/var/www/chimedAi/dataset/class_values.csv")["class_values"].tolist()
#     def display_class_probabilities(probabilities):
#         test_list = []
#         for instance_idx, probs in enumerate(probabilities):
#             main = {}
#             print(f"Instance {instance_idx + 1}:")
#             for class_idx, prob in enumerate(probs):
#                 cluster_num = plus_matrix_data.loc[plus_matrix_data["category"] == class_values[class_idx], "cluster_number"]
#                 index = cluster_num.index
#                 value = index.values[0]
#                 cluster_num = cluster_num[value]
#                 main[cluster_num] = {}
#                 main[cluster_num][class_values[class_idx]] = round(prob, 2)
#                 print(f"- {class_values[class_idx]}: {round(prob, 2)}%")
#             test_list.append(main)
#         return test_list

#     # num_nodes = node_features.shape[0]
#     new_node_features = np.concatenate([node_features, new_instances])
#     new_node_indices = [i + num_nodes for i in range(num_classes)]

#     def compute_word_similarities_gnn(word_list1, question_list):
#         similarities = []
#         for question in question_list:
#             try:
#                 if question in model_w2v.vocab and word_list1 in model_w2v.vocab:
#                     similarity = model_w2v.similarity(question, word_list1)
#                     similarities.append(similarity)
#             except:
#                 if question in model_w2v.index_to_key and word_list1 in model_w2v.index_to_key:
#                     similarity = model_w2v.similarity(question, word_list1)
#                     similarities.append(similarity)
#         return np.mean(similarities)


#     # new_citations = []
#     def instance_2(node_word_list, question):
#         token_similarity = [compute_word_similarities_gnn(node_words, question_words) for node_words in node_word_list]
#         token_similarity = np.nan_to_num(token_similarity, nan = 0)
#         instance = (token_similarity >= 0.3).astype(int)
#         stat = statistics.median(token_similarity)
#         if 1 not in instance:
#             instance = (token_similarity > 0.1).astype(int)
#             if 1 not in instance:
#                 instance = (token_similarity > stat).astype(int)
#         print(instance)
#         instance_indx = [i for i in range(len(instance)) if instance[i] == 1]
#         new_citations = [[new_node_indices[0],indx] for indx in instance_indx]
#         return new_citations


#     new_citations = instance_2(node_word_list2, question_words)
#     new_citations = np.array(new_citations).T

#     if len(new_citations) > 0:
#         print(f"{len(new_citations)}>{0}")
#         new_edges = np.concatenate([edges, new_citations], axis=1)
#     else:
#         print(f"{len(new_citations)}=={0}")
#         new_edges = edges

#     # Create graph info tuple with node_features, edges, and edge_weights.

#     edge_weights = np.load("/var/www/chimedAi/dataset/edge_model_weight.npy")
    
#     graph_info = (node_features, edges, edge_weights)
#     print("Edges shape:", edges.shape)
#     print("Nodes shape:", node_features.shape)

#     gnn_model = GNNNodeClassifier(
#         graph_info=graph_info,
#         num_classes=num_classes,
#         hidden_units=hidden_units,
#         dropout_rate=dropout_rate,
#         name="gnn_model",)

#     matrix_data_shape = matrix_data.shape[0] - 1
#     print("GNN output shape: ", gnn_model([1,2,matrix_data_shape]))
#     gnn_model([1,2,matrix_data_shape])
#     gnn_model.summary()

#     # +++++++++++ tvr zuur comment hiiw ++++++++++++
#     x_train_cluster_number_pd = pd.read_csv("/var/www/chimedAi/x_train_cluster_number.csv")["x_train_cluster_number"]
#     x_train = x_train_cluster_number_pd


#     y_train = pd.read_csv("/var/www/chimedAi/dataset/y_train.csv")["y_train"]
#     history = run_experiment(gnn_model, x_train, y_train)
    
#     # display_learning_curves(history)

#     #GNN 모델 평가
#     test_data = pd.read_csv("/var/www/chimedAi/dataset/test_data.csv")
#     test_data = test_data.set_index("index")

#     x_test_indx = pd.read_csv("/var/www/chimedAi/dataset/x_test_indx.csv")
#     x_test_index = list(x_test_indx["x_test_index"])

#     x_test = test_data.loc[x_test_index]["cluster_number"].to_numpy()

#     y_test = np.load("/var/www/chimedAi/dataset/y_test.npy")

#     _, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
#     print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
#     # #클러스터링 방법을 사용 후

#     #feature index
#     feature_index = gnn_model.edges[0]
#     #index->clusteruudiin bairshij bui
#     cluster_index = gnn_model.edges[1]
#     # num_nodes = node_features.shape[0]
#     num_nodes = len(matrix_data.cluster_number.unique())

#     gnn_model.node_features = new_node_features
#     gnn_model.edges = new_edges
#     gnn_model.edge_weights = tf.ones(shape=new_edges.shape[1])

#     logits = gnn_model.predict(tf.convert_to_tensor(new_node_indices))
#     probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
#     test_list = display_class_probabilities(probabilities)

#     global edge_df
#     edge_df = pd.DataFrame(gnn_model.edges).T
#     # edges_df = edges_df.rename(columns={0:"clusters",1:"features"})
#     edge_df["weights"] = gnn_model.edge_weights

#     adj_list = {}
#     for i, row in edge_df.iterrows():
#         src = row[0]
#         trgt = row[1]
#         if trgt not in adj_list:
#             adj_list[trgt] = []
#         if src not in adj_list:
#             adj_list[src] = []
#     adj_list[trgt].append(src)

#     edges_gnn_ = {}
#     edges_gnn_test = Counter([(int(node[0]), int(node[1])) for nodes in adj_list.values() for node in combinations(nodes, 2)])
#     #adjacency_list ni deer search hesegt vvsej baigaa list
#     # cluster_edge_wei = Counter([node for nodes in adjacency_list.values() for node in combinations(nodes, 2)])

#     for edge,weight in edges_gnn_test.items():
#         if edge[0] != edge[1]:
#             edges_gnn_[edge] = {"weight": weight}

#     edge_key_gnn = [edges for edges in edges_gnn_.keys()]
#     #clusteriin edges
#     # edges_cluster = [edges for edges in cluster_edge_wei.keys()]

#     edge_weight = edges_gnn_.values()
#     #clusteriin edgesees weight-iig salgaj awah
#     # weights = list(cluster_edge_wei.values())

#     graph = {nodes: weights for nodes, weights in edges_gnn_.items() if weights["weight"] >= 2}

#     def find_duplicate_nodes(graph):
#         nodes = graph.keys()
#         node0 = [node[0] for node in nodes]
#         duplicate_node = Counter(node0)
#         find_duplicated_node = [node for node, count in duplicate_node.items() if count > 1]
#         return find_duplicated_node

#     def find_weight_node_remove(graph):
#         find_nodes = find_duplicate_nodes(graph)
#         find_nodes_dict = {}
#         for nodes, weights in graph.items():
#             for node in find_nodes:
#                 # print(type(node))
#                 if nodes[0] == node:
#                     weight = list(weights.values())[0]
#                     find_nodes_dict[nodes] = {"weight":weight}
#         return find_nodes_dict

#     def add_new_edge(graph):
#         dic = {}
#         #olon node-tei holbogdson node-iig awch ireh
#         finded_nodes = find_weight_node_remove(graph)
#         #min weight-iig ustgahiin tuld weightuudiig neg listend
#         for nodes, weight in finded_nodes.items():
#             dic.setdefault(nodes[0], []).append(list(weight.values())[0])

#         #weight dund hamgiin baga weightiig ustgah
#         for key in dic:
#             dic[key].remove(min(dic[key]))

#         groupd_edges = {}
#         weight_val = [values for weight_values in dic.values() for values in weight_values]
#         max_weights = {nodes:weights["weight"] for nodes, weights in finded_nodes.items() for value in weight_val if weights["weight"] == value}

#         for edge, weigths in max_weights.items():
#             test_dictionary = {}
#             groupd_edges.setdefault(edge[0], []).append(edge[1])

#         new_edges = {edge:{"weight":1} for edges in groupd_edges.values() for edge in combinations(edges, 2)}
#         print("new_edges:")
#         return new_edges.keys()

#     add_new_edges = add_new_edge(graph)

#     global test_gnn_result_display_class
#     for dic in test_list:
#         for key in dic:
#             for val in dic[key]:
#                 test_gnn_result_display_class.setdefault(val, []).append(dic[key][val])

#     with open("/var/www/chimedAi/dataset/test_gnn_result_display_class.json", "w") as display_class:
#         json.dump(test_gnn_result_display_class, display_class)

 
def Search_recommendation(request):
    global class_idx
    global matrix_data_gnn_last_line
    if request.method == 'POST':
        question = request.POST.get("search")
        print(type(question))

        print("clusteriin search...")        
        #clusteriin search
        #질문에 속한 string.punctuantion character들을 제거 whit-me-> whit me
        clearCharacter = RemovePunctuation(question)
        clearedQuestoin = clearCharacter.remove()
        
        content = search_cluster(clearedQuestoin)
        
        #gnn
        # implementing_gnn()

        # print("new instance, new edges, predict...")
        #gnn의 cosine simillarity 실행시키기 전에 꼭 실행 시켜야 하는 부분. new instance, new edge
        # predict(question)
        
        print("gnn search is working...")

        #gnn의 search
        gnn_similar_clusters = matrix_data_gnn_last_line[matrix_data_gnn_last_line["category"].isin([class_idx[classes] for classes, simil in test_gnn_result_display_class.items() if round(np.mean(simil),2) > 0])]["cluster_number"].unique().tolist()
        print(f"gnn_similar_clusters, : {gnn_similar_clusters}")
        #recc_chapters = gnn_chapter_similarity_check(gnn_similar_clusters, question)
        recc_chapter = chapter_similarity_check(gnn_similar_clusters)
        content2 = []
        if not content:
            print("클러스터가 추천한 챕터 없음.")
            if len(recc_chapters.axes[0]) <= 1:
                print("신경망이 추천한 챕터 없음.")
            else:
                for chapters in recc_chapters["recommended chapters"]:
                    chapters_dict = {}
                    chapters_dict["chapter"] = chapters
                    chapters_dict["youtube_id"] = "nKW8Ndu7Mjw"
                    chapters_dict["category"] = "gnn"
                    print(chapters_dict)
                    content2.append(chapters_dict)
        else:
            if len(recc_chapters.axes[0]) <= 1:
                print("신경망이 추천 할 챕터 없음.")
                content2.extend(content)
            else:
                for chapters in recc_chapters["recommended chapters"]:
                    chapters_dict = {}
                    chapters_dict["chapter"] = chapters
                    chapters_dict["youtube_id"] = "nKW8Ndu7Mjw"
                    chapters_dict["category"] = "gnn"
                    print(chapters_dict)
                    print("this is cluster이 추천한 챕터들: ", content)
                    content2.append(chapters_dict)
                content2.extend(content)

        return render(request, "videos/search/searched_video.html", {'chapters': content2})


def sign_up(request):
    User = get_user_model()
    if request.method == 'POST':
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        if password1 == password2:
            username =request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password1')
            check_user = User.objects.filter(username=username).first()
            check_profile = User.objects.filter(email=email).first()
            if check_user or check_profile:
                user_exist_context = {'message': 'User allready exists', 'class': 'danger'}
                return render(request,'registration/register.html',user_exist_context)   
            request.session['username'] = username         
            request.session['email'] = email         
            request.session['password1'] = password 
            print(username,email,password)        
            return redirect('../signupnextpage/')
        else:
            context_password = {'message':'Password in not correct!', 'class':'danger'}
            return render(request,'registration/register.html',context_password)
    else:
        method_context = {'message':'request method in not correct','class':'danger'}
        return render(request,'registration/register.html',method_context)


def signup_next(request):
    if request.method == 'POST':
        username =request.session.get('username')
        email = request.session.get('email')
        password = make_password(request.session.get('password1'))
        phone_number = request.POST.get('phone')
        role = request.POST.get('role')
        print(username,email,password,phone_number,role) 

        profile = Users(username=username, email=email,role=role,phone_number=phone_number, password=password, is_staff=False)
        profile.save()

        if profile.pk:
            sign_up_success = {'message':'Sucessfully saved the profile!', 'class':'success'}
        else:
            sign_up_success = {'message':'Error occurred while saving the profile','class':'danger'}
        
        return render(request, 'registration/register_2.html', sign_up_success)


def login_view(request):
    User = get_user_model()
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request,username=username, password=password)
        if user is not None:
            if user.is_active:
                role = user.role
                login(request, user)
                if role == Users.professor:
                    return redirect('professor')
                    # return render(request,'users/professor/professor.html')  # 로그인 성공 후 이동할 페이지 설정
                else:
                    # return redirect('')
                    return render(request, 'users/student/student.html')
                # login_succ = {'message':'logged', 'class':'success'}
            else:
                login_err = {'message':'Invalid username or password','class':'danger'}
                return render(request, 'login/login.html', login_err)
        else:
            login_err = {'message':'Invalid username or password','class':'danger'}
            return render(request, 'login/login.html', login_err)
            
    else:
        method_context = {'message':'request method in not correct','class':'danger'}
        return render(request, 'login/login.html', method_context)

def Prof_profile(request):
    return render(request, "users/professor/prof_profile.html")

# def Prof_channels(request):
#     return render(request, "channels/professor/pro_channels.html")

class YoutubeVideoDownloadAssemblyAi:
    def __init__(self, link):
        self.link = link

    #instance
    def download_video(self, output_path = '/var/www/chimedAi/media/videos/'):
        try:
            url = self.link
            video = YouTube(url)
            stream_video = video.streams.get_lowest_resolution()
            video_path = stream_video.download(output_path)

        except Exception as e:
            print("video is not downloaded: ", e)
        return video_path
    
    @staticmethod
    def assemblyai(video_path):
        aai.settings.api_key = "a217a50eacaf41f981b04c2edb36daa2"
        transcript_response = aai.TranscriptionConfig(
            sentiment_analysis=True,
            auto_chapters = True
        )

        transcriber = aai.Transcriber(config=transcript_response)
        transcript = transcriber.transcribe(video_path)

        return transcript

title = ''
#video를 웹에 업로드하는 코드. 및 챕터 생성 start
def Insert_video(request):
    global mainData

    if request.method == 'POST':
        video_title = request.POST.get('title')
        # video = request.FILES.get('link')
        # file_content = video.read()

        video_link = request.POST.get('link')
        #youtube 링크에서 아이디를 구별하기
        parsed_url = urlparse(video_link)
        youtube_id = parse_qs(parsed_url.query).get('v')[0]
        # create_video = Video.objects.create(title = video_title, youtube_id=youtube_id)
        if youtube_id is not None:
        # if create_video.youtube_id is not None:
            # print("create_video.video_l is not None--->", create_video.youtube_id)
            method_context = {'message': 'created your chapter!', 'class': 'success'}

            downloadVideo = YoutubeVideoDownloadAssemblyAi(video_link)
            downloadedVideoPath = downloadVideo.download_video()

            assemblyai_json = YoutubeVideoDownloadAssemblyAi.assemblyai(downloadedVideoPath)

            #video-nii nerendeh temdegtvvd hooson zai bolj oorchlogdsonoos olj chadahgvi baih asuudal vvsew.
            #vvniig shiidehiin tuld automat-r nerendeh temdegtvvdiig 저게

            status = ''
            auto_chapters = ''

            chapters_json1 = {}
            status = assemblyai_json.status
            print(f"status {status}")
 
            if assemblyai_json.status == "completed":
                if assemblyai_json.chapters:
                    for chapters in assemblyai_json.chapters:
                        chapters_json2 = {}
                        for chapter in chapters:
                            chapters_json2[chapter[0]] = chapter[1]
                        if "summary" and "gist" in chapters_json2.keys():
                            chapters_json2["text(summary)"] = chapters_json2.pop("summary")
                            chapters_json2["gist(chapter)"] = chapters_json2.pop("gist")
                        else:
                            print("colums fine!")
                        for key in chapters_json2.keys():
                            print(key)
                            if key in chapters_json1.keys():
                                chapters_json1[key].append(chapters_json2[key])
                            else:
                                chapters_json1[key] = [chapters_json2[key]]
                                chapters_json1["videoname"] = video_title
                                chapters_json1["category"] = video_title
                                chapters_json1["type"] = "lecture"
                else:
                    print("no chapters!")
            else:
                status = assemblyai_json.status
                print(f"status {status}")
            
            print("created new chapters")
            chapters_json1_pd = pd.DataFrame(chapters_json1)
            print(chapters_json1_pd)

            # mainData = pd.concat([mainData, chapters_json1_pd], ignore_index=True)
    #         mainData=mainData.to_csv('/var/www/chimedAi/dataset/uurchilj_ashiglasan_chapter.csv')

    #         #start clustering
    #         choosed_chapters , tfidf_vect, kclusterer, choosed_cluster, saved_category = clustering()

    #         #각 클라스터에서 대표적인 단어를 후출하기
    #         vocab = tfidf_vect.vocabulary_
    #         feature_names = sorted(vocab.keys(), key= lambda x: vocab[x])
    #         # feature_names = tfidf_vect.get_feature_names()
    #         cluster_details = get_cluster_details(cluster_model=kclusterer,
    #                                             cluster_data=choosed_chapters,
    #                                             feature_names=feature_names,
    #                                             cluster_num=choosed_cluster,
    #                                             top_n_features=50)

    #         print_cluster_details(cluster_details)

    #         my_dict = {str(key):value for key, value in cluster_details.items()}

    #         with open("/var/www/chimedAi/dataset/cluster_details.json", "w",encoding="utf-8") as cluster_det:
    #             json.dump(my_dict, cluster_det, ensure_ascii=False, indent=4,default=str)

    #         my_clusterfeature = {str(key):value for key, value in clusterOfeatures.items()}
    #         with open("/var/www/chimedAi/dataset/clusterOfeatures.json","w", encoding="utf-8") as clusterOfeat:
    #             json.dump(my_clusterfeature, clusterOfeat, ensure_ascii=False,indent=4, default=str)

    #         #listend baigaa adilhan ugsiig ustgah
    #         for i in range(10):
    #             def remove_dupli_featrues(feature):
    #                 clusterOfeatures[i]['top_features'].remove(feature)
    #                 return clusterOfeatures

    #             for i in clusterOfeatures.keys():
    #                 for feature in clusterOfeatures[i]['top_features']:
    #                     if 1 < len(feature.split(' ')):
    #                         if feature.split(' ')[0] == feature.split(' ')[1]:
    #                             remove_dupli_featrues(feature)

    #         #c a d 제거
    #         for i in range(10):
    #             def remove_short_words(word):
    #                 clusterOfeatures[i]['top_features'].remove(word)
    #                 return clusterOfeatures

    #             for i in clusterOfeatures.keys():
    #                 for feature in clusterOfeatures[i]['top_features']:
    #                     if 3 > len(feature):
    #                         remove_short_words(feature)

    #         #append
    #         top_features1 = []
    #         #extend
    #         top_features2 = []
    #         top_features_vectors = []
    #         clusterWithTopFeatures = {}
    #         category_list = []
    #         for cluster_n, cluster_details in clusterOfeatures.items():
    #             clusterWithTopFeatures[cluster_details["cluster_number"]] = {}
    #             feature = ' '.join(cluster_details["top_features"])

    #             #####bvh top feature-iig neg listend oruulj ogoh

    #             top_features1.append(feature)
    #             top_features2.extend(cluster_details['top_features'][:40])
    #             top_features_vectors.extend(cluster_details['top_features_value'])

    #             ###category vvsgeh code-iig ajilluulwal commentiig arilgah yostoi

    #             #cluster number bolon top feature hoyroos bvrdsen dictionary vvsgeh
    #             clusterWithTopFeatures[cluster_details["cluster_number"]] = {}
    #             clusterWithTopFeatures[cluster_details["cluster_number"]]["top_features"] = cluster_details['top_features'][:40]
    #             # clusterWithTopFeatures[cluster_details["cluster_number"]]["chapter_category"] = chapter_category

    #         top_features_csv = pd.DataFrame({"top_features2":top_features2})
    #         top_features_csv = top_features_csv.to_csv("/var/www/chimedAi/dataset/top_features2.csv",index=False)


    #         my_clusterWithTopFeatures = {str(key):val for key, val in clusterWithTopFeatures.items()}

    #         with open("/var/www/chimedAi/dataset/clusterWithTopFeatures.json", "w") as clusterwith:
    #             json.dump(my_clusterWithTopFeatures, clusterwith, ensure_ascii=False, indent=4, default=str)


    #         # max_df 이상의 나타나는 단어를 무시
    #         # max_df=0.85
    #         coun_vect = CountVectorizer()
    #         count_matrix = coun_vect.fit_transform(top_features2)
    #         count_array = count_matrix.toarray()

            #   count_array = CountArray(top_features2)

    #         #해당 단어가 클러스터 내 있을 경우 1 없을 경우 0
    #         count_vec_vocabulary = coun_vect.vocabulary_
    #         feature_names_coun_vect = sorted(count_vec_vocabulary.keys(), key = count_vec_vocabulary.get)
    #         matrix_df = pd.DataFrame(data=count_array, columns=feature_names_coun_vect)

    #         #클러스터 number을 해당 클러스터의 대표적인 단어만큼 추출
    #         clusteriin_dugaar = [cluster for cluster, features in clusterWithTopFeatures.items() for feature in features['top_features']]

    #         #feature word-iin daraagiin column nemej uguw /xamgiin ard/
    #         columnii_urt = len(matrix_df.columns)
    #         matrix_df.insert(loc=columnii_urt,value=clusteriin_dugaar, column='cluster_number')

    #         #Category를 해당 클러스터에 속하는 대표적인 단어만큼 추출
    #         for a, b in clusterWithTopFeatures.items():
    #             if "chapter_category" in list(b.keys())[0]:
    #                 category_copy = [other["chapter_category"] for cluster, other in clusterWithTopFeatures.items() for feature in other["top_features"]]
    #         else:
    #             #clusteriin_dugaar = [cluster for cluster, features in clusterWithTopFeatures.items() for feature in features['top_features']]
    #             clusteriin_dugaar = matrix_df.cluster_number.tolist()
    #             category_cluster = {cluster:random.choice(saved_category) for cluster in set(clusteriin_dugaar)}
    #             category_copy = matrix_df["cluster_number"].apply(lambda key: category_cluster[key])

    #         columnii_urt = len(matrix_df.columns)
    #         matrix_df.insert(loc=columnii_urt, value=category_copy, column='category')
    #         from scipy.sparse import csr_matrix

    #         #feature
    #         #data_matrix-iin index=0 ni index=0 deel bairshix cluster, index=1 ni matrix_df-iin columnii index buyu feature
    #         #즉 data_matrix index=0 ni row, index=1 ni column
    #         data_matrix = csr_matrix(count_array)

    #         # 마지막으로 dok(Dictionary of Keys)방식을 소개하겠습니다.
    #         # dok는 좌표가 key이고 원소 값이 value인 딕셔너리 구조입니다.
    #         # dok 방식은 희소행렬을 점진적으로 구축할 때 사용하기 좋습니다.

    #         #Dictionary to Keys
    #         data_dict = data_matrix.todok()

    #         matrix_data = matrix_df.copy()
    #         matrix_data.to_csv('/var/www/chimedAi/dataset/matrix_data.csv',index=False)
            
    #         target = []
    #         source = []
    #         weights = []

    #         for edge, weight in data_dict.items():
    #             #row, neg ugeer clusteruud bairshij bui indx
    #             source.append(edge[0])
    #             #feaures
    #             target.append(edge[1])
    #             weights.append(weight)

    #         citations = pd.DataFrame({'source':source,'target':target,'we':weight})
    #         test = matrix_df.copy()
    #         citations_test = citations.copy()
    #         citations.to_csv("/var/www/chimedAi/dataset/citations.csv",index=False)

    #         #gnn
    #         implementing_gnn()

    #     else:
    #         method_context = {'message': 'error!', 'class': 'danger'}
    #         print(" 1 method_context ------> ", method_context)
        
    # else:

    #     method_context = {'message': 'error!', 'class': 'danger'}
    #     print(" 2 method_context------> ", method_context)

    # return render(request, "videos/professor/new_videos.html", method_context)
    return redirect("base")



#video를 웹에 업로드하는 코드. 및 챕터 생성 end

def New_video_page(request):
    return render(request, 'videos/professor/new_videos.html')

def get_video_data():
    videos = Video.objects.all()
    video_data = []
    current_date = datetime.now(timezone.utc)
    for video in videos:
        video.days_since_upload = (current_date - video.uploaded_at).days

        video.years_since_upload = video.days_since_upload // 365
        video.months_since_upload = (video.days_since_upload % 365) // 30
        days = (video.days_since_upload % 365) % 30
        
        # print("years+++++++++++>", years)
        parsed_url = urlparse(video.youtube_id)
        youtube_id = parse_qs(parsed_url.query).get('v')[0]

        # thumbnail_url = f'https://img.youtube.com/embed/{youtube_id[0]}/{video.title}.jpg'
        # if youtube_id:
        video_data.append({
            'title': video.title,
            'youtube_id': youtube_id,
            'years': video.years_since_upload,
            'months': video.months_since_upload,
            'days': days,
        })

    return video_data

#video vzdeg heseg    
def watch(request, youtube_id):
    video_data = get_video_data()
    context = {
        #tomoor harah main bichlegiig gargaj irhiin tuld shuud id-g damjuulj bui.
        "youtube_id": youtube_id,
        #busad bichlegvvdiig for-r damjuulj gargaj ireh heseg tul listend.
        "videos": video_data
    }
    return render(request, 'videos/watch/watch.html', context)

def LogoutView(request):
    logout(request)
    return redirect('base')

def Prof_channels(request):
    video_data = get_video_data()
    return render(request, 'channels/professor/pro_channels.html', {'videos':video_data})
    
def Searched_video(request):
    return render(request, 'videos/search/searched_video.html')

def base(request):
    video_data = get_video_data()
    return render(request, 'base.html', {'videos':video_data})

def login_page(request):
    return render(request, 'login/login.html')

def register_view(request):
    return render(request, 'registration/register.html')

def sign_up_page(request):
    return render(request, 'registration/register.html')

def signup_next_page(request):
    return render(request, 'registration/register_2.html')

def professor(request):
    return redirect('users/professor/professor.html')
