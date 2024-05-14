import streamlit as st
import subprocess
import pandas as pd
import numpy as np
import string
import spacy
from spacy.lang.id.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import time
from os import path
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import KMeans
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score

nlp = spacy.blank("id")

def home_page():
    st.title("Home Page")
    st.write("Selamat datang di")

def proses_page():
    st.title("BERTTopic Page")        

    if st.button("Prosess"):      
        
        st.write("Proses Membutuhkan waktu, mohon tunggu hingga seluruh proses selesai") 

        df = pd.read_csv("detik-fiks.csv")
        df_subset = df.head(15020)        
        df_subset.to_csv("dataset.csv", index=False)
        st.write("Tahap Prepocessing : Finish") 

        
        df = pd.read_csv('dataset.csv')
        df['description_lower_case'] = df['description'].str.lower()
        df.to_csv('dataset.csv', index=False)
        st.write("Tahap Case Folding : Finish") 

        
        df = pd.read_csv('dataset.csv')
        def remove_punctuation(text):
            return text.translate(str.maketrans('', '', string.punctuation))
        df['description_clean'] = df['description_lower_case'].apply(remove_punctuation)
        df.to_csv('dataset.csv', index=False)        

        df = pd.read_csv("dataset.csv")
        nlp = spacy.blank("id")
        def remove_stopwords(text):
            doc = nlp(text)
            filtered_text = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
            return " ".join(filtered_text)
        df['description_clean'] = df['description_clean'].apply(remove_stopwords)
        df.to_csv("dataset.csv", index=False)
        st.write("Tahap Cleaning : Finish") 


        st.write("Tahap BERTopic : Start, Please Wait")         
        train = pd.read_csv('dataset.csv')
        train

        docs = train['description_clean'].to_list()
        docs[:100]

        train = pd.read_csv('dataset.csv')
        docs = train['description_clean'].to_list()
        min_topic_size = 100
        nr_topics = 9
        topic_model = BERTopic(min_topic_size=min_topic_size, nr_topics=nr_topics)
        topics, _ = topic_model.fit_transform(docs)
        topics_info = topic_model.get_topics()

        def calculate_coherence(topics_info, docs):
            coherence_values = []
            for topic_id in topics_info:
                topic_words = topics_info[topic_id]
                topic_string = ' '.join([str(word) for word in topic_words])
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(docs)
                topic_vectors = vectorizer.transform([topic_string])
                similarity_matrix = cosine_similarity(X, topic_vectors)
                coherence = similarity_matrix.mean()
                coherence_values.append(coherence)
            return coherence_values
        coherence_values = calculate_coherence(topics_info, docs)

        topic_info = topic_model.get_topic_info()
        df = pd.DataFrame(topic_info)
        df.to_csv("topic_info_all.csv",index=False)

        name_topic_info = pd.read_csv('topic_info_all.csv')
        data = {"ID Topic": [f"{i}" for i in range(len(coherence_values))],
                "Koherensi": coherence_values
                }
        df = pd.DataFrame(data)
        df.to_csv('nilai_topic_koherensi.csv', index=False)

        topic_info_all =  pd.read_csv("topic_info_all.csv")
        nilai_topic_koherensi = pd.read_csv("nilai_topic_koherensi.csv")

        merged_df = pd.merge(topic_info_all, nilai_topic_koherensi, left_on ='Topic', right_on='ID Topic',how='left')

        merged_df.drop(columns=['ID Topic'], inplace=True)
        merged_df.to_csv('topic_info_all_koherensi.csv', index=False)

        if os.path.exists("nilai_topic_koherensi.csv") :
            os.remove("nilai_topic_koherensi.csv")

        topic_info_all_koherensi =  pd.read_csv("topic_info_all_koherensi.csv")
        topic_info_all_koherensi[:10]
        st.write("Tahap BERTopic : Finish")         


        
        jumlah_topik = (len(set(topics)) - 1)
        document_embeddings = topic_model.transform(docs)[0]
        document_embeddings = np.array(document_embeddings).reshape(-1, 1)

        n_clusters = jumlah_topik
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(document_embeddings)
        silhouette_avg = silhouette_score(document_embeddings, cluster_labels)
        st.write("Silhouette Score : ")
        st.write(silhouette_avg)

        
        st.write("Informasi Topic :")  
        topic_model.get_topic_info()

        topic_info = topic_model.get_topic_info()
        df = pd.DataFrame(topic_info)
        df.to_csv("topic_info_all.csv",index=False)
        st.write("Informasi Topic disimpan di topic_info_all.csv")  

        topic_model.get_topic_info()[:10]
        topic_model.get_topic(0)

        st.write("Informasi Visual :")  
        topic_model.visualize_topics()
        topic_model.visualize_hierarchy()
        topic_model.visualize_barchart()
        topic_model.visualize_heatmap()
        topic_model.visualize_term_rank()


        st.write("Elbow Method dengan KMeans")  
        df = pd.read_csv("topic_info_all_koherensi.csv")
        Topic = df['Topic'][1:].tolist()
        Count = df['Count'][1:].tolist()
        data = np.array([Topic, Count]).T

        wcss = []
        for i in range(1, len(data) + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, len(data) + 1), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()

        st.write("Word Cloud")  
        df = pd.read_csv("topic_info_all_koherensi.csv")
        topics = df[['Topic', 'Representation']]
        representations = topics['Representation'].tolist()
        all_representations = ' '.join(representations)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_representations)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud dari Representasi Topik')
        plt.show()



        st.write("Menampilkan Informasi Kelompok")  
        topic_model.get_document_info(docs)
        document_info = topic_model.get_document_info(docs)
        df = pd.DataFrame(document_info)
        df.to_csv('resultTopic.csv', index=False)

        df = pd.read_csv('resultTopic.csv')
        df
        st.write("Informasi Kelompok success disimpan di resultTopic.csv")  




        st.write("Seluruh Prosess selesai") 
        st.success("Prosess Finish")
        


def informasi_artikel():
    if st.button("Tampilkan"):   
        st.title("Informasi Artikel")       
        df = pd.read_csv('resultTopic.csv')
        df

        st.write("Finish") 
        st.success("Prosess Finish")


                     

pages = {
    "Home"   : home_page,    
    "BERTTopic": proses_page,
    "Informasi Artikel": informasi_artikel,
}
selection = st.sidebar.radio("Menu", list(pages.keys()))
pages[selection]()
