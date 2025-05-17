import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df=pd.read_csv("C:\\Users\\perar\\Desktop\\Netflix_movies_and_tv_shows_clustering.csv\\Cleaned_Netflix_Data.csv")

df['combined_column'] = df['listed_in'] + ' ' + df['cast']

vector=TfidfVectorizer(stop_words='english')
feature=vector.fit_transform(df['combined_column'])

similarity = cosine_similarity(feature)

def recommendation(title, no_of_recomd=5):
    if title not in df['title'].values:
        print(title,"not found")
        return
    
    idx = df[df['title'].str.lower() == title.lower()].index[0]

    similarity_score = list(enumerate(similarity[idx].flatten()))

    sort_score = sorted(similarity_score, key=lambda x : x[1], reverse=True)[1:no_of_recomd+1]

    print('Recommendations for', title)

    for i, (index, score) in enumerate(sort_score):
        print(f'{i+1}. {df.iloc[index]['title']} (Similarity:{score:.2f})')

recommendation('stranger things', no_of_recomd=5)


# Content Clustering

num_clusters=5

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(feature)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature.toarray())

df['pca1'] = reduced_features[:,0]
df['pca2'] = reduced_features[:,1]

sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set1', s=60, alpha=0.7)
plt.title('Netflix Titles Clustered by Genre and Cast')
plt.show()

for cluster in range(num_clusters):
    print(f"\nCluster {cluster} Titles:")
    print(df[df['cluster'] == cluster]['title'].head(5).to_list())


