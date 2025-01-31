from langchain_community.vectorstores import FAISS
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class SimilaritySearch:
    def __init__(self, vectordb : FAISS):
        self.vectordb = vectordb

    def score(self, cosine_distance_value):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([[0],[2]]))
        scaled_value = scaler.transform(np.array([[cosine_distance_value]]))
        return (1 - scaled_value).reshape(-1)[0]

    def search_by_vector(self, search_embeddings, top_k = 5, score_threshold = 0.8):
        search_results = self.vectordb.similarity_search_with_score_by_vector(search_embeddings, k=top_k)
        result = [{"title":d[0].metadata["title"], "artist":d[0].metadata["artist"], "score":self.score(d[1])} for d in search_results if self.score(d[1]) >= score_threshold]
        if len(result) == 0:
            result = [{"title":"No results found", "artist":"No results found", "score":f"No results found greater than {score_threshold}"}]
        return result 