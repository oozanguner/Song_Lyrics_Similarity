from langchain_community.vectorstores import FAISS
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class SimilaritySearch:
    """
    A class for performing similarity searches using vector embeddings with FAISS.

    This class provides functionality to compute a similarity score based on
    cosine distance and to search for similar vectors in a FAISS vector database.

    Attributes:
        vectordb (FAISS): A FAISS vector database instance used to perform similarity searches.
    """
    def __init__(self, vectordb : FAISS):
        """
        Initialize the SimilaritySearch instance with the specified FAISS vector database.

        Args:
            vectordb (FAISS): The FAISS vector database instance.
        """
        self.vectordb = vectordb

    def score(self, cosine_distance_value):
        """
        Calculate a similarity score based on the given cosine distance value.

        This method scales the provided cosine distance value to a [0, 1] range using a
        MinMaxScaler and then computes the similarity score as 1 minus the scaled value.

        Args:
            cosine_distance_value (float): The cosine distance value to transform.

        Returns:
            float: The computed similarity score in the range [0, 1].
        """
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(np.array([[0],[2]]))
        scaled_value = scaler.transform(np.array([[cosine_distance_value]]))
        return (1 - scaled_value).reshape(-1)[0]

    def search_by_vector(self, search_embeddings, top_k = 5, score_threshold = 0.8):
        """
        Perform a similarity search using the provided vector embeddings.

        This method queries the FAISS vector database to retrieve the top_k similar vectors along with their
        cosine distance scores, transforms these scores into similarity scores, and filters the results based on
        the specified score threshold.

        Args:
            search_embeddings: The vector embeddings used for performing the similarity search. This can be a NumPy array or a list.
            top_k (int, optional): The maximum number of results to retrieve. Defaults to 5.
            score_threshold (float, optional): The minimum similarity score required for a result to be included. Defaults to 0.8.

        Returns:
            list: A list of dictionaries for each result that meets the score threshold. Each dictionary contains:
                - "title": The title metadata of the item.
                - "artist": The artist metadata of the item.
                - "score": The calculated similarity score.
                If no results meet the threshold, a list with a single dictionary indicating no results is returned.
        """
        search_results = self.vectordb.similarity_search_with_score_by_vector(search_embeddings, k=top_k)
        result = [{"title":d[0].metadata["title"], "artist":d[0].metadata["artist"], "score":self.score(d[1])} for d in search_results if self.score(d[1]) >= score_threshold]
        if len(result) == 0:
            result = [{"title":"No results found", "artist":"No results found", "score":f"No results found greater than {score_threshold}"}]
        return result 