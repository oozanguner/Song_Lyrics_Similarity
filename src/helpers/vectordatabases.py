from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer
import faiss
import os 
from uuid import uuid4
import kagglehub
from dotenv import load_dotenv
import time
import pandas as pd
from tqdm import tqdm
import re


class LyricsLoader:
    """
    A class to download and load a lyrics dataset from Kaggle.

    This class downloads the dataset using KaggleHub and provides functionality
    to load and filter the lyrics data based on view counts.
    """
    def __init__(self, kaggle_handle, kaggle_lyrics_path):
        """
        Initialize the LyricsLoader and download the dataset.

        The dataset is downloaded from Kaggle using the provided handle and saved
        to the specified path. The time taken to download is printed.

        Args:
            kaggle_handle (str): The Kaggle dataset handle.
            kaggle_lyrics_path (str): The local path where the lyrics dataset will be stored.
        """
        start_time = time.time()
        self.lyrics_handle = kagglehub.dataset_download(kaggle_handle, path=kaggle_lyrics_path)
        end_time = time.time()
        print(f'Downloading Time : {round((end_time-start_time), 2)} seconds')

    def load_lyrics(self, chunksize = 500000, top_n = 1000):
        """
        Load and filter the lyrics dataset by views.

        The dataset is read in chunks from a CSV file. For each chunk, the top_n rows
        with the highest 'views' are selected. The final DataFrame consists of the
        overall top entries based on view counts.

        Args:
            chunksize (int, optional): Number of rows per chunk to read from the CSV. Defaults to 500000.
            top_n (int, optional): Number of top rows (by views) to retain per chunk. Defaults to 1000.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered lyrics data.
        """
        top_view_df = pd.DataFrame()
        for chunk in tqdm(pd.read_csv(self.lyrics_handle, chunksize=chunksize)):
            chunk_top = chunk.nlargest(top_n, 'views')
            top_view_df = pd.concat([top_view_df, chunk_top])
            top_view_df = top_view_df.nlargest(top_n, 'views')
        
        top_view_df = top_view_df.reset_index(drop=True)     
        return top_view_df

    
def clean_lyrics(lyrics):
    """
    Clean the lyrics text by removing text enclosed in square brackets and extra double quotes.

    This function removes any substrings enclosed within square brackets (e.g., [Chorus])
    and strips extraneous double quotes from the text.

    Args:
        lyrics (str): The raw lyrics text.

    Returns:
        str: The cleaned lyrics text.
    """
    pattern = r'\[.+]'
    cleaned_lyrics = re.sub(pattern, '', lyrics).strip()
    cleaned_lyrics = cleaned_lyrics.strip('""')
    return cleaned_lyrics

    
def remove_braces_double_quotes(lyrics):
    """
    Remove surrounding braces and double quotes from the lyrics text.

    This function strips the outermost curly braces and double quotes from the text.

    Args:
        lyrics (str): The raw lyrics text.

    Returns:
        str: The lyrics text with surrounding braces and double quotes removed.
    """
    lyrics = lyrics.strip('{}').strip('""')
    return lyrics


class LyricsEmbedder:
    """
    A class to embed lyrics text using a SentenceTransformer model.

    Provides functionality to split long lyrics into chunks suitable for the model,
    generate embeddings for given texts, and compute an averaged embedding.
    """
    def __init__(self, model_name):
        """
        Initialize the LyricsEmbedder with a specified SentenceTransformer model.

        Args:
            model_name (str): The name or path of the pre-trained SentenceTransformer model.
        """
        self.embed_model = SentenceTransformer(model_name)
        self.embed_model_max_seq_length = self.embed_model.get_max_seq_length()
    
    def split_lyrics(self, lyrics, chunk_overlap = 40):
        """
        Split lyrics into chunks that fit within the model's maximum sequence length.

        Utilizes a RecursiveCharacterTextSplitter to break the lyrics into smaller segments,
        considering a specified overlap between chunks.

        Args:
            lyrics (str): The lyrics text to split.
            chunk_overlap (int, optional): Number of overlapping characters between consecutive chunks. Defaults to 40.

        Returns:
            list: A list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.embed_model_max_seq_length, 
                                                       chunk_overlap=chunk_overlap, 
                                                       length_function=len, 
                                                       is_separator_regex=False,
                                                       separators=[' '])
        return text_splitter.split_text(lyrics)

    def embed(self, lyrics):
        """
        Generate embeddings for the provided lyrics text.

        Args:
            lyrics (str or list): The lyrics text or list of texts to embed.

        Returns:
            np.array: The embeddings for the given text.
        """
        return self.embed_model.encode(lyrics, normalize_embeddings=True)
    
    def embed_average(self, lyrics):
        """
        Generate an average embedding vector for the provided lyrics text.

        This method computes the embeddings and returns the mean vector across all tokens.

        Args:
            lyrics (str or list): The lyrics text or list of texts to embed.

        Returns:
            np.array: The averaged embedding vector.
        """
        return self.embed_model.encode(lyrics, normalize_embeddings=True).mean(axis=0).reshape(-1)


class VectorStore(LyricsEmbedder):
    """
    A class to manage a FAISS vector database for storing and retrieving lyrics embeddings.

    Inherits from LyricsEmbedder to generate embeddings and adds functionality to store
    these embeddings in a FAISS index along with associated metadata.
    """
    def __init__(self, db_path, db_index_name, model_name):
        """
        Initialize the VectorStore with a FAISS index and an embedding model.

        Args:
            db_path (str): The directory path where the FAISS index will be saved.
            db_index_name (str): The name of the FAISS index.
            model_name (str): The name or path of the pre-trained SentenceTransformer model.
        """
        super().__init__(model_name=model_name)
        embedding_dim = len(self.embed("hello world"))
        self.db_path = db_path
        self.db_index_name = db_index_name
        self.db = FAISS(embedding_function=self.embed_model,
                        index = faiss.IndexFlatL2(embedding_dim),
                        docstore=InMemoryDocstore(),
                        index_to_docstore_id={},
                        distance_strategy=DistanceStrategy.COSINE
                        )
        
    def add_embeds(self, embedding_vectors, metadatas):
        """
        Add embeddings along with their metadata to the vector database.

        Each embedding is assigned a unique identifier before being added.

        Args:
            embedding_vectors (list or np.array): A list or array of embedding vectors.
            metadatas (list): A list of metadata dictionaries corresponding to each embedding.
        """
        uuids = [str(uuid4()) for _ in range(len(embedding_vectors))]
        self.db.add_embeddings(text_embeddings=embedding_vectors, metadatas=metadatas, ids = uuids)

    def save_db(self):
        """
        Save the FAISS vector database to local storage.

        The vector database is stored in the specified directory using the given index name.
        A success message is printed if the operation succeeds; otherwise, an error message is printed.
        """

        try:
            self.db.save_local(folder_path=self.db_path, index_name=self.db_index_name)
            print("Vectordatabase is created successfully")
        except:
            print("Saving embeddings to the vectordatabase failed")

    def load_db(self):
        """
        Load the FAISS vector database from local storage.

        Returns:
            FAISS: The loaded FAISS vector database instance.
        """
        return FAISS.load_local(folder_path=self.db_path, 
                                embeddings=self.embed_model, 
                                index_name=self.db_index_name, 
                                allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.COSINE)    