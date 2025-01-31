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
    def __init__(self, kaggle_handle, kaggle_lyrics_path):
        start_time = time.time()
        self.lyrics_handle = kagglehub.dataset_download(kaggle_handle, path=kaggle_lyrics_path)
        end_time = time.time()
        print(f'Downloading Time : {round((end_time-start_time), 2)} seconds')

    def load_lyrics(self, chunksize = 500000, top_n = 1000):
        top_view_df = pd.DataFrame()
        for chunk in tqdm(pd.read_csv(self.lyrics_handle, chunksize=chunksize)):
            chunk_top = chunk.nlargest(top_n, 'views')
            top_view_df = pd.concat([top_view_df, chunk_top])
            top_view_df = top_view_df.nlargest(top_n, 'views')
        
        top_view_df = top_view_df.reset_index(drop=True)     
        return top_view_df

    
def clean_lyrics(lyrics):
    pattern = r'\[.+]'
    cleaned_lyrics = re.sub(pattern, '', lyrics).strip()
    cleaned_lyrics = cleaned_lyrics.strip('""')
    return cleaned_lyrics

    
def remove_braces_double_quotes(lyrics):
    lyrics = lyrics.strip('{}').strip('""')
    return lyrics


class LyricsEmbedder:
    def __init__(self, model_name):
        self.embed_model = SentenceTransformer(model_name)
        self.embed_model_max_seq_length = self.embed_model.get_max_seq_length()
    
    def split_lyrics(self, lyrics, chunk_overlap = 40):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.embed_model_max_seq_length, 
                                                       chunk_overlap=chunk_overlap, 
                                                       length_function=len, 
                                                       is_separator_regex=False,
                                                       separators=[' '])
        return text_splitter.split_text(lyrics)

    def embed(self, lyrics):
        return self.embed_model.encode(lyrics, normalize_embeddings=True)
    
    def embed_average(self, lyrics):
        return self.embed_model.encode(lyrics, normalize_embeddings=True).mean(axis=0).reshape(-1)


class VectorStore(LyricsEmbedder):
    def __init__(self, db_path, db_index_name, model_name):
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
        uuids = [str(uuid4()) for _ in range(len(embedding_vectors))]
        self.db.add_embeddings(text_embeddings=embedding_vectors, metadatas=metadatas, ids = uuids)

    def save_db(self):
        try:
            self.db.save_local(folder_path=self.db_path, index_name=self.db_index_name)
            print("Vectordatabase is created successfully")
        except:
            print("Saving embeddings to the vectordatabase failed")

    def load_db(self):
        return FAISS.load_local(folder_path=self.db_path, 
                                embeddings=self.embed_model, 
                                index_name=self.db_index_name, 
                                allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.COSINE)    