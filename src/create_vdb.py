from helpers.vectordatabases import *
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = os.getenv('EMBED_MODEL')
VDB_PATH = os.getenv('VDB_PATH')
VDB_INDEX_NAME = os.getenv('VDB_INDEX_NAME')
LYRICS_DATASET_HANDLE = os.getenv('LYRICS_DATASET_HANDLE')
LYRICS_FILE_PATH = os.getenv('LYRICS_FILE_PATH')

if __name__ == "__main__":
    lyrics_loader = LyricsLoader(kaggle_handle = LYRICS_DATASET_HANDLE, kaggle_lyrics_path = LYRICS_FILE_PATH)
    lyrics_df = lyrics_loader.load_lyrics()
    lyrics_df["cleaned_lyrics"] = lyrics_df["lyrics"].apply(clean_lyrics)
    lyrics_df["features"] = lyrics_df["features"].apply(remove_braces_double_quotes)

    lyrics_embedder = LyricsEmbedder(model_name=EMBED_MODEL)
    metadatas = []
    lyrics_vectors = []
    for i in tqdm(range(len(lyrics_df))):
        lyrics = lyrics_df.iloc[i]["cleaned_lyrics"]
        lyrics_splits = lyrics_embedder.split_lyrics(lyrics)
        mean_lyrics_embs = lyrics_embedder.embed_average(lyrics_splits)
        payload = {
            "title": lyrics_df.iloc[i]["title"],
            "artist": lyrics_df.iloc[i]["artist"],
            "year": lyrics_df.iloc[i]["year"],
            "features": lyrics_df.iloc[i]["features"],
            "language": lyrics_df.iloc[i]["language"],
            "id": lyrics_df.iloc[i]["id"],
            "views": lyrics_df.iloc[i]["views"]    
        }
        metadatas.append(payload)
        lyrics_vectors.append((lyrics_df["cleaned_lyrics"][i], mean_lyrics_embs))
    
    vectorstore = VectorStore(db_path=VDB_PATH, db_index_name=VDB_INDEX_NAME, model_name=EMBED_MODEL)
    vectorstore.add_embeds(embedding_vectors=lyrics_vectors, metadatas=metadatas)
    vectorstore.save_db()