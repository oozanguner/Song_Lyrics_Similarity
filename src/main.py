import pydantic
from fastapi import FastAPI, Response, Path
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
from typing import Union
import uvicorn
from pydantic import BaseModel
from helpers.audio_process import ExtractYoutubeAudio, AudioToText
from helpers.vectordatabases import VectorStore, LyricsEmbedder, clean_lyrics, remove_braces_double_quotes
from helpers.find_similarity import SimilaritySearch
import time
import os
import logging
from warnings import filterwarnings
filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_MODEL = os.getenv('AUDIO_MODEL')
EMBED_MODEL = os.getenv('EMBED_MODEL')
VDB_PATH = os.getenv('VDB_PATH')
VDB_INDEX_NAME = os.getenv('VDB_INDEX_NAME')

app = FastAPI(title="Song Lyrics Similarity")

class Item(BaseModel):
    url: str

class SimilarSong(BaseModel):
    title: str
    artist: str
    score: Union[float, str] 


@app.post("/get_title/", description="Extract title from youtube url")
async def get_title(item: Item):
    url = item.url
    start_time = time.time()
    try:
        audio = ExtractYoutubeAudio(url).extract_audio()
        end_time = time.time()
        if audio:
            return JSONResponse(status_code = 200,
                                content= {"title": audio['title'],
                                          "extraction_time": round((end_time - start_time), 2)}
            )
    except:
        return JSONResponse(status_code=500, 
                            content={"code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    "message":"Invalid URL"})


@app.post("/get_lyrics/", description="Extract lyrics from youtube url")
async def get_lyrics(item: Item):
    url = item.url
    start_time = time.time()
    try:
        audio = ExtractYoutubeAudio(url).extract_audio()
        end_time1 = time.time()
        if audio:
            lyrics = AudioToText(url=audio['url'], audio_model_name=AUDIO_MODEL).transcribe()
            end_time2 = time.time()
            return JSONResponse(status_code = 200,
                                content= {"title": audio['title'],
                                          "lyrics": lyrics,
                                          "extraction_time": round((end_time1 - start_time), 2),
                                          "transcription_time": round((end_time2 - end_time1), 2)}
            )
    except:
        return JSONResponse(status_code=500,
                            content={"code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    "message":"Invalid URL"})


@app.post("/find_similarity/", description="Find similar songs from the database", response_model=list[SimilarSong])
async def find_similarity(item: Item):
    url = item.url
    try:
        logger.info("Starting audio extraction")
        audio = ExtractYoutubeAudio(url).extract_audio()
        if not audio:
            logger.error("Audio extraction failed")
            return JSONResponse(status_code=500,
                                content={"code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                        "message":"Audio extraction failed"})
        logger.info("Starting audio transcription")
        lyrics = AudioToText(url=audio['url'], audio_model_name=AUDIO_MODEL).transcribe()
        if not lyrics:
            logger.error("Transcription failed")
            return JSONResponse(status_code=500,
                                content={"code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                         "message": "Transcription failed"})        
        logger.info("Processing lyrics")
        lyrics = clean_lyrics(lyrics)
        lyrics = remove_braces_double_quotes(lyrics)

        logger.info("Creating embeddings")
        lyrics_embedder = LyricsEmbedder(model_name=EMBED_MODEL)
        lyrics_chunks = lyrics_embedder.split_lyrics(lyrics)
        mean_lyrics_embs = lyrics_embedder.embed_average(lyrics_chunks)
        
        logger.info("Loading vector database")
        vectorstore = VectorStore(db_path = VDB_PATH, db_index_name = VDB_INDEX_NAME, model_name = EMBED_MODEL)
        vdb = vectorstore.load_db()
        if not vdb:
            logger.error("Failed to load vector database")
            return JSONResponse(status_code=500,
                                content={"code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                         "message": "Failed to load vector database"})
        
        logger.info("Performing similarity search")
        results = SimilaritySearch(vdb).search_by_vector(mean_lyrics_embs, top_k=5)
        
        logger.info("Search completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error in find_similarity: {str(e)}")
        return JSONResponse(status_code=500,
                            content={"code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    "message":f"Error processing request: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)