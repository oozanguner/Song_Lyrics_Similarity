import yt_dlp
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from .vectordatabases import *
from .find_similarity import *
import os
from dotenv import load_dotenv

load_dotenv()

HFACE_TOKEN = os.getenv('HFACE_TOKEN')

class ExtractYoutubeAudio:
    """
    Extract audio information from a YouTube video using yt_dlp.

    This class encapsulates the functionality needed to retrieve audio metadata
    (such as the title and audio URL) from a YouTube video without downloading the full video.

    Attributes:
        url (str): The URL of the YouTube video.
        ydl_opt (dict): Configuration options for yt_dlp to extract audio.
    """
    def __init__(self, url):
        """
        Initialize an instance of ExtractYoutubeAudio with the specified YouTube URL.

        Args:
            url (str): The URL of the YouTube video.
        """
        self.url = url
        self.ydl_opt = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'skip_download': True,
        }

    def extract_audio(self):
        """
        Extract audio metadata from the specified YouTube video.

        Uses yt_dlp to extract information from the provided YouTube URL and retrieves
        the video's title and the URL for the best available audio stream.

        Returns:
            dict: A dictionary with the following keys:
                - 'title': The title of the video.
                - 'url': The URL of the extracted audio stream.
        """
        with yt_dlp.YoutubeDL(self.ydl_opt) as ydl:
            info_dict = ydl.extract_info(self.url)
            title = info_dict['title']
            url = info_dict['url']
            return {'title': title, 'url': url}
    
        return None

class AudioToText:
    """
    Transcribe audio from a given URL into text using a speech recognition model.

    This class loads a pre-trained speech recognition model from Hugging Face and sets up
    an automatic speech recognition pipeline. It can process audio input from a URL and
    return the transcription text.

    Attributes:
        url (str): The URL of the audio source.
        whisper (pipeline): The Hugging Face pipeline for automatic speech recognition.
    """
    def __init__(self, url, audio_model_name):
        """
        Initialize an instance of AudioToText with the provided audio URL and model name.

        This method loads the pre-trained model and its processor from Hugging Face,
        configures the appropriate computation device (GPU if available, otherwise CPU),
        and initializes the speech recognition pipeline.

        Args:
            url (str): The URL of the audio source.
            audio_model_name (str): The name or path of the pre-trained audio model.
        """
        self.url = url
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForSpeechSeq2Seq.from_pretrained(audio_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, token=HFACE_TOKEN)
        model.to(device)
        processor = AutoProcessor.from_pretrained(audio_model_name, token=HFACE_TOKEN)
        self.whisper = pipeline('automatic-speech-recognition',
                                model=model,
                                tokenizer=processor.tokenizer,
                                feature_extractor=processor.feature_extractor,
                                torch_dtype = torch_dtype,
                                device = device,
                                return_timestamps=True
                                )
        
    def transcribe(self):
        """
        Transcribe the audio from the URL into text.

        This method processes the audio using the automatic speech recognition pipeline,
        aggregates text segments from each detected chunk, and concatenates them into a
        single transcription string.

        Returns:
            str: The complete transcription of the audio.
        """
        lyrics_list = []
        for segment in self.whisper(self.url)['chunks']:
            lyrics_list.append(segment['text'])
        lyrics = '\n'.join(lyrics_list)
        return lyrics