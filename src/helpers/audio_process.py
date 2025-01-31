import yt_dlp
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from .vectordatabases import *
from .find_similarity import *

class ExtractYoutubeAudio:
    def __init__(self, url):
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
        with yt_dlp.YoutubeDL(self.ydl_opt) as ydl:
            info_dict = ydl.extract_info(self.url)
            title = info_dict['title']
            url = info_dict['url']
            return {'title': title, 'url': url}
    
        return None

class AudioToText:
    def __init__(self, url, audio_model_name):
        self.url = url
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForSpeechSeq2Seq.from_pretrained(audio_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(audio_model_name)
        self.whisper = pipeline('automatic-speech-recognition',
                                model=model,
                                tokenizer=processor.tokenizer,
                                feature_extractor=processor.feature_extractor,
                                torch_dtype = torch_dtype,
                                device = device,
                                return_timestamps=True
                                )
        
    def transcribe(self):
        lyrics_list = []
        for segment in self.whisper(self.url)['chunks']:
            lyrics_list.append(segment['text'])
        lyrics = '\n'.join(lyrics_list)
        return lyrics