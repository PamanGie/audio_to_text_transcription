import os
import csv
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Coba beberapa kemungkinan path FFmpeg
FFMPEG_PATHS = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    os.environ.get("FFMPEG_PATH", ""),
]

def setup_ffmpeg():
    for path in FFMPEG_PATHS:
        if os.path.isfile(path):
            AudioSegment.converter = path
            logger.info(f"Using FFmpeg from: {path}")
            return True
    logger.error("FFmpeg not found in any of the specified paths!")
    return False

# Inisialisasi model dan processor Wav2Vec 2.0
def initialize_model():
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        return processor, model
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return None, None

# Fungsi untuk mengonversi audio ke transkrip
def transcribe_audio(file_path, processor, model):
    try:
        # Membaca file audio
        speech_array, sampling_rate = torchaudio.load(file_path)
        
        # Jika sampling rate tidak 16kHz, lakukan resampling
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech = resampler(speech_array).squeeze()
        else:
            speech = speech_array.squeeze()

        # Melakukan normalisasi dan inferensi
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Mendapatkan prediksi transkrip
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription.lower()
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def process_audio_folder(folder_path, output_csv, processor, model):
    if not os.path.exists(folder_path):
        logger.error(f"Folder path does not exist: {folder_path}")
        return

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['audio_file_name', 'transcription'])

        audio_files = [f for f in os.listdir(folder_path) if f.endswith((".wav", ".mp3", ".flac"))]
        
        if not audio_files:
            logger.warning(f"No audio files found in {folder_path}")
            return

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            audio_path = os.path.join(folder_path, audio_file)
            transcription = transcribe_audio(audio_path, processor, model)
            
            if transcription:
                logger.info(f"Transcription for {audio_file}: {transcription}")
                writer.writerow([audio_file, transcription])
            else:
                logger.warning(f"Failed to transcribe {audio_file}")

def main():
    # Setup FFmpeg
    if not setup_ffmpeg():
        return

    # Initialize model
    processor, model = initialize_model()
    if processor is None or model is None:
        return

    # Ganti path dengan folder audio Anda dan nama file CSV yang diinginkan
    audio_folder = "D:\\your\\corpusaudio\\folder\\clips"
    output_csv = "D:\\transcriptions.csv"

    # Proses folder dan simpan transkrip
    process_audio_folder(audio_folder, output_csv, processor, model)

if __name__ == "__main__":
    main()
