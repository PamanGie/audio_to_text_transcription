# Audio-to-Text Transcription with Wav2Vec 2.0

This project provides an automatic transcription pipeline for converting audio files to text using **Wav2Vec 2.0**, a state-of-the-art model for speech recognition developed by Facebook AI. The audio files are processed from a directory, and the transcriptions are stored in a CSV file.

## Features
- **Automatic Speech Recognition (ASR)** using pre-trained Wav2Vec 2.0 model from Hugging Face Transformers.
- Supports various audio formats via **pydub** and **torchaudio**.
- Transcriptions are saved into a CSV file with the corresponding audio filenames.

## Requirements

To use this project, you will need the following dependencies:

- **Python 3.8+**
- **transformers** (for Wav2Vec 2.0 model)
- **torchaudio** (for loading and processing audio)
- **pydub** (for handling various audio formats)
- **tqdm** (for progress bar)
- **ffmpeg** (for handling audio conversions)

### Install Required Packages

Install the necessary packages via `pip`:

pip install transformers torchaudio pydub tqdm

# Audio-to-Text Transcription with Wav2Vec 2.0

This project provides an automatic transcription pipeline for converting audio files to text using **Wav2Vec 2.0**, a state-of-the-art model for speech recognition developed by Facebook AI. The audio files are processed from a directory, and the transcriptions are stored in a CSV file.

## How to Run

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Prepare your audio files in a folder, ensuring they are in WAV format.

3. Run the transcription script:

    ```bash
    python transcribe.py
    ```

4. The transcriptions will be saved in a CSV file, specified by you.



