import numpy as np
import librosa
import joblib
import torch
import openunmix
import torchaudio
import tensorflow as tf
from skimage.transform import resize
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Load the pre-trained model
model = tf.keras.models.load_model(r"D:\Tag\Musical_Instruments_Classification\Trained_model.h5")

# Define instrument classes
classes = ['Accordion', 'Acoustic_Guitar', 'Banjo', 'Bass_Guitar', 'Clarinet', 'cowbell', 'Cymbals', 
           'Dobro', 'Drum_set', 'Electro_Guitar', 'Floor_Tom', 'flute', 'Harmonica', 'Harmonium', 
           'Hi_Hats', 'Horn', 'Keyboard', 'Mandolin', 'Organ', 'Piano', 'Saxophone', 'Shakers', 
           'Tambourine', 'Trombone', 'Trumpet', 'Ukulele', 'vibraphone', 'Violin']

# Function to separate and save vocals and accompaniment from an audio file
def separate_audio(audio_path):
    # Load the pre-trained model
    separator = openunmix.umxl()

    # Load the audio file and preprocess it
    try:
        audio, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Ensure audio has the correct shape (1, C, T)
    if audio.dim() == 1:  # mono
        audio = audio.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, T)
    elif audio.dim() == 2:  # stereo
        audio = audio.unsqueeze(0)  # Shape (1, C, T)

    # Perform separation
    with torch.no_grad():  # Disable gradient calculation
        estimates = separator(audio)

    # Extract the vocal and accompaniment parts
    if estimates.dim() == 3:
        vocals = estimates[0, 0]  # First output for vocals
        accompaniment = estimates[0, 1]  # Second output for accompaniment
    elif estimates.dim() == 4 and estimates.shape[1] >= 2:
        vocals = estimates[0, 0]  # First channel corresponds to vocals
        accompaniment = estimates[0, 1]  # Second channel corresponds to accompaniment
    else:
        return None

    # Ensure we save 2D tensors
    if vocals.dim() == 3:  # If the vocals tensor is still 3D
        vocals = vocals.squeeze(0)  # Reduce to [C, T]

    if accompaniment.dim() == 3:  # If the accompaniment tensor is still 3D
        accompaniment = accompaniment.squeeze(0)  # Reduce to [C, T]

    vocal_file = 'vocals.wav'
    accompaniment_file = 'accompaniment.wav'
    
    # Save the vocal audio file
    try:
        torchaudio.save(vocal_file, vocals, sample_rate=sample_rate)  # Save the extracted vocals
    except Exception as e:
        print(f"Error saving vocal audio file: {e}")

    # Save the accompaniment audio file
    try:
        torchaudio.save(accompaniment_file, accompaniment, sample_rate=sample_rate)  # Save the extracted accompaniment
        return vocal_file, accompaniment_file  # Return the audio files for instrument prediction
    except Exception as e:
        print(f"Error saving accompaniment audio file: {e}")
        return None

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Define chunk and overlap durations
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    chunks = []
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        
        # Ensure chunk length does not exceed audio length
        if end > len(audio_data):
            end = len(audio_data)
            start = end - chunk_samples
        
        chunk = audio_data[start:end]
        
        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        chunks.append(mel_spectrogram)
    
    return np.array(chunks), sample_rate, chunk_duration, overlap_duration

# Function to make predictions on chunks
def model_prediction(chunks):
    y_pred = model.predict(chunks)
    return y_pred

# Visualize the instrument probabilities over time
def plot_instrument_usage(y_pred, chunk_duration, overlap_duration):
    time_steps = np.arange(y_pred.shape[0]) * (chunk_duration - overlap_duration)
    
    plt.figure(figsize=(15, 8))
    for i, instrument in enumerate(classes):
        plt.plot(time_steps, y_pred[:, i], label=instrument)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.title('Instrument Usage Over Time')
    plt.legend(loc='upper right')
    plt.show()

# List instruments with mean probability > 5%
def list_prominent_instruments(y_pred, threshold=0.5):
    mean_probabilities = np.mean(y_pred, axis=0)  # Calculate mean probabilities for each instrument
    prominent_instruments = [classes[i] for i in range(len(classes)) if mean_probabilities[i] > threshold]
    
    return prominent_instruments

# Plot Mel Spectrograms of some chunks
def plot_spectrograms(chunks, sample_rate, n_chunks=3):
    plt.figure(figsize=(15, 5))
    for i in range(min(n_chunks, chunks.shape[0])):
        plt.subplot(1, n_chunks, i + 1)
        plt.title(f'Chunk {i+1}')
        mel_spectrogram = chunks[i].squeeze()  # Remove single-dimensional entries
        librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sample_rate, y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# Function to list the top 6 instruments by mean probability
def list_top_instruments(y_pred, classes, top_n=6):
    mean_probabilities = np.mean(y_pred, axis=0)  # Calculate mean probabilities for each instrument
    
    # Create a list of (instrument, mean_probability) tuples
    instrument_probabilities = [(classes[i], mean_probabilities[i]) for i in range(len(classes))]
    
    # Sort instruments by mean probability in descending order
    sorted_instruments = sorted(instrument_probabilities, key=lambda x: x[1], reverse=True)
    
    top_instruments = sorted_instruments[:top_n]
    return top_instruments

# FastAPI endpoint to process audio file
class PredictionResponse(BaseModel):
    top_instruments: list

@app.post("/process_audio", response_model=PredictionResponse)
async def process_audio(file: UploadFile = File(...)):
    audio_file_path = f"temp_{file.filename}"
    
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    vocal_file, accompaniment_file = separate_audio(audio_file_path)

    if accompaniment_file:
        chunks, sample_rate, chunk_duration, overlap_duration = load_and_preprocess_data(accompaniment_file)
        y_pred = model_prediction(chunks)
        plot_spectrograms(chunks, sample_rate)
        plot_instrument_usage(y_pred, chunk_duration, overlap_duration)
        top_instruments = list_top_instruments(y_pred, classes)
        return PredictionResponse(top_instruments=top_instruments)
    else:
        return {"error": "Audio separation failed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
