🎵 Music Genre Classification with CNN & Spectrograms
A deep learning project that classifies music genres by converting audio files into spectrogram images and training a Convolutional Neural Network (CNN).
Built using Python, Librosa, Matplotlib, and TensorFlow/Keras, and trained on the GTZAN Dataset.

🎯 Goal
Automatically predict the genre of a given music clip using spectrogram-based image classification.

🧰 Tech Stack
Python 3.x

Librosa – Audio processing & feature extraction

Matplotlib – Spectrogram generation

NumPy & Pandas – Data handling

TensorFlow / Keras – CNN model building and training

OpenCV (optional) – For preprocessing or visualization

🎼 Dataset
GTZAN Music Genre Dataset

10 Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

100 audio tracks per genre (1000 total)

Format: .wav, 30 sec each

Source: Marsyas/GTZAN

🔄 Workflow
Load Audio Files using librosa

Convert to Spectrograms (Mel Spectrogram or MFCCs)

Save Spectrograms as Images

Preprocess Images (resize, normalize)

Build CNN Model

Train & Evaluate

Predict Genre of New Audio

📁 Folder Structure
bash
Copy code
📦 music-genre-classification
├── genres_original/           # Raw GTZAN .wav files
├── images/                    # Generated spectrograms per genre
├── models/                    # Saved CNN models
├── notebooks/                 # Jupyter notebooks for EDA and model training
├── main.py                    # Main training script
├── predict.py                 # For genre prediction on new audio
├── utils.py                   # Helper functions
├── requirements.txt
└── README.md
📊 Model Architecture (Example)
text
Copy code
Input: 128x128x3 spectrogram image
→ Conv2D + ReLU + MaxPooling
→ Conv2D + ReLU + MaxPooling
→ Flatten
→ Dense + Dropout
→ Dense (10 units, softmax)
🚀 Getting Started
✅ Install Dependencies
bash
Copy code
pip install librosa matplotlib tensorflow opencv-python numpy pandas
🛠 Preprocess Audio
Convert .wav files to spectrograms:

python
Copy code
import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load('genres_original/jazz/jazz.00054.wav')
S = librosa.feature.melspectrogram(y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S), sr=sr)
plt.savefig('images/jazz/jazz00054.png')
🎓 Train the Model
bash
Copy code
python main.py
🔍 Predict Genre
bash
Copy code
python predict.py --file my_song.wav
📈 Results
Accuracy: ~85-90% (after tuning)

Confusion Matrix, F1 Score, etc. can be included

🧠 Highlights
Uses spectrograms as input instead of raw audio

CNN treats music as an image classification task

Supports adding your own genres with minimal changes

Easily extendable with RNNs, CRNNs, or transfer learning
