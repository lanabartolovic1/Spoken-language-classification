# Spoken Language Classification from Audio Recordings

This project focuses on recognizing the spoken language from short audio recordings using machine learning and deep learning techniques. The final model classifies audio into one of three languages: English, Spanish, or German.

## Project Description

The goal of the project is to develop and compare machine learning models for automatic spoken language identification (SLI). The models are trained on publicly available datasets and evaluated based on accuracy and F1-score. The best-performing model achieved **83% accuracy** using MFCC features and a convolutional neural network (CNN).

## Datasets

Two datasets were used in this project:

- **Spoken Language Identification Dataset** (Kaggle)  
  Augmented with noise, pitch, and speed variations. Stored in `.flac` format.

- **Spoken Languages Dataset** (Kaggle)  
  Clean speech samples in `.wav` format. Used as a separate test set to better represent real-world conditions.

## Models

Three models were implemented and compared:

1. **CNN + Spectrogram**  
   Raw spectrogram images generated from STFT used as CNN input.

2. **ResNet-50 + Spectrogram**  
   A pre-trained ResNet-50 fine-tuned on spectrogram data (converted to RGB).

3. **CNN + MFCC Features**  
   Mel-Frequency Cepstral Coefficients (MFCCs) used as input features.  
   **Best-performing model** with 83% accuracy and weighted F1-score.

## Technologies & Tools

- Python
- TensorFlow / Keras
- NumPy
- Librosa
- Matplotlib
- Jupyter Notebooks

## Results

| Model            | Accuracy | Weighted F1 Score |
|------------------|----------|-------------------|
| CNN + Spectrogram| 57%      | 56%               |
| ResNet-50        | 52%      | 54%               |
| CNN + MFCC       | 83%      | 83%               |

## Conclusion

MFCC features proved to be the most efficient representation of audio signals for this task. They enabled fast convergence and robust performance, outperforming models trained on raw spectrograms.

---


