import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import librosa
import numpy as np

SAMPLE_RATE = 22050
FIXED_DURATION = 2.5
NUM_MFCC = 20

def preprocess_audio(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        audio = librosa.util.fix_length(audio, size=int(FIXED_DURATION * SAMPLE_RATE))
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window='hamming'
        )
        return mfcc.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

EMOTION_LABELS_C = {0: 'ANG', 1: 'DIS', 2: 'FEA', 3: 'HAP', 4: 'NEU', 5: 'SAD'}
EMOTION_LABELS_R = {
    0: 'NEU', 
    1: 'CAL', 
    2: 'HAP', 
    3: 'SAD', 
    4: 'ANG', 
    5: 'FEA', 
    6: 'DIS', 
    7: 'SUR'
}

def predict_emotion(model, file_path, model_choice):
    mfcc = preprocess_audio(file_path)
    if mfcc is None:
        return None
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    predictions = model.predict(mfcc)
    predicted_label = np.argmax(predictions, axis=1)[0]
    
    if model_choice == "emotion_recognition_model_R":
        if predicted_label in EMOTION_LABELS_R:
            return EMOTION_LABELS_R[predicted_label]
        else:
            print(f"Invalid label predicted: {predicted_label}")
            return None
    
    elif model_choice == "emotion_recognition_model_C":
        if predicted_label in EMOTION_LABELS_C:
            return EMOTION_LABELS_C[predicted_label]
        else:
            print(f"Invalid label predicted: {predicted_label}")
            return None

def load_model_based_on_choice(model_choice):
    if model_choice == "emotion_recognition_model_R":
        model_path = 'emotion_recognition_model_R.keras'
    elif model_choice == "emotion_recognition_model_C":
        model_path = 'emotion_recognition_model_C.keras'
    else:
        return None
    return load_model(model_path)

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("400x300")
        self.model_choice = tk.StringVar(value="emotion_recognition_model_R")
        self.model_label = tk.Label(self.root, text="Choose Model:")
        self.model_label.pack(pady=10)
        self.model_r_radio = tk.Radiobutton(self.root, text="emotion_recognition_model_R", variable=self.model_choice, value="emotion_recognition_model_R")
        self.model_r_radio.pack()
        self.model_c_radio = tk.Radiobutton(self.root, text="emotion_recognition_model_C", variable=self.model_choice, value="emotion_recognition_model_C")
        self.model_c_radio.pack()
        self.upload_button = tk.Button(self.root, text="Upload Audio File", command=self.upload_audio)
        self.upload_button.pack(pady=10)
        self.result_label = tk.Label(self.root, text="Predicted Emotion: ")
        self.result_label.pack(pady=10)
        self.predict_button = tk.Button(self.root, text="Predict Emotion", command=self.predict_emotion)
        self.predict_button.pack(pady=10)
        self.audio_file = None

    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if file_path:
            self.audio_file = file_path
            messagebox.showinfo("Info", f"Audio file '{file_path}' uploaded successfully.")

    def predict_emotion(self):
        if not self.audio_file:
            messagebox.showerror("Error", "Please upload an audio file first.")
            return
        model_choice = self.model_choice.get()
        model = load_model_based_on_choice(model_choice)
        if model is None:
            messagebox.showerror("Error", "Invalid model selected.")
            return
        emotion = predict_emotion(model, self.audio_file, model_choice)
        if emotion:
            self.result_label.config(text=f"Predicted Emotion: {emotion}")
        else:
            messagebox.showerror("Error", "Failed to process the audio file.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()