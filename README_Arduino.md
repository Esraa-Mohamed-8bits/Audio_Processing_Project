# AI Person & Language Detection

An end-to-end voice recognition system that answers two questions from a short audio recording:

1. **Who is speaking?** — speaker/person identification
2. **Which language are they speaking?** — Arabic, English, French, or German

The project covers the full pipeline: audio preprocessing, leakage-safe data augmentation, feature extraction, model training/tuning, and a live web app where you record with your microphone and get predictions with confidence scores.

---

## Features

- **Speaker identification** — recognizes 3 enrolled speakers (`EsraaM`, `MWalaa`, `MariamB`)
- **Language detection** — classifies speech as Arabic, English, French, or German
- **Live web app (Streamlit)** — record with your mic, press *Predict*, see language + person with confidence and a flag image
- **Desktop app (Kivy)** — alternative native interface (`main_kivy.py`)
- **Leakage-safe pipeline** — train/test split happens *before* augmentation; only training files are ever augmented
- **Tuned models** — both classifiers are RBF-SVMs optimized with `GridSearchCV` (5-fold stratified CV)

## How It Works

### 1. Preprocessing (`voice_preprocessing.py`)
- Walks the dataset folder (`Dataset/<person>/<language>/*.wav`)
- Resamples everything to 22,050 Hz and trims silence
- Splits file paths into train/test (80/20) **before any audio is touched**, stratified by person+language
- Augments **training files only** — 3 random augmented copies per recording:
  - Added noise, time stretch (±15%), pitch shift (±2 semitones), time shift, volume change
- Saves `train_features.csv` and `test_features.csv` as separate files

### 2. Feature Extraction (librosa)
Each recording becomes a fixed-length vector of **55 features**:
| Feature group | Count | Captures |
|---|---|---|
| MFCCs | 17 | phonetic content / timbre |
| MFCC deltas | 17 | how phonetics change over time |
| Chroma | 12 | pitch-class / tonal content |
| Spectral contrast | 7 | peak-vs-valley energy (speaker cue) |
| Spectral centroid | 1 | brightness of the voice |
| Zero-crossing rate | 1 | noisiness / consonant-vowel balance |

### 3. Training (`person_training.py`, `language_training.py`, `language_training_specialized.py`)
- `StandardScaler` + `SVC` (RBF kernel, `probability=True`) in a single sklearn `Pipeline`
- Hyperparameters tuned with `GridSearchCV` over 5-fold stratified cross-validation
- Evaluation on a held-out test set + 5-fold CV on the full data
- Models and label encoders saved with `joblib`

## Results

| Task | Model | Test Accuracy | Notes |
|---|---|---|---|
| Person identification | RBF-SVM (C=5, gamma=scale) | **100%** (61 held-out samples) | 3 speakers; 5-fold CV: 99.8% |
| Language detection | RBF-SVM | **93.4%** (244 held-out samples) | macro precision 0.94 / recall 0.93 / F1 0.93 |

> **Caveat:** the dataset is small (300 original recordings per task), so high person-ID scores should be read as "works well for these enrolled speakers under these conditions" rather than production-grade speaker verification. Re-train on your own dataset for new speakers.

## Repository Structure

```
├── Dataset/                     # audio data (see structure below)
├── Models/                      # saved trained models (.pkl)
├── pics/                        # flag images shown in the web app
├── voice_preprocessing.py       # data collection, split, augmentation, feature extraction
├── person_training.py           # speaker-ID model training (SVM + GridSearchCV)
├── language_training.py         # language model training
├── language_training_specialized.py
├── app_streamlit.py             # live recording + prediction web app
├── main_kivy.py                 # desktop app (Kivy)
├── person_id_test.py            # standalone person-ID test script
├── language_predicition_test.py # standalone language test script
├── train_features.csv / test_features.csv
└── requirements.txt
```

## Dataset

Expected folder layout (all `.wav`):

```
Dataset/
├── person_1/
│   ├── Arabic/*.wav
│   ├── English/*.wav
│   ├── German/*.wav
│   └── French/*.wav
├── person_2/
│   └── ...
```

Adjust `collect_files()` in `voice_preprocessing.py` if your layout differs.

## Installation & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess audio -> train_features.csv / test_features.csv
python voice_preprocessing.py

# 3. Train the models
python person_training.py
python language_training_specialized.py

# 4. Run the web app
streamlit run app_streamlit.py
```

In the app: press **Record** and speak → press **Predict** → the detected language (with a flag image) and speaker appear with confidence scores. Press **Restart** to reset.

## Tech Stack

- **Python 3** with `librosa`, `numpy`, `scipy`, `soundfile` — audio processing & features
- **scikit-learn** — SVM, scaling, CV, metrics
- **Streamlit** — web UI (mic recording via `st.audio_input`)
- **Kivy** — desktop UI
- **joblib** — model persistence

## License

No license specified yet — add one (e.g., MIT) if you want others to reuse the code.
