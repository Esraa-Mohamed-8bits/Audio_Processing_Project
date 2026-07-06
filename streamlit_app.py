"""
Voice Command & Speaker Identification System — Streamlit App
IEEE SSSC AI Team — First Project (Bonus: real-time mic + GUI)

Run with:
    streamlit run streamlit_app.py

Requires the trained artifacts in the same folder:
    command_model.pkl, person_model.pkl, scaler.pkl
"""

import io
import tempfile
import time

import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st

# Optional deps — app still runs without them, with reduced features
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_MIC = True
except ImportError:
    HAS_MIC = False

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Voice Command & Speaker ID", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Command & Speaker Identification System")
st.caption("IEEE SSSC — AI Team First Project · Classical ML (Random Forest) · Arduino Integration")

N_MFCC = 17
SAMPLE_RATE = 22050


# --------------------------------------------------------------------------
# Load models (cached so they load once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    command_model = joblib.load("command_model.pkl")
    person_model = joblib.load("person_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return command_model, person_model, scaler


try:
    model_command, model_person, scaler = load_models()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `command_model.pkl`, `person_model.pkl`, "
        "and `scaler.pkl` are in the same folder as this app."
    )
    st.stop()


# --------------------------------------------------------------------------
# Feature extraction — mirrors process.ipynb exactly
# --------------------------------------------------------------------------
def extract_features(audio_bytes: bytes) -> np.ndarray:
    # Write to a temp file so librosa/audioread can handle wav/ogg/flac alike
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        y, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)

    removed_silence, _ = librosa.effects.trim(y, top_db=30)
    if len(removed_silence) == 0:
        removed_silence = y  # fallback if trimming removed everything

    mfcc = librosa.feature.mfcc(y=removed_silence, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_norm = librosa.util.normalize(mfcc)
    mfcc_flat = np.mean(mfcc_norm, axis=1).reshape(1, -1)
    return mfcc_flat


def predict(audio_bytes: bytes):
    features = extract_features(audio_bytes)
    scaled = scaler.transform(features)

    command_pred = model_command.predict(scaled)[0]
    person_pred = model_person.predict(scaled)[0]

    command_conf = dict(zip(model_command.classes_, model_command.predict_proba(scaled)[0]))
    person_conf = dict(zip(model_person.classes_, model_person.predict_proba(scaled)[0]))

    return command_pred, person_pred, command_conf, person_conf


# --------------------------------------------------------------------------
# Serial connection to Arduino (sidebar)
# --------------------------------------------------------------------------
st.sidebar.header("🔌 Arduino Connection")

if not HAS_SERIAL:
    st.sidebar.warning("Install `pyserial` to enable the Arduino link:\n\n`pip install pyserial`")
    ser_conn = None
else:
    ports = [p.device for p in serial.tools.list_ports.comports()]
    selected_port = st.sidebar.text_input("Serial Port (e.g. /dev/pts/1)", value="/dev/pts/2")
    baud = st.sidebar.number_input("Baud rate", value=9600, step=1)

    if "serial_conn" not in st.session_state:
        st.session_state.serial_conn = None

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Connect"):
        if selected_port != "(none)":
            try:
                st.session_state.serial_conn = serial.Serial(selected_port, baud, timeout=1)
                time.sleep(2)  # allow Arduino to reset after opening the port
                st.sidebar.success(f"Connected to {selected_port}")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {e}")
        else:
            st.sidebar.warning("Select a port first.")

    if col_b.button("Disconnect"):
        if st.session_state.serial_conn:
            st.session_state.serial_conn.close()
            st.session_state.serial_conn = None
        st.sidebar.info("Disconnected.")

    ser_conn = st.session_state.serial_conn
    if ser_conn:
        st.sidebar.success("🟢 Arduino connected")
    else:
        st.sidebar.info("⚪ Not connected (predictions still work without Arduino)")


if "last_command" not in st.session_state:
    st.session_state.last_command = None

def send_to_arduino(person: str, command: str):
    if HAS_SERIAL and st.session_state.get("serial_conn"):
        if command == st.session_state.last_command:
            return True
        try:
            message = f"{person},{command}\n"
            st.session_state.serial_conn.write(message.encode())
            st.session_state.serial_conn.flush()
            st.session_state.last_command = command
            return True
        except Exception as e:
            st.sidebar.error(f"Send failed: {e}")
    return False


# --------------------------------------------------------------------------
# History log
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def log_prediction(source, person, command, person_conf, command_conf):
    st.session_state.history.insert(0, {
        "Time": time.strftime("%H:%M:%S"),
        "Source": source,
        "Speaker": person,
        "Command": command,
        "Speaker Confidence": f"{person_conf[person]*100:.1f}%",
        "Command Confidence": f"{command_conf[command]*100:.1f}%",
    })


def show_result(audio_bytes, source_label):
    with st.spinner("Extracting features and predicting..."):
        command_pred, person_pred, command_conf, person_conf = predict(audio_bytes)

    col1, col2 = st.columns(2)
    col1.metric("🗣️ Speaker", person_pred, f"{person_conf[person_pred]*100:.1f}% confidence")
    col2.metric("🔊 Command", command_pred, f"{command_conf[command_pred]*100:.1f}% confidence")

    with st.expander("See full confidence breakdown"):
        st.write("**Speaker probabilities**")
        st.bar_chart(pd.Series(person_conf))
        st.write("**Command probabilities**")
        st.bar_chart(pd.Series(command_conf))

    sent = send_to_arduino(person_pred, command_pred)
    if sent:
        st.success(f"Sent to Arduino → {person_pred},{command_pred}")
    elif HAS_SERIAL and st.session_state.get("serial_conn") is None:
        st.info("Arduino not connected — connect it from the sidebar to drive the LCD/LED.")

    log_prediction(source_label, person_pred, command_pred, person_conf, command_conf)


# --------------------------------------------------------------------------
# Tabs: Live Mic (bonus) / Upload File
# --------------------------------------------------------------------------
tab_mic, tab_upload = st.tabs(["🎤 Real-time Microphone", "📁 Upload Audio File"])

with tab_mic:
    st.write("Record a short command (e.g. *\"ON\"* or *\"OFF\"*) directly from your browser.")
    if not HAS_MIC:
        st.warning(
            "Install the mic-recording component to enable this tab:\n\n"
            "`pip install audio-recorder-streamlit`"
        )
    else:
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8483a",
            neutral_color="#2b6cb0",
            icon_size="3x",
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            show_result(audio_bytes, "Microphone")

with tab_upload:
    st.write("Upload a `.wav` (or `.ogg`, converted beforehand) audio clip to test the models.")
    uploaded = st.file_uploader("Choose an audio file", type=["wav", "ogg", "flac"])
    if uploaded is not None:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)
        if st.button("Predict"):
            show_result(audio_bytes, uploaded.name)


# --------------------------------------------------------------------------
# History table
# --------------------------------------------------------------------------
st.divider()
st.subheader("📜 Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("No predictions yet — record or upload audio above to get started.")
