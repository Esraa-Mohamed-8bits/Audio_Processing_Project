import io
import joblib
import librosa
import numpy as np
import serial
import streamlit as st
from streamlit_mic_recorder import mic_recorder

#################################
## socat -d -d pty,raw,echo=0 pty,raw,echo=0
## this is the command to create a virtual serial port for testing with SimulIDE
#################################

st.set_page_config(page_title="Voice Control", page_icon="🎤")
st.title("Voice Command System")

@st.cache_resource
def load_models():
    cmd_model = joblib.load('command_model.pkl')
    per_model = joblib.load('person_model.pkl')
    scaler = joblib.load('scaler.pkl')
    try:
        arduino = serial.Serial('/dev/pts/1', 9600, timeout=1)
    except:
        arduino = None
    return cmd_model, per_model, scaler, arduino

cmd_model, per_model, scaler, arduino = load_models()

if "last_command" not in st.session_state:
    st.session_state.last_command = None

if arduino:
    st.success("Connected to Virtual Arduino")
else:
    st.warning("Arduino not connected. Screen-Only Mode.")

st.write("### Choose Input Method:")

audio = mic_recorder(start_prompt="Start Recording (Mic)", stop_prompt="Stop Recording", key='recorder')

st.write("---")
st.write("### OR Upload Audio File:")

uploaded_file = st.file_uploader("Upload your audio file (e.g., from Telegram)", type=['wav', 'ogg', 'mp3', 'm4a', 'opus', 'flac'])

audio_buffer = None

if audio:
    audio_bytes = audio['bytes']
    audio_file = io.BytesIO(audio_bytes)
    audio_buffer, sr = librosa.load(audio_file, sr=22050)
    st.success("Audio captured from Mic!")
    
elif uploaded_file is not None:
    audio_buffer, sr = librosa.load(uploaded_file, sr=22050)
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

if audio_buffer is not None:
    rms = np.sqrt(np.mean(audio_buffer**2))
    if rms < 0.005:
        st.error("Silence Detected. Please try again.")
    else:
        trimmed = librosa.effects.trim(audio_buffer, top_db=30)[0]
        mfcc = librosa.feature.mfcc(y=trimmed, sr=22050, n_mfcc=17)
        feat = np.mean(librosa.util.normalize(mfcc), axis=1).reshape(1, -1)

        scaled_feat = scaler.transform(feat)
        command = cmd_model.predict(scaled_feat)[0]
        person = per_model.predict(scaled_feat)[0]
        
        st.metric(label="Predicted Command", value=command)
        st.metric(label="Speaker", value=person)
        
        if arduino:
            if command != st.session_state.last_command:
                if command == "ON":
                    arduino.write(b'1')
                    arduino.flush()
                    st.info("Sent '1' to SimulIDE")
                elif command == "OFF":
                    arduino.write(b'0')
                    arduino.flush()
                    st.info("Sent '0' to SimulIDE")
                
                st.session_state.last_command = command
            else:
                st.info(f"Already {command}. Command ignored to prevent Simulator crash.")