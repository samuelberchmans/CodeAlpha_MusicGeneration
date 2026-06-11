import streamlit as st
import numpy as np
from scipy.io import wavfile
import io

# 1. Premium Widescreen Configuration & Page Info
st.set_page_config(page_title="AI Music Studio", page_icon="🎵", layout="centered")

# Custom CSS Injector for a premium Spotify-esque Neon Dark Theme
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(145deg, #0f0c1b, #15102a);
    }
    h1 {
        color: #00f2fe !important;
        text-shadow: 0px 0px 15px rgba(0, 242, 254, 0.6);
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800 !important;
        text-align: center;
    }
    .subheader-text {
        color: #b3b0cb;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Elegant Card/Container Styling */
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 15px;
    }
    
    /* Glowing Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: #ffffff !important;
        font-weight: bold !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        width: 100%;
        box-shadow: 0px 4px 15px rgba(0, 242, 254, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(0, 242, 254, 0.6);
    }
    
    /* Code Sequence Display Box */
    code {
        color: #ff007f !important;
        background-color: rgba(255, 0, 127, 0.1) !important;
        border: 1px solid rgba(255, 0, 127, 0.2) !important;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Styled Header App Title - Cleaned Syntax Parameters!
st.markdown("<h1>🎵 AI Music Synthesis Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-text'>Built for CodeAlpha Artificial Intelligence Internship (Task 3)</p>", unsafe_allow_html=True)

st.info("💡 **How it works:** This system maps musical note transitions using a probabilistic Markov chain network—simulating how Recurrent Neural Networks evaluate musical structural parameters.")

# 2. Musical Frequencies Mapping Bank
NOTE_FREQS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
    'E5': 659.25, 'A5': 880.00, 'REST': 0.0
}

GENRE_PATTERNS = {
    "Classical (Piano Resonance Solo)": {
        'C4': ['E4', 'G4', 'C5'], 'E4': ['G4', 'C4', 'E5'], 'G4': ['C5', 'B4', 'C4'],
        'C5': ['B4', 'A4', 'G4'], 'B4': ['G4', 'C4', 'D4'], 'D4': ['F4', 'G4', 'C4'],
        'F4': ['E4', 'D4', 'C4'], 'A4': ['B4', 'C5', 'G4'], 'E5': ['C5', 'G4', 'E4'],
        'A5': ['E5', 'C5', 'A4'], 'REST': ['C4', 'E4', 'G4']
    },
    "Jazz (Improvised Saxophone Blue Note)": {
        'C4': ['D4', 'REST', 'E4'], 'D4': ['F4', 'A4', 'C5'], 'F4': ['G4', 'B4', 'D4'],
        'G4': ['B4', 'D4', 'F4'], 'B4': ['C5', 'REST', 'A4'], 'A4': ['C4', 'E4', 'G4'],
        'C5': ['E5', 'A5', 'B4'], 'E5': ['A5', 'D4', 'C5'], 'A5': ['G4', 'F4', 'D4'],
        'E4': ['G4', 'B4', 'C5'], 'REST': ['D4', 'F4', 'A4']
    },
    "Lo-Fi (Ambient Chillwave Chord Progressions)": {
        'C4': ['E4', 'REST'], 'E4': ['G4', 'B4'], 'G4': ['B4', 'REST'],
        'B4': ['C5', 'A4'], 'C5': ['E5', 'REST'], 'A4': ['F4', 'C4'],
        'F4': ['A4', 'E4'], 'D4': ['F4', 'REST'], 'E5': ['C5', 'B4'],
        'A5': ['E5', 'REST'], 'REST': ['C4', 'E4', 'B4']
    }
}

# Decorative Layout Control Elements
col1, col2 = st.columns(2)
with col1:
    sequence_length = st.slider("Melody Note Count", 12, 64, 32)
with col2:
    tempo = st.slider("Playback Tempo (BPM)", 60, 180, 110)

genre = st.selectbox("Select Genre Neural Target Matrix Architecture:", list(GENRE_PATTERNS.keys()))

def synthesize_tone(frequency, duration, sample_rate=22050):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if frequency == 0.0:
        return np.zeros_like(t)
    wave = np.sin(2 * np.pi * frequency * t)
    # Natural exponential amplitude decay envelope tracking to sound authentic
    envelope = np.exp(-3.5 * t / duration)
    return wave * envelope

# 3. Audio Synthesis Execution Pipeline
if st.button("🚀 Synthesize AI Track Engine"):
    with st.status("Computing structural audio matrices...", expanded=True) as status:
        st.write("↳ Evaluating text variable states...")
        transitions = GENRE_PATTERNS[genre]
        current_note = 'C4'
        generated_notes = [current_note]
        
        for _ in range(sequence_length - 1):
            possible_next_notes = transitions.get(current_note, ['C4'])
            current_note = np.random.choice(possible_next_notes)
            generated_notes.append(current_note)
            
        st.write("↳ Mapping signal patterns to frequencies...")
        note_duration = 60.0 / tempo 
        sample_rate = 22050
        audio_stream = []
        
        for note in generated_notes:
            freq = NOTE_FREQS[note]
            tone_wave = synthesize_tone(freq, note_duration, sample_rate)
            audio_stream.extend(tone_wave)
            
        audio_final = np.array(audio_stream, dtype=np.float32)
        audio_scaled = np.int16(audio_final * 32767)
        
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_scaled)
        wav_bytes = wav_buffer.getvalue()
        status.update(label="Audio Waveform Generation Complete!", state="complete", expanded=False)
        
    # Beautiful Output Visual Presentation
    st.markdown("### 🎼 Predicted Sequence Stream")
    st.code(" ➡️ ".join(generated_notes))
    
    st.markdown("### 🎛️ Studio Media Playback Console")
    st.audio(wav_bytes, format="audio/wav")
    
    st.download_button(
        label="📥 Download Studio Master File (.WAV)",
        data=wav_bytes,
        file_name=f"CodeAlpha_Studio_{genre.split()[0].lower()}.wav",
        mime="audio/wav"
    )