# CodeAlpha_MusicGeneration
A Deep Learning-inspired AI Music Synthesis Studio utilizing probabilistic Markov Chain transitions and raw digital waveform generators built with Streamlit for the CodeAlpha AI Internship (Task 3).
# 🎵 AI Music Synthesis Studio

A production-ready deep learning-inspired sequential audio model developed for the **CodeAlpha Artificial Intelligence Internship** (Task 3).

## 🧠 Core Architecture & Theory
Instead of utilizing high-compute external APIs, this tool builds a locally processed digital signal synthesizer:
* **Markov Chain Sequence Modeling:** Mimics basic Recurrent Neural Network (RNN) behavior by utilizing transition probability states to logically predict the next sequential note based on genre matrix models.
* **Audio Waveform Synthesis:** Mathematically maps musical notation characters into raw scientific frequencies (Hz) using `numpy` sine-wave structures.
* **Exponential Decay Enveloping:** Implements standard audio fade curves via `scipy` to eliminate clicking artifacts and create a smooth listening experience.

## 💻 Installation & Execution
```bash
pip install streamlit numpy scipy
streamlit run app.py
