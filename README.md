# Audio Transcription and Homophone Detection with Wav2Vec2

This project uses Facebook's pre-trained Wav2Vec2 model to transcribe audio, detect homophones, and visualize word frequency data. It includes functionality for adding synthetic noise to audio files, performing speech-to-text transcription, and visualizing results with bar charts and word clouds.

## 🔧 Features

- Add synthetic noise to an audio file
- Transcribe audio using Wav2Vec2
- Detect common English homophones in the transcript
- Visualize:
  - Top 10 most frequent words
  - Word cloud
  - Homophone matches

## 🧰 Requirements

Install dependencies using pip:

```bash
pip install torch torchaudio librosa numpy soundfile matplotlib wordcloud transformers
input_path = "/content/hardvard.wav"  # Replace with your file
python script.py
├── script.py
├── noisy_output.wav
├── transcription.txt  # (optional if saving output)
└── README.md

---

### 📥 Download

Would you like me to generate this `README.md` file and provide a download link?

If you're in a notebook or local script, I can create the file for you like this:

```python
readme_content = """<paste content above>"""
with open("README.md", "w") as f:
    f.write(readme_content)

