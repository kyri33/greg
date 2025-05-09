# Always-On Voice Assistant with Context Window

This is a proof-of-concept (POC) implementation of an always-on voice assistant that:

1. Continuously listens to and transcribes audio in real-time
2. Maintains a buffer of transcribed text
3. Detects a wake word ("Rachel" by default)
4. Extracts a context window around the wake word
5. Sends only the relevant context to an LLM to generate a response
6. Speaks the response back using text-to-speech

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

3. Run the assistant:

```bash
python main.py
```

## How It Works

- The assistant continuously records audio and periodically sends chunks to be transcribed
- Transcribed text segments are added to a rolling buffer
- When the wake word "rachel" is detected in any segment, the assistant:
  - Extracts a window of text from before and after the wake word
  - Sends this context to the OpenAI API
  - Processes the response
  - Speaks back using text-to-speech

## Configuration

You can modify these settings in `main.py`:

- `WAKE_WORD`: The wake word to listen for (default: "rachel")
- `BUFFER_DURATION_SECONDS`: How many seconds of audio/text to keep in the buffer
- `TRANSCRIPTION_INTERVAL`: How often to transcribe (in seconds)
- `CONTEXT_WINDOW_BEFORE`/`CONTEXT_WINDOW_AFTER`: How many text segments to include before/after the wake word
- `MODEL_SIZE`: Whisper model size (tiny, base, small, medium, large) - smaller is faster but less accurate

## Requirements

- Python 3.8+
- OpenAI API key
- A microphone
- Speakers for text-to-speech output

## Troubleshooting

- If you encounter issues with PyAudio installation, you may need to install portaudio first:
  - On macOS: `brew install portaudio`
  - On Ubuntu: `sudo apt-get install python3-pyaudio`
- For faster transcription, ensure you have a GPU available and configure faster-whisper accordingly 