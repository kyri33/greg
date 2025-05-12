import pyaudio
import numpy as np
import threading
import queue
import time
import os
import pyttsx3
from collections import deque
import wave
import tempfile
from openai import OpenAI
import logging
import datetime
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import faster-whisper, fall back to whisper if not available
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
    logger.info("Using faster-whisper for transcription")
except ImportError:
    try:
        import whisper
        USE_FASTER_WHISPER = False
        logger.info("Using whisper for transcription")
    except ImportError:
        raise ImportError("Please install either whisper or faster-whisper: pip install faster-whisper or pip install whisper")

# Configuration
WAKE_WORD = "greg"
SAMPLE_RATE = 16000
BUFFER_DURATION_SECONDS = 5  # How many seconds of audio/text to keep in buffer
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
TRANSCRIPTION_INTERVAL = 1.5  # How often to transcribe (seconds)
CONTEXT_WINDOW_BEFORE = 3  # How many segments to include before wake word
CONTEXT_WINDOW_AFTER = 10   # How many segments to include after wake word
MODEL_SIZE = "tiny"  # Whisper model size: tiny, base, small, medium, large
WAKE_WORD_COOLDOWN = 5  # Seconds to wait after wake word before detecting it again
TRANSCRIPTION_TIMEOUT = 5  # Maximum seconds allowed for transcription

# Initialize OpenAI client if API key exists
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    else:
        logger.warning("OPENAI_API_KEY environment variable not set. LLM features disabled.")
        client = None
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    client = None

class SpeechManager:
    """Thread-safe manager for text-to-speech operations"""
    
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        self.tts_engine = None
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize or reinitialize the TTS engine"""
        try:
            # Close existing engine if it exists
            if self.tts_engine:
                try:
                    self.tts_engine.endLoop()
                except:
                    pass
            
            self.tts_engine = pyttsx3.init()
            logger.info("TTS engine initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            return False
    
    def start_speech_thread(self):
        """Start the speech processing thread if not already running"""
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self._process_speech_queue)
            self.speech_thread.daemon = True
            self.speech_thread.start()
            logger.info("Speech thread started")
    
    def _process_speech_queue(self):
        """Process the speech queue in a dedicated thread"""
        while True:
            try:
                # Get the next text to speak (block until available)
                text = self.speech_queue.get(timeout=1.0)
                
                # Set speaking flag
                self.is_speaking = True
                
                # Speak the text
                logger.info(f"Speaking: {text[:50]}...")
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except RuntimeError as e:
                    logger.error(f"TTS engine error: {e}")
                    traceback.print_exc()
                    # Reinitialize the engine
                    self.initialize_engine()
                    # Try again with the new engine
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                # Mark task as done
                self.speech_queue.task_done()
                
                # Small delay to ensure clean audio
                time.sleep(0.5)
                
            except queue.Empty:
                # No more items to process
                self.is_speaking = False
                return
            except Exception as e:
                logger.error(f"Error in speech thread: {e}")
                traceback.print_exc()
                self.is_speaking = False
                # Reinitialize for next time
                self.initialize_engine()
                return
            finally:
                # Always reset speaking flag when done
                self.is_speaking = False
    
    def speak(self, text):
        """Queue text to be spoken"""
        if not text:
            return
            
        # Add to queue
        self.speech_queue.put(text)
        
        # Ensure thread is running
        self.start_speech_thread()
        
        # Return immediately, speech happens asynchronously
        return True

class VoiceAssistant:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.text_buffer = deque(maxlen=100)  # Stores transcribed text segments
        self.last_timestamps = deque(maxlen=100)  # Timestamps for text segments
        self.audio_buffer = deque(maxlen=int(BUFFER_DURATION_SECONDS * SAMPLE_RATE / CHUNK_SIZE))
        self.is_running = True
        self.is_processing_wake_word = False
        self.last_transcription_time = 0
        self.last_wake_word_time = 0
        self.speaking_mode = False  # Flag to indicate when the assistant is speaking
        self.speaking_lock = threading.Lock()
        
        # Initialize speech manager
        self.speech_manager = SpeechManager()
        
        # Initialize audio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.audio_callback
        )
        
        # Initialize transcription model
        logger.info(f"Loading {MODEL_SIZE} Whisper model...")
        start_time = time.time()
        if USE_FASTER_WHISPER:
            self.model = WhisperModel(MODEL_SIZE, device="cpu")
        else:
            self.model = whisper.load_model(MODEL_SIZE)
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Debug option
        self.debug = True

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream to collect chunks into queue"""
        # Only add audio to buffer if not speaking and not processing a wake word
        if not self.speaking_mode and not self.speech_manager.is_speaking:
            self.audio_buffer.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def save_audio_buffer_to_file(self):
        """Save current audio buffer to a temporary file for transcription"""
        buffer_length = len(self.audio_buffer)
        logger.info(f"Saving audio buffer ({buffer_length} chunks) to file")
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            filename = temp_file.name
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(list(self.audio_buffer)))
        wf.close()
        
        logger.debug(f"Audio saved to file in {time.time() - start_time:.3f} seconds")
        return filename
    
    def transcribe_audio(self):
        """Periodically transcribe the audio buffer"""
        logger.info("Starting transcription thread")
        while self.is_running:
            current_time = time.time()
            
            # Only transcribe when:
            # 1. Enough time has passed since last transcription
            # 2. There's audio to transcribe
            # 3. Not currently speaking
            # 4. Speech manager is not speaking
            if (current_time - self.last_transcription_time >= TRANSCRIPTION_INTERVAL and 
                len(self.audio_buffer) > 0 and 
                not self.speaking_mode and
                not self.speech_manager.is_speaking):
                
                self.last_transcription_time = current_time
                
                # Save audio buffer to file for transcription
                audio_file = self.save_audio_buffer_to_file()
                
                try:
                    # Transcribe the audio with timeout monitoring
                    logger.info("Starting transcription...")
                    transcription_start = time.time()
                    
                    if USE_FASTER_WHISPER:
                        segments, info = self.model.transcribe(audio_file, language="en")
                        segments = list(segments)  # Convert generator to list
                        if info.duration:
                            logger.info(f"Processing audio with duration {datetime.timedelta(seconds=info.duration)}")
                    else:
                        result = self.model.transcribe(audio_file)
                        segments = result["segments"]
                    
                    transcription_time = time.time() - transcription_start
                    logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
                    
                    if transcription_time > TRANSCRIPTION_TIMEOUT:
                        logger.warning(f"Transcription took too long: {transcription_time:.2f}s > {TRANSCRIPTION_TIMEOUT}s timeout")
                    
                    # Process segments and add to text buffer
                    for segment in segments:
                        text = segment.text.strip().lower() if USE_FASTER_WHISPER else segment["text"].strip().lower()
                        if text:
                            self.text_buffer.append(text)
                            self.last_timestamps.append(current_time)
                            
                            logger.info(f"Transcribed: {text}")
                            
                            # Check for wake word in this segment if:
                            # 1. Wake word is in the text
                            # 2. Not already processing a wake word
                            # 3. Enough time has passed since last wake word
                            # 4. Not currently speaking or processing speech
                            current_time = time.time()
                            if (WAKE_WORD in text and 
                                not self.is_processing_wake_word and 
                                current_time - self.last_wake_word_time > WAKE_WORD_COOLDOWN and
                                not self.speech_manager.is_speaking):
                                
                                logger.info(f"Wake word detected in: '{text}'")
                                self.last_wake_word_time = current_time
                                self.process_wake_word(text)
                    
                    # Clean up the temporary file
                    os.unlink(audio_file)
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    traceback.print_exc()
            
            time.sleep(0.1)  # Sleep to prevent CPU overuse
    
    def get_context_window(self):
        """Extract the context window around the wake word"""
        buffer_list = list(self.text_buffer)
        
        # Find the most recent segment containing the wake word
        wake_word_indices = [i for i, text in enumerate(buffer_list) if WAKE_WORD in text]
        
        if not wake_word_indices:
            return ""
        
        # Get the most recent occurrence of the wake word
        wake_word_idx = wake_word_indices[-1]
        
        # Extract context before and after wake word
        start_idx = max(0, wake_word_idx - CONTEXT_WINDOW_BEFORE)
        end_idx = min(len(buffer_list), wake_word_idx + CONTEXT_WINDOW_AFTER + 1)
        
        context_segments = buffer_list[start_idx:end_idx]
        context = " ".join(context_segments)
        logger.info(f"Extracted context window from index {start_idx} to {end_idx}")
        return context
    
    def process_wake_word(self, trigger_text):
        """Process detected wake word by extracting context and querying LLM"""
        if self.is_processing_wake_word or self.speech_manager.is_speaking:
            logger.info("Already processing a wake word or speaking, ignoring this one")
            return
            
        # Set flag to prevent multiple wake word processing at once
        self.is_processing_wake_word = True
        
        try:
            # Extract context around wake word
            context = self.get_context_window()
            
            logger.info("=== WAKE WORD DETECTED ===")
            logger.info(f"Context: {context}")
            
            # Activate speaking mode to prevent hearing our own speech
            with self.speaking_lock:
                self.speaking_mode = True
                # Clear the audio buffer to prevent processing assistant's speech
                self.audio_buffer.clear()
            
            # Send to LLM
            response = self.query_llm(context)
            
            # Speak the response using the speech manager
            logger.info(f"Assistant: {response}")
            self.speech_manager.speak(response)
            
            # Give a short delay before resuming listening
            # This prevents the assistant from hearing its own speech
            time.sleep(1.0)
            
            # Deactivate speaking mode
            with self.speaking_lock:
                self.speaking_mode = False
                # Clear the audio buffer again after speaking
                self.audio_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error processing wake word: {e}")
            traceback.print_exc()
        finally:
            # Reset flag after processing
            self.is_processing_wake_word = False
    
    def query_llm(self, context):
        """Send the context to an LLM and get a response"""
        logger.info("Sending context to LLM")
        
        if client is None:
            logger.warning("OpenAI client not initialized. Using mock response.")
            # Extract what the user asked after the wake word
            try:
                user_query = context.split(WAKE_WORD)[-1].strip()
                if user_query:
                    message = f"I'm sorry, I can't process that request because the OpenAI API key is not configured. You asked about: {user_query}"
                else:
                    message = "I'm sorry, I can't process that request because the OpenAI API key is not configured."
            except:
                message = "I'm sorry, I can't process that request because the OpenAI API key is not configured."
            return message
            
        try:
            query_start = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful voice assistant named {WAKE_WORD.capitalize()}. "
                                                 f"You are being provided with a buffer of voice conversation. "
                                                 f"You are to respond to the user based on the conversation in the buffer. "
                                                 f"The user would have used your name to wake you up. Based on the conversation in the buffer, "
                                                 f"Try to determine what the user wants. "
                                                 f"Due to the nature of the app there will be a lot of noise in the user prompt, "
                                                 f"Most of it should be ignored, it was clearly from another conversation. Using the wake word: '{WAKE_WORD}' try to determine what the user wanted. "
                                                 f"so focus on the text surrounding the wake word. "
                                                 f"If you don't know the answer, just say you don't know. "
                                                 f"Don't make up an answer."},
                    {"role": "user", "content": context}
                ],
                max_tokens=500
            )
            query_time = time.time() - query_start
            logger.info(f"LLM response received in {query_time:.2f} seconds")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            traceback.print_exc()
            return "I'm sorry, I encountered an error processing your request."
    
    def speak(self, text):
        """Legacy method for backward compatibility - use speech_manager instead"""
        return self.speech_manager.speak(text)
    
    def run(self):
        """Start the voice assistant"""
        logger.info(f"Voice Assistant started. Listening for wake word: '{WAKE_WORD}'")
        
        # Start transcription thread
        transcription_thread = threading.Thread(target=self.transcribe_audio)
        transcription_thread.daemon = True
        transcription_thread.start()
        
        try:
            # Keep the main thread alive
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the voice assistant"""
        logger.info("Stopping voice assistant...")
        self.is_running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
