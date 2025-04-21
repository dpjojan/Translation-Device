# PINOUT:
# Button 1 (Power/Activate): GPIO 2
# Button 2 (Language Cycle): GPIO 3
# Button 3 (Mode Cycle): GPIO 4
# (Connect other side of buttons to a Ground pin)

import argparse
import queue
import sys
import sounddevice as sd
import requests
import uuid
import json
import azure.cognitiveservices.speech as speechsdk
import cv2
import pytesseract
from PIL import Image
from vosk import Model, KaldiRecognizer
import threading
import time
import os
import numpy # Required by sounddevice

# Import gpiozero for button handling and signal for pausing
try:
    from gpiozero import Button
    from signal import pause
    print("gpiozero library loaded successfully.")
    HAS_GPIOZERO = True
except ImportError:
    print("WARN: gpiozero library not found or not running on a compatible device (like Raspberry Pi).")
    HAS_GPIOZERO = False
    # Fallback: Use keyboard if gpiozero fails
    try:
        import keyboard
        HAS_KEYBOARD_FALLBACK = True
        print("Attempting keyboard input simulation as fallback.")
    except ImportError:
        HAS_KEYBOARD_FALLBACK = False
        print("ERROR: Keyboard library also not found. No input method available.")
        sys.exit(1)
    Button = None # Ensure Button is None if gpiozero is unavailable
    if 'pause' not in locals():
        pause = lambda: time.sleep(3600) # Crude pause

# ================== CONFIGURATION ==================

# --- GPIO Pin Configuration ---
BUTTON_1_PIN = 2
BUTTON_2_PIN = 3
BUTTON_3_PIN = 4

# --- Keyboard Fallback Configuration ---
BUTTON_1_KEY = 'p'
BUTTON_2_KEY = 'l'
BUTTON_3_KEY = 'm'

# --- Essential Settings ---
# !!! IMPORTANT: Replace placeholders with your actual keys! Consider using environment variables. !!!
AZURE_TRANSLATE_KEY = # Replace with your Translator Key
AZURE_TRANSLATE_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
AZURE_TRANSLATE_REGION = "eastus" # Replace with your Translator service region

AZURE_SPEECH_KEY = # Replace with your Speech Key
AZURE_SPEECH_REGION = "eastus" # Replace with your Speech service region

# --- Tesseract OCR Configuration ---
TESSERACT_CMD_PATH = '/usr/bin/tesseract' # <<< ADJUST THIS PATH
try:
    if 'pytesseract' in sys.modules:
         pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
except Exception as e:
    print(f"WARN: Could not set Tesseract path at import time: {e}")

# --- Language and Voice Configuration ---
# Format: (Azure Source Lang, Azure Target Lang, Vosk Lang Code, Azure TTS Voice Name)
# Ensure the 'Vosk Lang Code' (3rd item) matches a model Vosk can find via Model(lang=...).
LANGUAGE_CONFIG = [
    ('en', 'es', 'en-us', 'es-ES-ElviraNeural'),   # Listen for English -> Translate to Spanish
    ('es', 'en', 'es', 'en-US-JennyMultilingualNeural'),  # Listen for Spanish -> Translate to English
    ('en', 'fr', 'en-us', 'fr-FR-DeniseNeural'),   # Listen for English -> Translate to French
    ('fr', 'en', 'fr', 'en-US-JennyMultilingualNeural'),  # Listen for French -> Translate to English
    ('fr', 'es', 'fr', 'es-ES-ElviraNeural'),     # Listen for French -> Translate to Spanish
    ('es', 'fr', 'es', 'fr-FR-DeniseNeural'),     # Listen for Spanish -> Translate to French
]

# --- Audio Settings ---
INPUT_DEVICE = None
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

# ================== GLOBAL STATE ==================
q = queue.Queue()
current_language_index = 0
current_mode = "speech"
is_running = False
speech_thread = None
vosk_model = None
vosk_recognizer = None
stop_speech_thread = threading.Event()
button1 = None
button2 = None
button3 = None
callback_lock = threading.Lock() # Lock for button callbacks

# ================== AZURE & OCR FUNCTIONS ==================
# (Translation, TTS, OCR functions remain unchanged)
def translate_text_azure(text, source_lang, target_lang):
    """Translates text using Azure Translator."""
    if not text or not AZURE_TRANSLATE_KEY:
        print("WARN: Empty text or missing Azure Translate Key for translation.")
        return text

    path = '/translate'
    constructed_url = AZURE_TRANSLATE_ENDPOINT + path
    params = {'api-version': '3.0', 'to': [target_lang]}
    if source_lang:
        params['from'] = source_lang

    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATE_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_TRANSLATE_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    result = None

    try:
        response = requests.post(constructed_url, params=params, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result and isinstance(result, list) and result[0].get("translations"):
            translated = result[0]["translations"][0]["text"]
            detected_lang_info = result[0].get("detectedLanguage", {})
            detected_lang = detected_lang_info.get("language", source_lang or "auto")
            print(f"Translation: '{text}' ({detected_lang}) -> '{translated}' ({target_lang})")
            return translated
        else:
            print(f"WARN: Unexpected translation response format: {result}")
            return text
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Azure Translation request failed: {e}")
    except (KeyError, IndexError, TypeError) as e:
        response_content = result if result is not None else "N/A"
        print(f"ERROR: Could not parse translation response: {e} - Response: {response_content}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during translation: {e}")
    return text

def speak_text_azure(text, voice_name):
    """Synthesizes text to speech using Azure TTS."""
    if not text or not AZURE_SPEECH_KEY:
        print("WARN: Empty text or missing Azure Speech Key for synthesis.")
        return
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = voice_name
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            pass
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
    except Exception as e:
        print(f"ERROR: Azure Speech synthesis failed: {e}")

def get_text_from_image(image_path):
    """Extracts text from an image using Tesseract OCR."""
    global TESSERACT_CMD_PATH
    try:
        if 'pytesseract' in sys.modules and pytesseract.pytesseract.tesseract_cmd != TESSERACT_CMD_PATH:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    except Exception as e_set:
         print(f"WARN: Could not set Tesseract path: {e_set}")
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img).strip()
        print(f"OCR Extracted Text: '{text}'")
        return text
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
    except pytesseract.TesseractNotFoundError:
         print(f"ERROR: Tesseract executable not found at '{TESSERACT_CMD_PATH}'. Is it installed and path correct?")
    except Exception as e:
        print(f"ERROR: OCR failed: {e}")
    return ""

# ================== AUDIO PROCESSING ==================
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    if is_running and current_mode == "speech":
        q.put(bytes(indata))

def speech_recognition_loop():
    """Handles the real-time speech recognition and translation."""
    global vosk_recognizer, is_running
    print("Speech recognition thread started.")
    stream = None
    try:
        if vosk_recognizer is None:
            raise RuntimeError("Vosk recognizer not ready.")

        stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=INPUT_DEVICE,
                               dtype='int16', channels=1, callback=audio_callback)
        print(f"\n--- Microphone stream opened (Device: {stream.device}). Speak now! ---")
        with stream:
            while not stop_speech_thread.is_set():
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    if stop_speech_thread.is_set(): break
                    continue

                if vosk_recognizer and vosk_recognizer.AcceptWaveform(data):
                    result_json = vosk_recognizer.Result()
                    try:
                        result = json.loads(result_json)
                        spoken_text = result.get("text", "")
                        if spoken_text:
                            print(f"You said: {spoken_text}")
                            azure_source_lang, target_lang, _, tts_voice = LANGUAGE_CONFIG[current_language_index]
                            translated_text = translate_text_azure(spoken_text, azure_source_lang, target_lang)
                            if translated_text and translated_text.lower() != spoken_text.lower():
                                speak_text_azure(translated_text, tts_voice)
                            elif translated_text:
                                print("(Translation resulted in the same text)")
                            else:
                                 print("(Translation failed or returned empty)")
                    except json.JSONDecodeError:
                        print(f"WARN: Could not decode Vosk result: {result_json}")

    except RuntimeError as e: print(f"ERROR: {e}")
    except sd.PortAudioError as pae: print(f"ERROR: Sounddevice PortAudioError: {pae}")
    except Exception as e: print(f"ERROR in speech recognition loop: {type(e).__name__}: {e}")
    finally:
        if stream and not stream.closed:
            print("Closing microphone stream...")
            stream.close()
        print("Speech recognition thread finished.")
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break

# ================== MODE FUNCTIONS ==================
def run_visual_mode():
    """Captures video stream, detects text for a timed interval,
    sends detected text to Azure for translation and speaking.
    """
    global is_running
    print("\n--- Activating Visual Mode (Timed Detection) ---")
    _, target_lang, _, tts_voice = LANGUAGE_CONFIG[current_language_index]
    print(f"Current Language Target: {target_lang} | Voice: {tts_voice}")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        is_running = False
        return

    # Attempt to set a high resolution (prioritizing video resolution)
    high_width = 1920
    high_height = 1080
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, high_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, high_height)

    current_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if current_width != high_width or current_height != high_height:
        print(f"Warning: Could not set resolution to {high_width}x{high_height}.")
        print(f"Using resolution: {current_width}x{current_height}")
    else:
        print(f"Camera resolution set to: {current_width}x{current_height}")

    text_buffer = []
    start_detection_time = time.time()
    detection_interval = 2  # Detect text for this many seconds

    try:
        while is_running and current_mode == "visual":
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray).strip()
                if text:
                    text_buffer.append(text)
                    print(f"Detected (in buffer): {text}")

            except pytesseract.TesseractNotFoundError:
                print("Error: Tesseract is not installed or not in your PATH.")
                break
            except Exception as e:
                print(f"Error during OCR: {e}")

            cv2.imshow("Timed Text Detection", frame)

            current_time = time.time()
            if current_time - start_detection_time >= detection_interval:
                final_detected_text = " ".join(text_buffer)
                if final_detected_text:
                    print(f"\n--- Sending Detected Text (after {detection_interval} seconds) ---")
                    print(final_detected_text)
                    translated_text = translate_text_azure(final_detected_text, source_lang=None, target_lang=target_lang)
                    if translated_text:
                        print(f"Translated Text: {translated_text}")
                        speak_text_azure(translated_text, tts_voice)
                    else:
                        print("Translation failed.")
                text_buffer = []
                start_detection_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'): # Allow quitting the display window
                print("Stopping visual mode via 'q' key.")
                is_running = False
                break

            time.sleep(0.01) # Small delay to reduce CPU usage

    except Exception as e:
        print(f"Error in run_visual_mode: {e}")
    finally:
        if camera.isOpened():
            camera.release()
            cv2.destroyAllWindows()
        print("--- Visual Mode Finished ---")


# == MODIFIED Function ==
def start_speech_mode():
    """Loads model using Model(lang=...) and starts the speech recognition thread."""
    global speech_thread, vosk_model, vosk_recognizer, is_running
    if speech_thread and speech_thread.is_alive():
        print("WARN: Speech mode already running.")
        return False # Indicate failure

    print("\n--- Preparing Speech Mode ---")
    # Retrieve the Vosk language code from the config
    azure_source_lang, target_lang, vosk_lang_code, tts_voice = LANGUAGE_CONFIG[current_language_index]
    print(f"Using Language: {azure_source_lang} -> {target_lang} | Vosk Lang Code: '{vosk_lang_code}' | TTS: {tts_voice}")

    # --- Load Vosk Model using lang code ---
    try:
        # Check if model needs reloading (based on lang code change)
        # This assumes vosk_model object might store the lang code used to init it.
        # If not, this check might not prevent reloading unnecessarily, but loading is idempotent.
        current_model_lang = getattr(vosk_model, 'lang', None) if vosk_model else None
        if current_model_lang != vosk_lang_code:
            print(f"Loading Vosk model for lang='{vosk_lang_code}'...")
            # *** This is the key change: Using lang parameter ***
            vosk_model = Model(lang=vosk_lang_code)
            # Store lang code for comparison next time (optional)
            setattr(vosk_model, 'lang', vosk_lang_code)
            # Need to recreate recognizer whenever model changes
            vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
            print("Vosk model and recognizer loaded/updated.")
        elif vosk_recognizer is None: # Model okay, but no recognizer
             vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
             print("Vosk recognizer created.")
        # else: Model and recognizer already loaded and match current lang code

    except Exception as e:
        # This error often means Vosk couldn't find a model for the lang code in its search paths
        print(f"ERROR: Failed to load Vosk model for lang='{vosk_lang_code}': {e}")
        print("Ensure a Vosk model for this language code is installed where Vosk can find it.")
        print("This script version does NOT look in a local 'models' folder.")
        vosk_model = None
        vosk_recognizer = None
        return False # Indicate failure

    # --- Start Thread ---
    stop_speech_thread.clear()
    speech_thread = threading.Thread(target=speech_recognition_loop, daemon=True)
    speech_thread.start()
    return True # Indicate success

def stop_current_mode():
    """Stops the currently active mode (specifically the speech thread)."""
    global is_running, speech_thread
    stopped = False
    if current_mode == "speech" and speech_thread and speech_thread.is_alive():
        print("Stopping speech recognition thread...")
        stop_speech_thread.set()
        q.put(b'\0'*1024)
        speech_thread.join(timeout=2.5)
        if speech_thread.is_alive(): print("WARN: Speech thread did not stop gracefully.")
        speech_thread = None
        print("Speech thread stopped.")
        stopped = True
    is_running = False
    return stopped

# ================== BUTTON CALLBACKS ==================
# (Callbacks remain unchanged)
def on_button1_press(): # Power / Activate
    global is_running
    with callback_lock:
        current_state_running = is_running
        print(f"\n[Button 1] State: {'Running' if current_state_running else 'Stopped'}")
        if current_state_running:
            print("Deactivating...")
            stop_current_mode()
        else:
            print("Activating...")
            is_running = True
            success = False
            if current_mode == "speech":
                success = start_speech_mode()
            elif current_mode == "visual":
                print("Starting visual sequence in background thread...")
                visual_thread = threading.Thread(target=run_visual_mode, daemon=True)
                visual_thread.start()
                success = True # Assume thread start is success
            else:
                 print(f"ERROR: Unknown mode '{current_mode}'")
                 success = False
            if not success:
                 print("Activation failed immediately.")
                 is_running = False

def on_button2_press(): # Language Cycle
    global current_language_index
    with callback_lock:
        current_state_running = is_running
        current_language_index = (current_language_index + 1) % len(LANGUAGE_CONFIG)
        azure_source_lang, target_lang, vosk_lang_code, tts_voice = LANGUAGE_CONFIG[current_language_index]
        print(f"\n[Button 2] Language set to: {azure_source_lang} -> {target_lang}")
        print(f"  (Vosk Lang: '{vosk_lang_code}' | TTS: {tts_voice})")
        if current_state_running: print(f"  (Stop and restart with Button 1 for changes to take effect)")
        else: print(f"  (Effective on next activation)")

def on_button3_press(): # Mode Cycle
    global current_mode
    with callback_lock:
        current_state_running = is_running
        if current_state_running:
             print(f"\n[Button 3] INFO: Stop the system (Button 1) before changing mode.")
             return
        previous_mode = current_mode
        if current_mode == "speech": current_mode = "visual"
        elif current_mode == "visual": current_mode = "speech"
        else: current_mode = "speech"
        print(f"\n[Button 3] Mode changed from {previous_mode.upper()} to: {current_mode.upper()}")
        print(f"  (Effective on next activation)")

# ================== MAIN EXECUTION ==================
# == MODIFIED Function ==
def print_initial_state():
    print("\n" + "="*50)
    print(" Translator Control Application")
    print("="*50)
    input_method = "GPIO" if HAS_GPIOZERO and Button else "Keyboard Fallback" if HAS_KEYBOARD_FALLBACK else "None"
    print(f"Input Method: {input_method}")

    if input_method == "GPIO":
        print(f"  Button 1 (GPIO {BUTTON_1_PIN}): Activate / Deactivate")
        print(f"  Button 2 (GPIO {BUTTON_2_PIN}): Cycle Language")
        print(f"  Button 3 (GPIO {BUTTON_3_PIN}): Cycle Mode")
    elif input_method == "Keyboard Fallback":
         print(f"  Button 1 ({BUTTON_1_KEY.upper()}): Activate / Deactivate")
         print(f"  Button 2 ({BUTTON_2_KEY.upper()}): Cycle Language")
         print(f"  Button 3 ({BUTTON_3_KEY.upper()}): Cycle Mode")

    print("\nInitial State:")
    azure_source_lang, target_lang, vosk_lang_code, tts_voice = LANGUAGE_CONFIG[current_language_index]
    print(f"  Mode       : {current_mode.upper()}")
    print(f"  Language   : {azure_source_lang} -> {target_lang}")
    # Print Vosk Lang Code instead of model name/path
    print(f"  Vosk Lang  : '{vosk_lang_code}' (Ensure model is installed for Vosk)")
    print(f"  TTS Voice  : {tts_voice}")
    print("\nWaiting for activation...")
    print("Press Ctrl+C in the terminal to exit.")
    print("="*50)

def check_audio_devices():
    # (This function remains unchanged)
    print("\n--- Checking Audio Devices ---")
    try:
        print(sd.query_devices())
        default_input = sd.query_devices(kind='input')
        if default_input: print(f"\nDefault Input Device: {default_input['name']}")
        else: print("\nWARN: No default input device found by sounddevice.")
    except Exception as e: print(f"ERROR: Could not query audio devices: {e}")
    print("----------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech/Visual Translation Tool")
    parser.add_argument("-d", "--device", type=str, help="Input audio device name substring or ID")
    parser.add_argument("-r", "--samplerate", type=int, default=SAMPLE_RATE, help=f"Sampling rate (default: {SAMPLE_RATE})")
    parser.add_argument("-l", "--list-devices", action="store_true", help="Show list of audio devices and exit")
    parser.add_argument("--no-gpio", action="store_true", help="Force disable GPIO and use keyboard fallback")
    args = parser.parse_args()

    if args.list_devices: print(sd.query_devices()); sys.exit(0)
    check_audio_devices()
    if args.device: INPUT_DEVICE = args.device; print(f"Using specified input device hint: {INPUT_DEVICE}")
    SAMPLE_RATE = args.samplerate

    # --- Setup Input Method ---
    # (This section remains unchanged)
    input_method_active = False
    use_gpio = HAS_GPIOZERO and not args.no_gpio
    if use_gpio:
        print("Attempting to initialize GPIO buttons...")
        try:
            button1 = Button(BUTTON_1_PIN, pull_up=True, bounce_time=0.05)
            button2 = Button(BUTTON_2_PIN, pull_up=True, bounce_time=0.05)
            button3 = Button(BUTTON_3_PIN, pull_up=True, bounce_time=0.05)
            button1.when_released = on_button1_press
            button2.when_released = on_button2_press
            button3.when_released = on_button3_press
            print(f"GPIO Buttons initialized on pins {BUTTON_1_PIN}, {BUTTON_2_PIN}, {BUTTON_3_PIN}.")
            input_method_active = True
        except Exception as e:
            print(f"ERROR: Failed to initialize GPIO buttons: {e}")
            if HAS_KEYBOARD_FALLBACK: print("Attempting keyboard fallback..."); use_gpio = False
            else: print("No fallback available. Exiting."); sys.exit(1)

    if not use_gpio and HAS_KEYBOARD_FALLBACK:
        print(f"Using keyboard fallback (Keys: {BUTTON_1_KEY.upper()}, {BUTTON_2_KEY.upper()}, {BUTTON_3_KEY.upper()}).")
        print("NOTE: Keyboard listener might require root/admin privileges.")
        try:
            keyboard.add_hotkey(BUTTON_1_KEY, on_button1_press, trigger_on_release=True)
            keyboard.add_hotkey(BUTTON_2_KEY, on_button2_press, trigger_on_release=True)
            keyboard.add_hotkey(BUTTON_3_KEY, on_button3_press, trigger_on_release=True)
            input_method_active = True
        except Exception as e:
             print(f"ERROR: Failed to set up keyboard hotkeys: {e}")
             input_method_active = False

    if not input_method_active:
         print("\nERROR: No input method (GPIO or Keyboard) could be initialized. Exiting.")
         sys.exit(1)

    # --- Main Application Loop ---
    # (This section remains unchanged)
    print_initial_state()
    try:
        if use_gpio:
            print("System ready. Waiting for button presses (using GPIO).")
            pause()
        else:
            print("System ready. Waiting for key presses (using Keyboard).")
            while True: time.sleep(60)
    except KeyboardInterrupt: print("\nCtrl+C detected. Exiting...")
    except Exception as e: print(f"\nAn unexpected error occurred in the main loop: {e}")
    finally:
        print("Cleaning up...")
        if is_running: stop_current_mode()
        if not use_gpio and HAS_KEYBOARD_FALLBACK:
            try: keyboard.unhook_all(); print("Keyboard hooks removed.")
            except Exception as e: print(f"Error unhooking keyboard: {e}")
        print("Exited.")



