import argparse
import queue
import sys
import sounddevice as sd
import requests, uuid, json
import azure.cognitiveservices.speech as speechsdk

from vosk import Model, KaldiRecognizer

# Queues
q = queue.Queue()

# Azure config (you can move these to env variables later for safety)
translate_key = 'Input Key'
translate_endpoint = "https://api.cognitive.microsofttranslator.com"
translate_region = "eastus"

speech_key = "inputkey"
speech_region = "eastus"

# Translation Function
def pull_to_azure(text):
    path = '/translate'
    constructed_url = translate_endpoint + path
    params = {
        'api-version': '3.0',
        'from': ['en'],
        'to': ['es']
    }
    headers = {
        'Ocp-Apim-Subscription-Key': translate_key,
        'Ocp-Apim-Subscription-Region': translate_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    response = requests.post(constructed_url, params=params, headers=headers, json=body)
    result = response.json()
    return result[0]["translations"][0]["text"]

# Text-to-Speech Function
def speak_text(text):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = 'en-US-AvaMultilingualNeural'
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(text).get()

# Audio Callback
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Argparse helpers
def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

# CLI setup
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-l", "--list-devices", action="store_true", help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument("-f", "--filename", type=str, metavar="FILENAME", help="audio file to store recording to")
parser.add_argument("-d", "--device", type=int_or_str, help="input device (numeric ID or substring)")
parser.add_argument("-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument("-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
args = parser.parse_args(remaining)

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        args.samplerate = int(device_info["default_samplerate"])

    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None

    with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
                           dtype="int16", channels=1, callback=callback):
        print("#" * 80)
        print("Speak into the mic. Press Ctrl+C to stop.")
        print("#" * 80)

        rec = KaldiRecognizer(model, args.samplerate)

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                spoken_text = result.get("text", "")
                if spoken_text:
                    print(f"You said: {spoken_text}")
                    translated = pull_to_azure(spoken_text)
                    print(f"Translated: {translated}")
                    speak_text(translated)
            else:
                print(rec.PartialResult())

            if dump_fn is not None:
                dump_fn.write(data)

except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))
