import argparse
import queue
import sys
import sounddevice as sd

from vosk import Model, KaldiRecognizer

q = queue.Queue()


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit()
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
args = parser.parse_args(remaining)

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info["default_samplerate"])

    if args.model is None:
        model = Model(lang="es")
    else:
        model = Model(lang=args.model)

    with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
                           dtype="int16", channels=1, callback=callback):

        print("Press Ctrl+C to stop the recording")

        recording = KaldiRecognizer(model, args.samplerate)
        while True:
            data = q.get()
            if recording.AcceptWaveform(data):
                result = rec.Result()
                print(result)
            import json

            rec_test = json.loads(result).get('text', '')
            if rec_text:
                translated_text = pull_to_azure(rec_test)
                if translated_text:
                    print('Translated', translated_text)
                else:
                    print(rec.PartialResult())
            if dump_fn is not None:
                dump_fn.write(data)


def pull_to_azure:
    pass


def text_to_speech(translated_text):
    import os
    import azure.cognitiveservices.speech as speechsdk
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('Speechkey'),
                                           region=os.environ.get('Speech Region')
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = 'en-US-AvaMultilingualNeural'
    speech_init = speechdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_result = speech_init.speak_text_async(translated_text).get()
    if speech_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print('Speech synthesized for text [{}]'.format(translated_text))
    elif speech_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
    print('Speech synthesis canceled: {}'.format(cancellation_details.reason))
    if cancellation_details.error_details:
        print('Error details: {}'.format(cancellation_details.error_details))
    print('Did you set the speech resrource key and region values?')

except KeyboardInterrupt:
print("\nDone")
parser.exit(0)
except Exception as e:
parser.exit(type(e).__name__ + ": " + str(e))
