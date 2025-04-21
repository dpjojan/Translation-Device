mkdir -p ~/piper_models
cd ~/piper_models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/en_US-libritts-high.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/en_US-libritts-high.onnx.json -O config.json

parage --model ~/piper_models/en_US-libritts-high.onnx --config ~/piper_models/config.json &


import requests

PARAGE_URL = "http://localhost:8080/synthesize"

def speak(text):
    response = requests.post(PARAGE_URL, json={"text": text})
    if response.status_code == 200:
        with open("output.wav", "wb") as f:
            f.write(response.content)
        print("Speech synthesis complete. Playing audio...")
        subprocess.run(["aplay", "output.wav"])
    else:
        print("Error:", response.text)

# Example usage
speak("Hello, this is a test.")
