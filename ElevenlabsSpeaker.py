from elevenlabs import generate, play, set_api_key
from Speaker import Speaker

class ElevenlabsSpeaker(Speaker):
    def __init__(self, elevenlabs_api_key):
        set_api_key(elevenlabs_api_key)

    def speak(self, text) -> None:
        voice = generate(text=text, voice="Bella", model="eleven_monolingual_v1")
        play(voice)