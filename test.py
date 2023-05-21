import pyaudio
import wave
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import openai

from ChatGPTChatbot import ChatGPTBot
from ElevenlabsSpeaker import ElevenlabsSpeaker

chatbot = ChatGPTBot("KEY_HERE")
speaker = ElevenlabsSpeaker("KEY_HERE")

# Choose your audio input device, frames per buffer, etc.
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16_000  # change this depending on your audio input device
PROCESS_RATE = 0.5  # compute every `PROCESS_RATE` seconds
CHUNK = int(PROCESS_RATE * RATE)
DETECTION_SPEED_SEC = 0.25  # check for speech every `DETECTION_SPEED_SEC` seconds
restart_stream = False

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


pipeline = VoiceActivityDetection(segmentation="philschmid/pyannote-segmentation")
HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.1,
    "offset": 0.3,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.5,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)


# Create a callback method that processes your audio data
def callback(in_data, frame_count, time_info, status):
    global wf
    global restart_stream
    if restart_stream:
        return (in_data, pyaudio.paContinue)
    print("Callback called")
    # Write data to wav file
    wf.writeframes(in_data)
    frames = wf.getnframes()
    # Calculate the duration
    duration = frames / RATE

    output = pipeline("output.wav")
    # check if we are in a speech region
    speeches = list(output.get_timeline().support())
    if len(speeches) > 0:
        # check if the last speech region is still active
        if duration - speeches[-1].end < DETECTION_SPEED_SEC:
            print("Still speaking...")
        else:
            print("Not speaking...")
            # clear the audio file
            wf.close()
            # load the audio file using pyaudio
            wf = wave.open("output.wav", "rb")
            # read the audio file as a float32 array
            data = wf.readframes(int(CHUNK * duration / PROCESS_RATE))
            # convert the float32 array to a tensQAor
            data = torch.from_numpy(np.frombuffer(data, dtype=np.float32))
            # process the audio file
            input_features = processor(
                data, sampling_rate=RATE, return_tensors="pt"
            ).input_features
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            print(transcription[0])
            reply = chatbot.chat(transcription[0])
            print(f"reply:\n{reply}")
            speaker.speak(reply)
            restart_stream = True

    return (in_data, pyaudio.paContinue)


audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=callback,
)

# Open the wav file
wf = wave.open("output.wav", "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)

# Start the stream
stream.start_stream()

# Keep the stream open until we're done processing audio.
try:
    while stream.is_active():
        if restart_stream:
            stream.stop_stream()
            stream.close()
            wf.close()

            # Reset the wav file
            wf = wave.open("output.wav", "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            # Open a new stream
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback,
            )
            stream.start_stream()

            # Reset the restart flag
            restart_stream = False

except KeyboardInterrupt:
    # If the user hits Ctrl+C, close the stream gracefully.
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf.close()
