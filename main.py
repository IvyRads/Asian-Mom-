import gun
import serial
import sleepy2
import its
import time

import os
import random
from playsound import playsound  # pip install playsound==1.2.2

serial_inst = serial.Serial(port="COM6", baudrate=9600)

# 5 water
def play_random_audio(folder_path):
    # Get list of audio files (mp3 and wav)
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp3', '.wav'))]

    if not audio_files:
        print("No audio files found in the folder.")
        return

    # Pick a random file
    chosen_file = random.choice(audio_files)
    file_path = os.path.join(folder_path, chosen_file)

    print(f"Playing: {chosen_file}")
    playsound(file_path)

mygun = gun.GUN()
mysleep = sleepy2.Sleep()
phone = its.ObjectDetector()

while True:
    detected = phone.detect()
    if detected == 'cell phone':
        mygun.run(serial_inst)
    elif detected == 'book':
        # playMotivation()
        folder = os.path.join(os.getcwd(), "C:\\Users\\VIDYA S R\\OneDrive\\Desktop\\useless\\audio_water")
        play_random_audio(folder)

    time.sleep(1)

    mysleep.main(serial=serial_inst)

    time.sleep(1)