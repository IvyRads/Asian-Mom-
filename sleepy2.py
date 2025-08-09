"""
sleepy_class.py

Requirements:
    pip install mediapipe opencv-python numpy playsound==1.2.2

Usage:
    - Place some .mp3/.wav files in a folder named "audio_water" next to this script.
    - Provide a serial.Serial instance when creating Sleep(serial_inst) if you want to write to serial.
    - Run: python sleepy_class.py
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import math
import random
import os
from playsound import playsound  # pip install playsound==1.2.2
from threading import Thread, Event


class Sleep:
    def __init__(self, serial_inst=None, active_duration=10):
        self.mp_face = mp.solutions.face_mesh
        self.serialInst = serial_inst

        # --- Landmark indices (Mediapipe FaceMesh) ---
        self.L_EYE = [33, 160, 158, 133, 153, 144]
        self.R_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}

        # --- Parameters ---
        self.EAR_THRESH = 0.25
        self.EAR_CONSEC_FRAMES = 18
        self.MAR_THRESH = 0.6
        self.YAWN_CONSEC_FRAMES = 10
        self.ACTIVE_DURATION = active_duration  # seconds

    # --- Helpers ---
    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        # Ensure we have 6 points
        if len(pts) != 6:
            return 0.0, pts
        p1, p2, p3, p4, p5, p6 = pts
        vert1 = self.euclidean(p2, p6)
        vert2 = self.euclidean(p3, p5)
        hor = self.euclidean(p1, p4)
        if hor == 0:
            return 0.0, pts
        return (vert1 + vert2) / (2.0 * hor), pts

    def mouth_aspect_ratio(self, landmarks, mouth_dict, w, h):
        top = (int(landmarks[mouth_dict["top"]].x * w), int(landmarks[mouth_dict["top"]].y * h))
        bottom = (int(landmarks[mouth_dict["bottom"]].x * w), int(landmarks[mouth_dict["bottom"]].y * h))
        left = (int(landmarks[mouth_dict["left"]].x * w), int(landmarks[mouth_dict["left"]].y * h))
        right = (int(landmarks[mouth_dict["right"]].x * w), int(landmarks[mouth_dict["right"]].y * h))
        vert = self.euclidean(top, bottom)
        hor = self.euclidean(left, right)
        if hor == 0:
            return 0.0, top, bottom, left, right
        return vert / hor, top, bottom, left, right

    @staticmethod
    def draw_text(img, text, pos=(30, 30), color=(0, 0, 255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- Main ---
    def main(self, serial):
        self.serialInst = serial
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        ear_counter = 0
        yawn_counter = 0
        flag = False
        ear_history = deque(maxlen=5)
        last_frame = None

        start_time = time.time()

        with self.mp_face.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5) as face_mesh:

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.ACTIVE_DURATION:
                    break  # stop after active_duration seconds

                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # EAR
                    left_ear, left_pts = self.eye_aspect_ratio(landmarks, self.L_EYE, w, h)
                    right_ear, right_pts = self.eye_aspect_ratio(landmarks, self.R_EYE, w, h)
                    ear = (left_ear + right_ear) / 2.0
                    ear_history.append(ear)
                    smoothed_ear = sum(ear_history) / len(ear_history) if len(ear_history) else 0.0

                    # draw eye points
                    for p in left_pts + right_pts:
                        if isinstance(p, tuple):
                            cv2.circle(frame, p, 1, (0, 255, 0), -1)

                    # MAR
                    mar, top, bottom, left_corner, right_corner = self.mouth_aspect_ratio(landmarks, self.MOUTH, w, h)
                    if isinstance(top, tuple):
                        cv2.circle(frame, top, 2, (255, 0, 0), -1)
                        cv2.circle(frame, bottom, 2, (255, 0, 0), -1)

                    # Sleepy detection counters
                    if smoothed_ear < self.EAR_THRESH:
                        ear_counter += 1
                    else:
                        ear_counter = 0

                    if mar > self.MAR_THRESH:
                        yawn_counter += 1
                    else:
                        yawn_counter = 0

                    if ear_counter >= self.EAR_CONSEC_FRAMES or yawn_counter >= self.YAWN_CONSEC_FRAMES:
                        flag = True  # once True, stays True until end

                    # Display info
                    self.draw_text(frame, f"EAR: {smoothed_ear:.3f}", (10, 30))
                    self.draw_text(frame, f"MAR: {mar:.3f}", (10, 60))
                    self.draw_text(frame, f"Flag: {flag}", (10, 90),
                                   (0, 255, 0) if flag else (0, 0, 255))
                    self.draw_text(frame, f"Time left: {self.ACTIVE_DURATION - int(elapsed)}s", (10, 120),
                                   (255, 255, 0))

                else:
                    self.draw_text(frame, "No face detected", (10, 30), color=(0, 255, 255))

                last_frame = frame.copy()
                cv2.imshow("Sleepy Detection (Active)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # user wants to quit early
                    flag = False
                    break

        # --- After active period: handle flag and play audio (if any) ---
        # Keep the last_frame displayed while audio plays (if we have one).
        if flag:
            # Send serial command if serial instance provided
            try:
                if self.serialInst is not None:
                    # make sure it's bytes; many serial libraries accept bytes
                    if hasattr(self.serialInst, "write"):
                        self.serialInst.write(b"5\n")
                        print("Serial write sent: b'5\\n'")
                    else:
                        print("Provided serialInst has no write() method.")
            except Exception as e:
                print("Serial write failed:", e)

            audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "C:\\Users\\VIDYA S R\\OneDrive\\Desktop\\useless\\audio_water")
            if os.path.exists(audio_dir):
                files = [f for f in os.listdir(audio_dir) if f.lower().endswith((".mp3", ".wav"))]
                if files:
                    chosen = random.choice(files)
                    audio_path = os.path.join(audio_dir, chosen)
                    print(f"Playing audio: {chosen}")

                    # play audio in background thread and keep showing last_frame until it finishes
                    done = Event()

                    def _play_and_flag_done(path, done_event):
                        try:
                            playsound(path)
                        except Exception as e:
                            print("Audio playback error:", e)
                        finally:
                            done_event.set()

                    t = Thread(target=_play_and_flag_done, args=(audio_path, done), daemon=True)
                    t.start()

                    # Keep showing last_frame until audio thread finishes
                    while not done.is_set():
                        if last_frame is not None:
                            cv2.imshow("Sleepy Detection (Alert)", last_frame)
                        if cv2.waitKey(100) & 0xFF == ord('q'):
                            # allow user to quit early while audio is playing
                            break
                    t.join(timeout=0.1)
                else:
                    print("No audio files found in 'audio_water' folder.")
            else:
                print("Audio folder 'audio_water' not found.")
        else:
            print("No sleepy event detected during active window.")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Demo starter:
    # If you have pyserial: create a serial.Serial instance and pass it to Sleep(...).
    # Example (UNCOMMENT and adapt if you actually want serial):
    # import serial
    # ser = serial.Serial('COM3', 9600, timeout=1)
    # s = Sleep(ser)
    #
    # For testing without serial, pass None:
    s = Sleep(serial_inst=None, active_duration=10)
    s.main()
