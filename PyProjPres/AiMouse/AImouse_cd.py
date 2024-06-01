import sys

import cv2
import numpy as np
import time
import mouse
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import keyboard
import math
import mediapipe as mp
import pyautogui as pag
import tkinter as tk
from tkinter import messagebox


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, minTrackCon=0.6):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                               connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 0), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 0), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def whatHand(self):
        hand = [0,0]
        if len(self.lmList):
            if self.lmList[4][1] > self.lmList[17][1]:
                hand = [0, 1]
            else:
                hand = [1, 0]
        return hand

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            return length, info, img
        else:
            return length, info


def LCM(img, x1, y1, length):
    cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
    mouse.click('left')


def RCM(img, x1, y1, length):
    cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
    mouse.click(button='right')


def chngVol(length, minVol, maxVol, volume, distanse170cm):
    coff = distanse170cm / 100
    length = length * coff
    # vol = np.interp(length, [10, 90], [-82, 0])
    vol = np.interp(length, [10, 90], [minVol, maxVol])
    print(length, vol)
    volume.SetMasterVolumeLevel(vol, None)


def save_frame(img):
    _, buffer = cv2.imencode('.jpg', img)
    frame_bytes = buffer.tobytes()
    sys.stdout.buffer.write(frame_bytes)
    sys.stdout.flush()

def get_available_cameras():
    available_cameras = []
    for i in range(10):  # Usually, up to 10 cameras are reasonable to check
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera(index_var, root):
    index = int(index_var.get())
    root.destroy()
    run_cursor(index)

def show_camera_selection_window():
    available_cameras = get_available_cameras()
    if not available_cameras:
        messagebox.showerror("Ошибка", "Нет доступных камер")
        return

    root = tk.Tk()
    root.title("Выберите нужную какмеру")

    tk.Label(root, text="Выберите камеру:").pack(pady=10)

    index_var = tk.StringVar(value=available_cameras[0])
    for cam in available_cameras:
        tk.Radiobutton(root, text=f"Камера {cam}", variable=index_var, value=cam).pack(anchor=tk.W)

    tk.Button(root, text="Выбрать", command=lambda: select_camera(index_var, root)).pack(pady=10)


def run_cursor(index):
    try:
        cam = cv2.VideoCapture(index)
        if not cam.isOpened():
            raise ValueError(f"Не могу открыть камеру {index}")

        root = tk.Tk()
        root.title(f"Камера {index}")

        label = tk.Label(root)
        label.pack()

        wCam, hCam = 640, 480
        cam.set(3, wCam)
        cam.set(4, hCam)


        pTime = 0

        detector = HandDetector(detectionCon=0.7, maxHands=2)

        wScr, hScr = pag.size()
        frameR = 100

        smooth = 5.6
        plocX, plocY = 0, 0
        clockX, clockY = 0, 0

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volRange = volume.GetVolumeRange()
        minVol = volRange[0]
        maxVol = volRange[1]

        flag = False
        flag4time = False

        prevLenght = 0

        while True:
            success, img = cam.read()
            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img, flipType=False)

            if hands:
                hand1 = hands[0]
                lmList1 = hand1["lmList"]
                bbox1 = hand1["bbox"]
                centerPoint1 = hand1["center"]
                handType1 = hand1["type"]
                fingers1 = detector.fingersUp(hand1)

                if len(lmList1) != 0:
                    coord17x, coord17y = lmList1[17][1:]
                    coord0x, coord0y = lmList1[0][1:]
                    coord5x, coord5y = lmList1[5][1:]
                    coord517x, coord517y = (coord17x + coord5x) / 2, (coord17y + coord5y) / 2
                    shx17 = coord17x - coord0x
                    shy17 = coord17y - coord0y
                    shx517 = coord517x - coord0x
                    shy517 = coord517y - coord0y
                    ratioalpha = np.arctan(0)
                    try:
                        alphaplusbeta = np.arctan(shx517 / shy517)
                    except ZeroDivisionError:
                        alphaplusbeta = np.arctan(shx517 / (shy517 + 0.1))
                    ratiobeta = -(alphaplusbeta - ratioalpha * 0)
                    shxnew = (shx17 * np.cos(ratiobeta)) + (shy17 * np.sin(ratiobeta))
                    shynew = (-shx17 * np.sin(ratiobeta)) + (shy17 * np.cos(ratiobeta))
                    ratioXY = abs(shxnew / shynew)
                    constratioXY = abs(-0.4)

                    if ratioXY >= constratioXY:
                        l = np.abs(shxnew * np.sqrt(1 + (1 / constratioXY) ** 2))
                        distanse170cm = 5503.9283512 * l ** (-1.0016171)
                    else:
                        l = np.abs(shynew * np.sqrt(1 + constratioXY ** 2))
                        distanse170cm = 5503.9283512 * l ** (-1.0016171)
                    cv2.putText(img, f'{str(int(distanse170cm))}cm', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                if len(hands) > 0:
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)
                    if fingers[0] and fingers[4] and not any(fingers[1:4]):
                        length, _, _ = detector.findDistance(hand["lmList"][8][:2], hand["lmList"][20][:2], img)
                        vol = np.interp(length, [20, 120], [minVol, maxVol])
                        volume.SetMasterVolumeLevel(vol, None)
                        print(f"Громкость: {vol}")

                if len(lmList1) != 0:
                    x1, y1 = lmList1[8][:2]

                finup = detector.fingersUp(hand1)
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
                if finup[0] == 0 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0:
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    clockX = clockX + (x3 - plocX) / smooth
                    clockY = clockY + (y3 - plocY) / smooth
                    mouse.move(clockX, clockY)
                    cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                    plocX, plocY = clockX, clockY

                if finup[0] == 0 and finup[1] == 1 and finup[2] == 1 and finup[3] == 0 and finup[4] == 0:
                    length, info, img = detector.findDistance(lmList1[8][:2], lmList1[12][:2], img)
                    if length > 40:
                        flag = True
                    if length < 35 and flag == True:
                        LCM(img, x1, y1, length)
                        flag = False

                if finup[0] == 0 and finup[1] == 1 and finup[2] == 0 and finup[3] == 0 and finup[4] == 1:
                    length, info, img = detector.findDistance(lmList1[8][:2], lmList1[20][:2], img)
                    if length > 30:
                        flag = True
                    if length < 50:
                        RCM(img, x1, y1, length)
                        flag = False

                if finup[0] == 1 and finup[1] == 1 and finup[2] == 1 and finup[3] == 0 and finup[4] == 0:
                    length, info, img = detector.findDistance(lmList1[8][:2], lmList1[12][:2], img)
                    if length < 25:
                        mouse.press(button="left")
                        mouse.move(clockX, clockY)

                if finup[0] == 1 and finup[1] == 0 and finup[2] == 0 and finup[3] == 0 and finup[4] == 0:
                    if len(lmList1) != 0:
                        x1, y1 = lmList1[4][:2]
                        x2, y2 = lmList1[5][:2]
                        if y1 > y2:
                            mouse.wheel(delta=-0.5)
                        elif y1 < y2:
                            mouse.wheel(delta=0.5)

            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                centerPoint2 = hand2["center"]
                handType2 = hand2["type"]
                fingers2 = detector.fingersUp(hand2)

                if finup[0] == 1 and finup[1] == 1 and finup[2] == 1 and finup[3] == 1 and finup[4] == 1:
                    break

            save_frame(img)
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Работа завершена")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera_selection_window()