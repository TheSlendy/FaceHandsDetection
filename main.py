import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
from visualize import Visualize


# init mediapipe solutions
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(refine_landmarks=True)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils

# open camera
cap = cv2.VideoCapture(0)

# init open3d visualization
vis = Visualize()

while True:
    point_cloud = []
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hands_detector = hands.process(imgRGB)
    #
    # if hands_detector.multi_hand_landmarks:
    #     for hand in hands_detector.multi_hand_landmarks:
    #         mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                point_cloud.append([lm.x, lm.y, lm.z])
            vis.update_mesh(point_cloud)
            vis.show()

    cv2.waitKey(1)
