import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands()
face = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands_detector = hands.process(img_rgb)
    face_detector = face.process(img_rgb)
    if hands_detector.multi_hand_landmarks:
        for hand in hands_detector.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
    if face_detector.multi_face_landmarks:
        for face in face_detector.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face, mp_face.FACEMESH_TESSELATION)
            mp_draw.draw_landmarks(img, face, mp_face.FACEMESH_CONTOURS)
            mp_draw.draw_landmarks(img, face, mp_face.FACEMESH_IRISES)

    cv2.imshow("Detector", img)
    cv2.waitKey(1)
