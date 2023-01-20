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
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands_detector = hands.process(imgRGB)

    if hands_detector.multi_hand_landmarks:
        for hand in hands_detector.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(img)

    img.flags.writeable = True
    img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(image=img, landmark_list=face_landmarks, connections=mp_face.FACEMESH_TESSELATION, landmark_drawing_spec=None)
            mp_draw.draw_landmarks(image=img, landmark_list=face_landmarks, connections=mp_face.FACEMESH_CONTOURS, landmark_drawing_spec=None)
            mp_draw.draw_landmarks(image=img, landmark_list=face_landmarks, connections=mp_face.FACEMESH_IRISES, landmark_drawing_spec=None)

    cv2.imshow("Detector", img)
    cv2.waitKey(1)
