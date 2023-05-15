import cv2
import face_recognition
from tqdm import tqdm

# load video
cap = cv2.VideoCapture('crop.mp4')

# load deid video
deid = cv2.VideoCapture('obama.mp4')

same_identity = 0

# TODO: tqdm
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


for p in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    _, frame_deid = deid.read()
    # get frame

    # compare identity
    biden_encoding = face_recognition.face_encodings(frame)[0]
    unknown_encoding = face_recognition.face_encodings(frame_deid)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
    if results[0]:
        same_identity += 1

print(same_identity, total_frames)
