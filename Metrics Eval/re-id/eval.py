# Evaluate proportion of recognizable faces

import face_recognition
import os 
from tqdm import tqdm
import pandas as pd

input_path = '../../datasets/raf/Image/original'
output_path = '../../datasets/raf/Image/deid'
# input_path = '../../datasets/small/images'
# output_path = '../../datasets/small/deid_dp'

# list everything in input path
face_list = os.listdir(input_path)

# define compare faces
def compare_faces(known, unknown):

    known_image = face_recognition.load_image_file(known)
    unknown_image = face_recognition.load_image_file(unknown)

    known_encoding = face_recognition.face_encodings(known_image)
    unknown_encoding = face_recognition.face_encodings(unknown_image)
    
    if len(known_encoding) == 0:
        return None
    known_encoding = known_encoding[0]

    if len(unknown_encoding) == 0:
        return None
    unknown_encoding = unknown_encoding[0]


    return face_recognition.compare_faces([known_encoding], unknown_encoding)

# iterate through images
total_faces = 0
recognized = 0
failed = 0

start_index = 0

csv_name = 'progress_ksn.csv'

# Load csv, or create df if the csv doesn't exist
if os.path.exists(csv_name):
    df = pd.read_csv(csv_name)
    results = df["result"].value_counts()
    # Update starting index to continue prev run
    start_index = df["result"].value_counts().sum()
    print("start_index:", start_index)
else:
    df = pd.DataFrame(index=range(len(face_list)), columns=["result"])
    start_index = 0

for x in tqdm(range(start_index, len(face_list))):
    face = face_list[x]
    known_path = os.path.join(input_path, face)
    unknown_path = os.path.join(output_path, face)

    if not os.path.exists(unknown_path):
        continue
    
    result = compare_faces(known_path, unknown_path)

    # reserve 0 for default, so we can remove unused later
    if result is None: # failed
        df.loc[total_faces+start_index] = [-1]
    elif result[0]: # recognized
        df.loc[total_faces+start_index] = [1]
    else: # not recognized
        df.loc[total_faces+start_index] = [0]
    
    total_faces += 1

    # save csv
    if x % 10 == 0:
        df.to_csv(csv_name, index=False)

# save finally after loop finishes
df.to_csv(csv_name, index=False)

detections = df["result"].value_counts()
failed = detections[-1]
try:
    recognized = detections[1]
except KeyError:
    recognized = 0
total_faces = detections[0]+recognized
print("Total identified faces:", recognized, 'out of', total_faces, str(recognized/total_faces))
print(failed, "failed")
