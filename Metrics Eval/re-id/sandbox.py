# import face_recognition
# import os 

# known_path = os.path.abspath("../../datasets/small/images/train_00001.jpg")
# unknown_path = os.path.abspath("../../datasets/small/deid_dp/train_00001.jpg")
# # unknown_path = known_path

# known_image = face_recognition.load_image_file(known_path)
# unknown_image = face_recognition.load_image_file(unknown_path)

# known_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# results = face_recognition.compare_faces([known_encoding], unknown_encoding)

# print(results)

# # ok this seems to work. Might test with images of myself

import pandas as pd
import os

# def write_file(text):
#     file_name = 'progress.txt'
#     if not os.path.exists(file_name):
#         with open(file_name, 'a'):
#             pass

df = pd.DataFrame(index=range(1000), columns=["recognized", "failed"])

df.loc[2] = [0, 1]

print(df)