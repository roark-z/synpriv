import os
from deepface import DeepFace

''' Tool to evaluate accuracy of deepface on a test dataset '''

# root = "../datasets/small/images/"
# root = "../datasets/raf/Image/original/"
root = "../datasets/raf/Image/deid_ksn/"

# read the labels into an array
file_path = "../datasets/raf/EmoLabel/list_patition_label.txt"
f = open(file_path, "r")
lines = f.readlines()

# define number of train and test examples
n_train = 12271
n_test  = 3068

# predict dict

predict_dict = {
    "surprise": 1, 
    "fear": 2,
    "disgust": 3,
    "happy": 4,
    "sad": 5,
    "angry": 6, 
    "neutral": 7
}

def load_emotion_label(dpath):

    # remove file ext
    im_name = dpath.split('.')[0]
    
    # get the number
    im_index = int(im_name.split('_')[1])

    # get emotion label
    label = lines[im_index-1].split()[1]
    return int(label)

n_correct = 0
face_fails = 0
img_set = os.listdir(root)

# total images to test
n_total = n_train

for dpath in img_set[0:n_total]:

    # Avoid test images
    if("test" in dpath):
        continue

    # Avoid test images
    if("detected" in dpath):
        continue

    print("inferencing", dpath)

    # catch failed face detections

    try:
        # Run emotion detection

        obj = DeepFace.analyze(img_path = os.path.join(root, dpath), 
                actions = ['emotion']
        )

        predicted = obj["dominant_emotion"]
        pred_num = predict_dict[predicted]


        # Load emotion label

        label = load_emotion_label(dpath)


        # Compare prediction and label and update running accuracy
        emos = list(predict_dict.keys())
        print("predicted", emos[pred_num-1], "label was", emos[label-1])

        if (pred_num == label):
            n_correct += 1

    except ValueError as v:
        print("No face detected, continue...")
        face_fails += 1
        continue

# score = n_correct / len(img_set)
score = n_correct / (n_total-face_fails)
print("final score:", score, "with", face_fails, "failed detections out of", n_total)
