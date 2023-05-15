# test py
import os
import cv2
from tqdm import tqdm
from deid_command import anonymize_video
import face_recognition

if __name__ == "__main__":
    from rmn import RMN
    base_directory = "../datasets/CK+/extended-cohn-kanade-images/cohn-kanade-images"

    total_frames = 0
    detected_frames = 0
    same_emotion = 0

    m = RMN()

    try:


        # Iterate through subjects

        subdir_list = os.listdir(base_directory)

        for indx in tqdm(range(0, len(subdir_list))):
            subject = subdir_list[indx]

            subpath = os.path.join(base_directory, subject)
            if not os.path.isdir(subpath):
                continue
            
            # print("processing subject", subject)

            # iterate through subfolders
            for exp in os.listdir(subpath):
                
                
                exppath = os.path.join(subpath, exp)
                if not os.path.isdir(exppath):
                    continue

                # print("processing expression folder", exp)
                

                # create the video
                videoname = subject + "_" + exp + '.mp4'
                inputname = os.path.join('eval_vids', videoname)
                outname = os.path.join('deid_vids', videoname)

                identity = cv2.imread(os.path.join(exppath, os.listdir(exppath)[0]))
                h, w = identity.shape[:2]
                out = cv2.VideoWriter(inputname,cv2.VideoWriter_fourcc(*'mp4v'), 10, (w,h))
                # iterate through the images
                for image in os.listdir(exppath):
                    # print("writing image", image)
                    img = cv2.imread(os.path.join(exppath, image))
                    out.write(img)
                    # write video

                out.release()
                
                anonymize_video(inputname, outname)

                # compare identity

                orig_cap = cv2.VideoCapture(inputname)
                anon_cap = cv2.VideoCapture(outname)
                length = int(anon_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for x in range(0, length):
                    # compare identtiy
                    _, frame = anon_cap.read()
                    _, origframe = orig_cap.read()
                    
                    known_encoding = face_recognition.face_encodings(origframe)[0]
                    unknown_encoding = face_recognition.face_encodings(frame)[0]
                    
                    total_frames+=1
                    if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
                        detected_frames+=1


                    # compare emotion
                    original_emotions = m.detect_emotion_for_single_frame(origframe)[0]["emo_label"]
                    anonymized_emotions = m.detect_emotion_for_single_frame(frame)[0]["emo_label"]

                    if original_emotions == anonymized_emotions:
                        same_emotion += 1

        print("total_frames", total_frames, "detected_frames", detected_frames, "emotion success", same_emotion)
    except KeyboardInterrupt:
        print("total_frames", total_frames, "detected_frames", detected_frames, "emotion success", same_emotion)



    
    
