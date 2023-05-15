import os
import cv2
from tqdm import tqdm

root_img_folder = os.path.abspath("../datasets/CK+/extended-cohn-kanade-images/cohn-kanade-images")



# iterate over folders
img_folders = os.listdir(root_img_folder)

for i in tqdm(range(0, len(img_folders))):
    img_folder = img_folders[i]
    # check if the folder is a folder

    # print(img_folder)
    folder_path = os.path.join(root_img_folder, img_folder)
    sub_folders = os.listdir(folder_path)

    # iterate over subfolders
    for sub_folder in sub_folders:

        # skip ds store
        if sub_folder == '.DS_Store':
            continue

        sub_folder_path = os.path.join(folder_path, sub_folder)
        img_names = os.listdir(sub_folder_path)

        # remove ds store in subfolders
        if('.DS_Store' in img_names):
            img_names = img_names.remove(".DS_Store")
            
        # skip empty folders
        if(img_names == None or len(img_names) == 0):
            continue

        # get image width and height
        try:
            img = cv2.imread(os.path.join(sub_folder_path, img_names[0]))
            height, width, layers = img.shape
            size = (width,height)
        except TypeError:
            print(sub_folder_path, img_names)
        
        out = cv2.VideoWriter(os.path.join("videos", img_folder + "_" + sub_folder + '.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

        # read images in the folder
        for image in img_names:

            # convert to video
            cv_img = cv2.imread(os.path.join(sub_folder_path, image))
            out.write(cv_img)

        # save to same folder
        out.release()
