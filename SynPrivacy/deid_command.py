import sys
import pathlib
import argparse
import cv2
import os


def anonymize_video(source, outpath, mode='swap'):
    sys.path.insert(1, 'deepprivacy')
    sys.path.insert(1, 'deepprivacy/deep_privacy')

    assert source is not None, "source path cannot be empty!" 
    assert len(source) > 0, "source path cannot be empty!" 
    # print("source path:", args.source_path)

    # Get first frame of the video

    # print(" #> Finding frame to anonymize...")
    print(source)

    cap = cv2.VideoCapture(source)
    _, img = cap.read()

    cv2.imwrite("temp.jpg", img)


    # Generate the de-id image with deepprivacy 

    # print(" #> Anonymizing frame...")
    from deepprivacy.deep_privacy.build import build_anonymizer

    anonymizer = build_anonymizer()
    anonymizer.anonymize_image_paths([pathlib.Path('temp.jpg')], [pathlib.Path('out.jpg')])



    # reenact with fsgan

    # print(" #> Reenacting video...")
    sys.path.insert(1, 'fsgan')
    sys.path.insert(1, 'fsgan/inference')
    sys.path.insert(1, 'fsgan/datasets')

    # switch into the fsgan directory to run their code
    curdir = os.getcwd()
    fsgan_dir = os.path.join(curdir, 'fsgan')
    os.chdir(fsgan_dir)

    if mode == 'reenact':

        # reenact
        from fsgan.inference import reenact
        reenact.main(source=['../out.jpg'], target=['../'+source], output='../'+outpath)

    elif mode == 'swap':

        # swap
        from fsgan.inference import swap
        swap.main(source=['../out.jpg'], target=['../'+source], output='../'+outpath)

    else:
        raise Exception("Bad mode argument: should be one of 'reenact', 'swap'")

    curdir = os.getcwd()
    fsgan_dir = os.path.join(curdir, "..")
    os.chdir(fsgan_dir)
