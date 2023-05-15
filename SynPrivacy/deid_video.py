import sys
import pathlib
import argparse
import cv2
import os


if __name__ == "__main__":
    sys.path.insert(1, 'deepprivacy')
    sys.path.insert(1, 'deepprivacy/deep_privacy')

    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s", "--source_path",
            help="Target path to infer. Can be video or image, or directory",
        )
        parser.add_argument(
            "-t", "--target_path",
            help="Target path to save anonymized result.\
                    Defaults to subdirectory of config file."
        )
        parser.add_argument(
            "-m", "--mode",
            help="Anonymization mode.", default="swap"
        )
        return parser

    parser = get_parser()
    args = parser.parse_args()

    assert args.source_path is not None, "source path cannot be empty!" 
    assert len(args.source_path) > 0, "source path cannot be empty!" 
    print("source path:", args.source_path)

    # Get first frame of the video

    print(" #> Finding frame to anonymize...")

    cap = cv2.VideoCapture(args.source_path)
    _, img = cap.read()

    cv2.imwrite("temp.jpg", img)


    # Generate the de-id image with deepprivacy 

    print(" #> Anonymizing frame...")
    from deepprivacy.deep_privacy.build import build_anonymizer

    anonymizer = build_anonymizer()
    anonymizer.anonymize_image_paths([pathlib.Path('temp.jpg')], [pathlib.Path('out.jpg')])

    # reenact with fsgan

    print(" #> Reenacting video...")
    sys.path.insert(1, 'fsgan')
    sys.path.insert(1, 'fsgan/inference')
    sys.path.insert(1, 'fsgan/datasets')

    # switch into the fsgan directory to run their code
    curdir = os.getcwd()
    fsgan_dir = os.path.join(curdir, 'fsgan')
    os.chdir(fsgan_dir)

    if args.mode == 'reenact':

        # reenact
        from fsgan.inference import reenact
        reenact.main(source=['../out.jpg'], target=['../'+args.source_path], output='../output.mp4')

    elif args.mode == 'swap':

        # swap
        from fsgan.inference import swap
        swap.main(source=['../out.jpg'], target=['../'+args.source_path], output='../output.mp4')

    else:
        raise Exception("Bad mode argument: should be one of 'reenact', 'swap'")


