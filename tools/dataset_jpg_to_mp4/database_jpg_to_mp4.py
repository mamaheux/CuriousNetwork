import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
from IO_model import FileHandler
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Convert jpg to avi')
    parser.add_argument('-i', '--curiosity_path', type=str, help='Choose the curiosity database path', required=True)
    parser.add_argument('-t', '--test_path', type=str, help='Choose test database path', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()
    convert_to_video(args)


def get_identifying_frame(size):
    img = np.zeros(size, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Carlito-Regular.ttf", 45, encoding='utf-8')
    draw.text((275, 270), 'Vidéo annotée manuellement: ', font=font)
    draw.text((275, 810), 'Vidéo annotée par le \nmodèle (backend VGG-16): ', font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def convert_to_video(args):
    pathOut = os.path.join(args.output_path, 'video.avi')
    file_handle = FileHandler(args.curiosity_path, args.test_path)

    fps = 24
    size = (1920, 1080)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'3IVD'), fps, size)
    alpha = 0.85

    file_handle.generate_annotated_frames_pairs(args.output_path, alpha)
    paires = file_handle.get_annotated_pair(args.output_path)

    for index, _ in enumerate(paires[0]):
        test_img = cv2.imread(paires[0][index])
        curiosity_img = cv2.imread(paires[1][index])
        assert test_img.shape == curiosity_img.shape

        text_layer = get_identifying_frame((1080, 960, 3))

        test_img = cv2.resize(test_img, (np.int_(size[0]/2), np.int_(size[1]/2)))
        curiosity_img = cv2.resize(curiosity_img, (np.int_(size[0] /2), np.int_(size[1]/2)))

        im_v = cv2.vconcat([test_img, curiosity_img])
        im_v = cv2.hconcat([text_layer, im_v])

        out.write(im_v)
        print(f'Creating video frame: {index + 1}/{len(paires[0])}')

    out.release()


if __name__ == '__main__':
    main()