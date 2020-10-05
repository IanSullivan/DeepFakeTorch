from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image
from img_rotate import rotate
import os
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device, margin=50, select_largest=True, image_size=256)


def extract_face(frame, align=True, margin=5):
    if align:
        frame = rotate(np.array(frame))
    frame = Image.fromarray(frame)
    # mtcnn(frame, save_path=name)
    boxes, _ = mtcnn.detect(frame)
    for box in boxes:
        box_list = box.tolist()
        # bounding box coordinated
        x1 = int(box_list[0])
        y1 = int(box_list[1])
        x2 = int(box_list[2])
        y2 = int(box_list[3])
        #  find the middle of the image to get a perfect square, mtcnn gives a rectangle image of the face so making
        #  the image a square makes it easier to train
        y1 += margin
        y2 -= margin
        diff = abs(y1 - y2)
        mid_x = (x2 + x1) // 2
        # mid_y = (y2 + y1) // 2
        x1 = mid_x - (diff // 2)
        x2 = mid_x + (diff // 2)
        return frame.crop((x1, y1, x2, y2))  # sends back only the square around the face, possible no face detected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_src', type=str, required=True, help='where to save the images')
    parser.add_argument('-out_name', type=str, required=True, help='The name of the output movie')
    parser.add_argument('-margin', type=int, default=5, help='margin around the detected face box')
    parser.add_argument('-align_faces', type=int, default=5, help='align the faces')

    args = parser.parse_args()
    full_dir = os.path.join(args.out_name, 'aligned_faces')

    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    cap = cv2.VideoCapture(args.video_src)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                crop_img = extract_face(frame, margin=args.margin)
                crop_img.save(full_dir + "/{}.png".format(str(i).zfill(4)))
                print('\rTracking frame: {}'.format(i + 1), end='')
                i += 1
            # No face found, not good code
            except Exception as e:
                print(e)
        else:
            break
