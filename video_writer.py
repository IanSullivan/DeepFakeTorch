import os
import cv2
import numpy as np
from faceswap import swap_faces
from face_detect import extract_face
import torch
from model import AutoEncoder
from skimage import img_as_ubyte
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-original_video', type=str, default='data_dst.mp4', help='where to save the images')
parser.add_argument('-model_location', type=str, default='model.pt', help='model name')
parser.add_argument('-decoder', type=str, default='b', help='Which way to decode the image, decoder a or b')
parser.add_argument('-out_name', type=str, default='video_swaped', help='The name of the output movie')

args = parser.parse_args()

cap = cv2.VideoCapture(args.original_video)
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

model = AutoEncoder(image_channels=3).to(device)
model.load_state_dict(torch.load(args.model_location))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_tracked = cv2.VideoWriter('{}.mp4'.format(args.out_name), fourcc, fps, (width, height))
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        try:
            print('\rTracking frame: {}'.format(i + 1), end='')
            i += 1
            # Retrive face from frame, align it, resize it in cv2 to fit into model
            img1_face = extract_face(frame)
            img1_face = np.array(img1_face)
            img1_face = cv2.resize(img1_face, (128, 128))

            #  convert the frame
            frame = np.array(frame)

            #  pytorch takes in channel, height and width,  so transpose to change into correct dimensions
            img1_face = cv2.cvtColor(img1_face, cv2.COLOR_BGR2RGB)
            img_tensor = img1_face[:, :, ::-1].transpose((2, 0, 1)).copy()  # chw, RGB order,[0,255]
            img_tensor = torch.from_numpy(img_tensor).float().div(255)  # chw , FloatTensor type,[0,1]
            img_tensor = img_tensor.unsqueeze(0)  # nch*w
            x = img_tensor.to(device)
            model.eval()
            out = model(x, version=args.decoder)

            # convert the pytorch output into cv2
            out = out.data.cpu().squeeze().numpy()
            out = np.transpose(out, (1, 2, 0))
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out = img_as_ubyte(out)
            out2 = swap_faces(out, frame)
            video_tracked.write(out2)

        except Exception as e:
            print(e)
    else:
        break
    # except cv2.error as e:
    #     #     print(e)

