import math
import cv2
import numpy as np
import dlib


# eye_detector = cv2.CascadeClassifier("C:\\Users\\Donna\\Anaconda3\\envs\\torch\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml")
# face_detector = cv2.CascadeClassifier("C:\\Users\\Donna\\Anaconda3\\envs\\torch\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


def rotate(image, output_size=256):
    image = image[:, :, ::-1]  # BGR to RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) > 0:
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)  # get facial features
            shape = np.array([(shape.part(j).x, shape.part(j).y) for j in range(shape.num_parts)])

            # center and scale face around mid point between eyes
            center_eyes = shape[27].astype(np.int)
            eyes_d = np.linalg.norm(shape[36] - shape[45])
            face_size_x = int(eyes_d * 2.)
            if face_size_x < 50:
                continue

            # rotate to normalized angle
            d = (shape[45] - shape[36]) / eyes_d  # normalized eyes-differnce vector (direction)
            a = np.rad2deg(np.arctan2(d[1], d[0]))  # angle
            scale_factor = float(output_size) / float(face_size_x * 2.)  # scale to fit in output_size
            # rotation (around center_eyes) + scale transform
            M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1]), a, scale_factor), [[0, 0, 1]], axis=0)
            # apply shift from center_eyes to middle of output_size
            M1 = np.array([[1., 0., -center_eyes[0] + output_size / 2.],
                           [0., 1., -center_eyes[1] + output_size / 2.],
                           [0, 0, 1.]])
            # concatenate transforms (rotation-scale + translation)
            M = M1.dot(M)[:2]
            # warp
            try:
                face = cv2.warpAffine(image, M, (output_size, output_size), borderMode=cv2.BORDER_REPLICATE)
            except:
                continue
            face = cv2.resize(face, (output_size, output_size), cv2.COLOR_BGR2RGB)
            return face
