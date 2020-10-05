import sys
import numpy as np
import cv2
import dlib
import re
import os


detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Read points from text file
def readPoints(path):
    # Create an array of points.
    points = []

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)
    # Insert points into subdiv
    for p in points:
        p = tuple(p)
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        # print("Too Many Faces") #raise TooManyFaces
        return []
    if len(rects) == 0:
        # print("No Faces") #raise NoFaces
        return []

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def swap_faces(img1, img2):
    img1Warped = np.copy(img2)

    points1 = get_landmarks(img1)
    points1 = points1.tolist()

    points2 = get_landmarks(img2)
    points2 = points2.tolist()

    # Find convex hull
    hull1 = []
    hull2 = []
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
    # Find delanauy traingulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = calculateDelaunayTriangles(rect, hull2)
    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img1, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    # Clone seamlessly.
    return cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)


if __name__ == '__main__':

    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        print('ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    # detector = dlib.get_frontal_face_detector()
    # PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
    # predictor = dlib.shape_predictor(PREDICTOR_PATH)

    path = 'swap_images\\b'

    alist = os.listdir(path)
    alist.sort(key=natural_keys)
    for idx, image_name in enumerate(alist):
        try:
            im_path = os.path.join(path, image_name)
            image = cv2.imread(im_path)
        # cropped_img = image[image.shape[0]:image.shape[0]/2]
            height, width, channels = image.shape
            img1 = image[0:height, width // 2:width]
            img2 = image[0:height, 0:width // 2]
            output = swap_faces(img1, img2)
            new_img = np.hstack([output, img2])
            cv2.imshow("Face Swapped", new_img)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
            # cv2.imwrite('swap_images/merged_b/{}'.format(image_name), new_img)
        except Exception as e:
            print(e)
