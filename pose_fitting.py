# -*- coding: utf-8 -*-
from openni import openni2
import cv2
import mediapipe as mp
import numpy as np
import math
from time import sleep
# import serial
from PIL import Image

# ser = serial.Serial(
#     port='\\\\.\\COM4',
#     baudrate = 9600,
#     parity=serial.PARITY_ODD,
#     stopbits=serial.STOPBITS_ONE,
#     bytesize=serial.EIGHTBITS
# )
# if ser.isOpen():
#     ser.close()
# ser.open()
# ser.isOpen()

openni2.initialize()     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()
print(dev.get_device_info())

# depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
# depth_stream.start()
color_stream.start()


DESIRED_HEIGHT = 768
DESIRED_WIDTH = 1024


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(
            image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(
            image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow(img)

def calc_angle(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    return (y2-y1) / (x2-x1)


def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new

def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加
        y1,y2,x1,x2为叠加位置坐标值
    """

    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)

    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]

    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]

    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png

    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))

    return jpg_img



def change_cloth(cloth):
    global origin_cloth
    origin_cloth = cv2.imread(cloth, cv2.IMREAD_UNCHANGED)

# offset for shoulder pos
offset_x = 290
offset_y = 90
# para for cloth
cloth_width = 422
cloth_height = 800
scale = 1
# IMAGE_FILES = ['test_img.jpg']

origin_cloth = cv2.imread('trans_cloth.png', cv2.IMREAD_UNCHANGED)
BG_COLOR = (192, 192, 192)  # gray

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fps = cap.get(cv2.CAP_PROP_FPS)
# width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
#     cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
# out = cv2.VideoWriter('result1.mp4', fourcc, fps,
#                         (width*2, height))  # 写入视频

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# For static images:
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# with mp_pose.Pose(
#         static_image_mode=True,
#         model_complexity=2,
#         enable_segmentation=True,
#         min_detection_confidence=0.5) as pose:
while True:
    # success, image = cap.read()

    # if not success:
    #     print("Ignoring empty camera frame.")
    #     # If loading a video, use 'break' instead of 'continue'.
    #     continue
    # image = cv2.imread(file)
    # image = cv2.resize(cv2.imread(file), (600, 800))
    image = color_stream.read_frame()
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        cv2.imshow("", image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        print("no landmarks")
        continue

    # print(
    #     f'Left shoulder coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height})'
    # )
    # print(
    #     f'Right shoulder coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height})'
    # )
    mid_x = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 * image_width)
    mid_y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 * image_width)
    shoulderL_x = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]).x * image_width)
    shoulderL_y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]).y * image_height)
    shoulderR_x = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]).x * image_width)
    shoulderR_y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]).y * image_height)
    shoulder_width = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * image_width
    pos_x = shoulderR_x
    pos_y = shoulderR_y
    scale = shoulder_width / cloth_width
    angle = calc_angle(shoulderR_x, shoulderR_y, shoulderL_x, shoulderL_y)
    if angle < -0.2 or angle > 0.2 or scale <= 0:
        cv2.imshow("", image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord("a"):
            test_cloth = change_cloth("test_change.png")
            print("on pressed a")
            # ser.write("%01#RDD0010000107**\r")
        print("angle problem")
        continue
    print(scale)
    # print(pos_x, pos_y)
    print("angle = ", angle)
    # for name in mp_pose.PoseLandmark:
    #   print(name)
    # print(results.pose_landmarks.landmark)

    # annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # condition = np.stack(
    #     (results.segmentation_mask,) * 3, axis=-1) > 0.1
    # bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = BG_COLOR
    # annotated_image = np.where(condition, annotated_image, bg_image)


    # Draw pose landmarks on the image.
    # mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # show
    # cv2.imshow(annotated_image)
    # cv2.imwrite('skeleton' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)



    test_cloth = cv2.resize(origin_cloth, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x1 = pos_x - int(offset_x * scale)
    y1 = pos_y - int(offset_y * scale)
    x2 = x1 + test_cloth.shape[1]
    y2 = y1 + test_cloth.shape[0]
    print(x1, y1)

    # 开始叠加
    res_img = merge_img(image, test_cloth, y1, y2, x1, x2)

    cv2.imshow("", res_img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
    if key & 0xFF == ord("a"):
        test_cloth = change_cloth("test_change.png")
        print("on pressed a")
        # ser.write("a")


# depth_stream.stop()
color_stream.stop()
dev.close()
