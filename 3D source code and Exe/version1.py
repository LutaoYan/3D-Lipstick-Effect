import mediapipe as mp
import cv2
import numpy as np
import google.protobuf
import matplotlib.cm as cm


def change_color_lip_with_depth(img, list_lms, index_lip_up, index_lip_down, color, depth_map):
    # Create a mask for the lip region
    mask = np.zeros_like(img)
    points = list_lms[index_lip_up + index_lip_down, :]
    cv2.fillPoly(mask, [points], (255, 255, 255))

    ### depth_map = [depth_map[i] for i in index_lip_up + index_lip_down]
    # Apply depth-based colors to the lip region
    np.zeros_like(img)
    minval = min(depth_map)
    maxval = max(depth_map)
    i = len(depth_map) - 1
    intensity = (depth_map[i] - minval) / (maxval - minval)
    fix = 0.35540665
    if intensity > fix:
        diff = intensity - fix
        intensity = fix - diff*1.8
    print("intensity:", intensity)
    # Scale the base color based on intensity
    ratio = 1.88
    red = (1 + intensity * ratio) * color[2]
    green = (1 + intensity * ratio) * color[0]
    blue = (1 + intensity * ratio) * color[1]
    if red > 255:
        red = 255
    if blue > 255:
        blue = 255
    if green > 255:
        green = 255
    scaled_color = (color[0], color[1], red)
    img_color_lip = scaled_color

    img_color_lip = cv2.bitwise_and(mask, img_color_lip)
    # cv2.imshow("color lip",img_color_lip)

    img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 30)
    img_color_lip = cv2.addWeighted(img, 1, img_color_lip, 0.8, 0)
    # print(img_color_lip)
    return img_color_lip


def change_color_lip(img, list_lms, index_lip_up, index_lip_down, color):
    # cv2.imshow("input",img)

    mask = np.zeros_like(img)
    # points_lip_up = list_lms[index_lip_up,:]
    # mask = cv2.fillPoly(mask,[points_lip_up],(255,255,255))

    # points_lip_down = list_lms[index_lip_down,:]
    # mask = cv2.fillPoly(mask,[points_lip_down],(255,255,255))

    points = list_lms[index_lip_up + index_lip_down, :]
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    # cv2.imshow("mask",mask)

    img_color_lip = np.zeros_like(img)
    img_color_lip[:] = color
    # cv2.imshow("color lip",img_color_lip)
    img_color_lip = cv2.bitwise_and(mask, img_color_lip)
    # cv2.imshow("color lip",img_color_lip)

    img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 10)
    img_color_lip = cv2.addWeighted(img, 1, img_color_lip, 0.8, 0)

    return img_color_lip


# def change_color_glass(img, list_lms, list_glass, color):
#     # cv2.imshow("input",img)
#
#     mask = np.zeros_like(img)
#     # points_lip_up = list_lms[index_lip_up,:]
#     # mask = cv2.fillPoly(mask,[points_lip_up],(255,255,255))
#
#     # points_lip_down = list_lms[index_lip_down,:]
#     # mask = cv2.fillPoly(mask,[points_lip_down],(255,255,255))
#
#     points = list_lms[list_glass, :]
#     #print("print point:", points)
#     mask = cv2.fillPoly(mask, [points], (255, 255, 255))
#     # cv2.imshow("mask",mask)
#
#     img_color_lip = np.zeros_like(img)
#     img_color_lip[:] = color
#     # cv2.imshow("color lip",img_color_lip)
#     img_color_lip = cv2.bitwise_and(mask, img_color_lip)
#     # cv2.imshow("color lip",img_color_lip)
#
#     img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 10)
#     img_color_lip = cv2.addWeighted(img, 1, img_color_lip, 0.8, 0)
#
#     return img_color_lip


def empty(a):
    pass


if __name__ == "__main__":

    # 创建人脸关键点检测对象
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    # 口红颜色调节
    cv2.namedWindow("BGR")
    cv2.resizeWindow("BGR", 640, 240)
    cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
    cv2.createTrackbar("Green", "BGR", 0, 255, empty)
    cv2.createTrackbar("Red", "BGR", 0, 255, empty)

    cap = cv2.VideoCapture(0)

    while True:

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue

        # 获取宽度和高低
        image_height, image_width, channel = np.shape(img)

        # BGR 转 RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_RGB)
        list_lms = []
        list_dep = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            for i in range(478):
                pos_x = int(face_landmarks.landmark[i].x * image_width)
                pos_y = int(face_landmarks.landmark[i].y * image_height)
                pos_z = face_landmarks.landmark[i].z
                list_lms.append((pos_x, pos_y))
                list_dep.append(pos_z)

            list_lms = np.array(list_lms, dtype=np.int32)
            index_lip_up = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 80, 191,
                            78, 61]
            index_lip_down = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181,
                              91, 146, 61, 78]
            index_glass = [226, 113, 225, 224, 223, 222, 221, 189, 244, 245, 233, 232, 231, 230, 229, 228, 31]

            # 获取口红颜色
            b = cv2.getTrackbarPos("Blue", "BGR")
            g = cv2.getTrackbarPos("Green", "BGR")
            r = cv2.getTrackbarPos("Red", "BGR")

            color = (b, g, r)
            # color = (100,210,210)
            # img = change_color_lip(img,list_lms,index_lip_up,index_lip_down,color)

            # 灰度图像 彩色嘴唇
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            # img = change_color_lip(img, list_lms, index_lip_up,index_lip_down, color)
            img = change_color_lip_with_depth(img, list_lms, index_lip_up, index_lip_down, color, list_dep)
            # img = change_color_glass(img,list_lms,index_glass,color)

        cv2.imshow("BGR", img)

        key = cv2.waitKey(1) & 0xFF

        # 按键 "q" 退出
        if key == ord('q'):
            break
    cap.release()
