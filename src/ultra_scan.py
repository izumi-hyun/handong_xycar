#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
import os
import time
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor
from cv_bridge import CvBridge
from std_msgs.msg import Int32MultiArray

bridge = CvBridge()  # ROS에서 OpenCV를 편하게 사용하기 위한 CvBridge 사용 준비
cv_image = np.empty(shape=[0])
new_angle = 0  # 모터 조향각 초기값
new_speed = 0  # 모터 속도 초기값
motor_msg = xycar_motor()  # 모터 토픽 메시지
flag = False
stopline_count = 0
flag_start_time = 0  # flag가 True로 설정된 시점을 기록하는 변수
stopline_time = 0
max_time_end = 0

motor = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)  # 차량 제어를 위한 퍼블리셔

ultra_msg = None

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def roi_for_edge(canny, x, y, w, h):
    mask = np.zeros_like(canny)
    polygon = np.array([[
        (x, y + h),
        (x + w, y + h),
        (x + w, y),
        (x, y)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def draw_red_lines(img, lines, thickness=2):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                if -0.15 <= slope <= 0.15:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)  # 빨간색

def draw_green_lines(img, lines, thickness=2):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                if not (-0.15 <= slope <= 0.15):
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)  # 초록색

def process_image(img, x, y, w, h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur_gray, 50, 150)
    roi_edges = roi_for_edge(edges, x, y, w, h)
    lines = hough_lines_detection(roi_edges, 1, np.pi / 180, 20, 20, 30)
    line_img = np.zeros_like(img)  # 라인을 그릴 빈 이미지
    
    valid_line_count = 0 # detect stop line
    green_lines = []
    left_green_lines = []
    right_green_lines = []

    if lines is not None:
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 기울기 계산
            if x2 != x1:  # x 값이 같으면 수직선이므로 제외
                slope = (y2 - y1) / (x2 - x1)
                
                # 기울기 필터링 (-0.15 ~ 0.15 범위 내의 라인만 선택)
                if -0.15 <= slope <= 0.15:
                    filtered_lines.append(line)
                    valid_line_count += 1
                else:
                    filtered_lines.append(line)
                    
        # 빨간색 라인 그리기
        draw_red_lines(line_img, filtered_lines)
        
        # 초록색 라인 그리기
        draw_green_lines(line_img, filtered_lines)
        
        # x=320 기준으로 왼쪽과 오른쪽 라인 나누기
        fixed_x = 320
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # 기울기가 0인 경우를 제외
                slope = (y2 - y1) / (x2 - x1)
                if not (-0.15 <= slope <= 0.15):  # 수평선이 아닌 경우
                    if x1 < fixed_x and x2 < fixed_x:
                        left_green_lines.append((x1, x2))
                    elif x1 > fixed_x and x2 > fixed_x:
                        right_green_lines.append((x1, x2))
        
        # x=320 기준으로 가장 가까운 왼쪽과 오른쪽 라인 각각 하나씩 선택
        if left_green_lines:
            closest_left = min(left_green_lines, key=lambda coords: abs(coords[0] - fixed_x) + abs(coords[1] - fixed_x))
            x1, x2 = closest_left
            cv2.circle(line_img, (int(x2), int(img.shape[0] / 2)), 5, (255, 0, 0), -1)  # 파란색 점

        if right_green_lines:
            closest_right = min(right_green_lines, key=lambda coords: abs(coords[0] - fixed_x) + abs(coords[1] - fixed_x))
            x1, x2 = closest_right
            cv2.circle(line_img, (int(x1), int(img.shape[0] / 2)), 5, (255, 0, 0), -1)  # 파란색 점

        # 왼쪽과 오른쪽 green line의 x좌표 평균 계산
        if left_green_lines and right_green_lines:
            left_x = sum([x for x1, x2 in left_green_lines for x in (x1, x2)]) / len(left_green_lines)
            right_x = sum([x for x1, x2 in right_green_lines for x in (x1, x2)]) / len(right_green_lines)
            avg_x = (left_x + right_x) / 4
        else:
            avg_x = fixed_x

        # 빨간색 점으로 avg_x 위치 표시
        cv2.circle(line_img, (int(avg_x), int(img.shape[0] / 2)), 5, (0, 0, 255), -1)  # 빨간색 점
    
    # 원본 이미지와 라인 이미지를 합성
    combined_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    # 고정된 위치에 세로라인 그리기
    img_height = img.shape[0]
    cv2.line(combined_img, (fixed_x, 0), (fixed_x, img_height), (0, 255, 225), 2)  # 빨간색 세로선
    
    return combined_img, avg_x, valid_line_count

def cam_exposure(value):
    command = 'v4l2-ctl -d /dev/videoCAM -c exposure_absolute=' + str(value)
    os.system(command)

def nothing(x):
    pass
    
def ultra_callback(data):
    global ultra_msg
    ultra_msg = data.data
    
def drive(angle, speed):
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)

def start():
    rospy.init_node('ultra_node', anonymous=True)

    rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)  # Image 토픽이 오면 콜백함수가 호출되도록 세팅

    rospy.wait_for_message("/usb_cam/image_raw", Image)
    print("Camera Ready --------------")  # Image 토픽이 도착할 때까지 잠시 기다린다

    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback)

    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("Ultrasonic Ready ----------") 

    # 슬라이더 창 생성
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('ROI X', 'Trackbars', 0, 640, nothing)
    cv2.createTrackbar('ROI Y', 'Trackbars', 0, 480, nothing)
    cv2.createTrackbar('ROI Width', 'Trackbars', 640, 640, nothing)
    cv2.createTrackbar('ROI Height', 'Trackbars', 480, 480, nothing)
    while not rospy.is_shutdown():
        if cv_image.size != 0:
            # 슬라이더 값 읽기
            '''
            roi_x = cv2.getTrackbarPos('ROI X', 'Trackbars')
            roi_y = cv2.getTrackbarPos('ROI Y', 'Trackbars')
            roi_w = cv2.getTrackbarPos('ROI Width', 'Trackbars')
            roi_h = cv2.getTrackbarPos('ROI Height', 'Trackbars')
            '''
            roi_x = 0
            roi_y = 188
            roi_w = 509
            roi_h = 263
        
            # ROI 적용 및 크롭
            cropped_image = cv_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
            processed_img, avg_x, valid_line_count = process_image(cropped_image, roi_x, roi_y, roi_w, roi_h)
        
            # ROI 표시
            roi_display = cv_image.copy()
            cv2.rectangle(roi_display, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

            # 이미지 표시
            cv2.imshow("original", roi_display)
            cv2.imshow("processed", processed_img)
            cv2.waitKey(1)
        
            fixed_x = 320  # 고정된 x 위치(카메라 중앙값)
        
            # 파란 점으로 표시된 평균 x 좌표를 사용하는 예시
            new_angle = avg_x - fixed_x
        
            # stopline_count가 2가 되면 속도를 0으로 설정하여 차량을 멈춤
            if stopline_count >= 2:
                time.sleep(1)
                new_speed = 0
            else:
                new_speed = 20  # 현재 속도는 20으로 유지

            current_time = time.time()  # 현재 시간을 가져옴

            if flag:
                # flag가 True로 설정된 이후 10초가 경과했는지 확인
                if current_time - flag_start_time >= 3:
                    if valid_line_count < 5:
                        flag = False
            else:
                if valid_line_count > 5:
                    flag = True
                    flag_start_time = current_time  # flag가 True로 설정된 시점을 기록
                    stopline_count += 1
                # valid_line_count < 5 인 경우에는 flag를 False로 설정할 필요 없음 (이미 False 상태임)

            drive(new_angle, new_speed)
            print(f"valid lines detected: {valid_line_count}")
            print(flag, stopline_count)
        
            print(ultra_msg)
            rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
            
if __name__ == '__main__':
    start()
