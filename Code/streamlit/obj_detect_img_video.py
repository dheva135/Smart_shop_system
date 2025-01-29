from ultralytics import YOLO
from PIL import Image
import cv2
import math
import cvzone
import numpy as np
import streamlit as st
import pyautogui
import pandas as pd
from sort import *
import torch
import mysql.connector
import time
import qrcode
from prettytable import PrettyTable
import io
from twilio.rest import Client
from datetime import datetime

Products_added = []
out_line=[]
in_line=[]
Final=[]
U_Final=[]
current_total =0
Free = []

######################## SMS MESSAGE ############################################
# Twilio credentials
twilio_phone_number = '+15703655007'
account_sid = 'AC2246a93ce00c3636204035137701d444'
auth_token = '51e1e50dc35ebe577fc65f388143f3c9'
client = Client(account_sid, auth_token)


def send_sms(to_phone_number, message):
    try:
        # Send SMS message
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=to_phone_number
        )
        print("Message sent successfully. SID:", message.sid)
    except Exception as e:
        print("Error:", e)
#######################################################################

def load_product_counter(video_name_s,video_name_t, kpi1_text, kpi2_text, kpi3_text, kpi4_text,kpi5_text,stframe_s,stframe_t,usr_name,phno):
    cap_s = cv2.VideoCapture(video_name_s)
    cap_t = cv2.VideoCapture(video_name_t)

    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))

    # -----Background Subtractor---------(Mixture of Gaussians)
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2) #History is the number of the last frame that are taken into consideretion
    kernel = np.ones((3, 3), np.uint8)                         #strucuring Kernel used for morphological operations like erosion  and dialtion

    #-----VIDEO Ratio-------------------

    cap_s.set(3, 1920)
    cap_s.set(4, 1080)

    cap_t.set(3, 1920)
    cap_t.set(4, 1080)

    image_width = int(cap_s.get(3))
    image_height = int(cap_s.get(4))

    #print("image ({}, {})\n".format(image_width, image_height))

    screen_width, screen_height = pyautogui.size()
    #print("screen ({}, {})\n".format(screen_width, screen_height))

    frame_width = (image_width / screen_width)
    frame_height = (image_height / screen_height)

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    # All Dataset--------------------------
    model = YOLO("../YOLO-Weights/seg3n_25.pt")
    classNames=['Cinthol_Soap', 'Hamam_Soap', 'Him_Face_Wash', 'Maa_Juice', 'Mango', 'Mysore_Sandal_Soap', 'Patanjali_Dant_Kanti', 'Tide_Bar_Soap', 'ujala_liquid']
    model.to(device)
    load = st.button("STOP")

    # Tracking
    tracker_s = Sort(max_age=20, min_hits=3, iou_threshold=0.3) #Max_age=the maximum number of frames that a track is kept without receiving an associated detection.
    tracker_t = Sort(max_age=20, min_hits=3, iou_threshold=0.3) #min_hits= the minimum number of detections required to establish a track.

    #----Mysql Connection-------------------

    connection = mysql.connector.connect(host="localhost", user="root", password="", database="shop3")  #Custom
    cursor = connection.cursor()

    # -------------DISCOUNT OFFER----------------------------

    query = "select * from Offer"
    cursor.execute(query)
    table = cursor.fetchall()
    connection.commit()
    for row in table:
        Original_Product = row[0]
        Discount_Product = row[1]
        print(Original_Product,Discount_Product)

    # query = "select * from free"
    # cursor.execute(query)
    # table = cursor.fetchall()
    # connection.commit()
    # for row in table:
    #     Original_Product=row[0]
    #     Original_Product_cnt=row[1]
    #     Discount_Product=row[2]
    #     Discount_Product_cnt=row[3]

    #----SORT Tracking-----------------------

    tracker_s = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    tracker_t = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    #max_age no of frames to wait for the ID to be detected
    #iou_threshold minimum threshold needs to be detected

    #---------Boundary Lines-------------------------------------------------------------------------------------------

    left_limits1 = [int(250 * frame_width), int(100 * frame_height), int(250 * frame_width), int(1000 * frame_height)]
    left_limits2 = [int(350 * frame_width), int(100 * frame_height), int(350 * frame_width), int(1000 * frame_height)]

    right_limits1 = [int(1650 * frame_width), int(100 * frame_height), int(1650 * frame_width), int(1000 * frame_height)]
    right_limits2 = [int(1550 * frame_width), int(100 * frame_height), int(1550 * frame_width), int(1000 * frame_height)]

    top_limits1 = [int(250 * frame_width), int(100 * frame_height), int(1650 * frame_width), int(100 * frame_height)]
    top_limits2 = [int(250 * frame_width), int(200 * frame_height), int(1550 * frame_width), int(200 * frame_height)]

    bottom_limits1 = [int(250 * frame_width), int(1000 * frame_height), int(1650 * frame_width),int(1000 * frame_height)]
    bottom_limits2 = [int(350 * frame_width), int(950 * frame_height), int(1550 * frame_width), int(950 * frame_height)]

    #-------SIDE Window-----------------------------------------------------------------------------------
    top_limits1_s = [int(0 * frame_width), int(450 * frame_height), int(1920 * frame_width),
                     int(450 * frame_height)]
    top_limits2_s = [int(0 * frame_width), int(500 * frame_height), int(1920 * frame_width),
                     int(500 * frame_height)]

    top_limits3_s = [int(0 * frame_width), int(650 * frame_height), int(1920 * frame_width),
                     int(650 * frame_height)]

    current_total=0

    # ----SIDE-----------
    totalCount_s = []

    Total_products_s = 0
    Products_added_s = []
    Products_removed_s = []
    out_line_s = []
    in_line_s = []

    # ---TOP----------------
    totalCount_t = []

    Total_products_t = 0
    Products_added_t = []
    Products_removed_t = []

    out_line_t = []
    in_line_t = []

    #Final = []
    Hide = []
    Hide_remove = []
    Segment_remove = []

    # -----------Directions------------
    Left = []
    Right = []
    Top = []
    Bottom = []

    # ----Time--------------
    start = time.time()
    Hide_add_time = 0
    Hide_remove_time = 0

    while not load :
#_________________________________________________________________________

            success_s, img_s = cap_s.read()
            results_s = model(img_s, stream=True)
            detections_s = np.empty((0, 5))
            # classArray = []
            allArray_s = []
            currentClass_s=""

            success_t, img_t = cap_t.read()
            results_t = model(img_t, stream=True)
            detections_t = np.empty((0, 5))
            # classArray = []
            allArray_t = []
            currentClass_t = ""

            #---------------DETECT-----------------------------------------

            movement_detector=[]
            # ------Time----------------
            fps = time.time()
            fps = int(fps - start)

            if fps - Hide_add_time > 50:
                Hide = []

            if fps - Hide_remove_time > 50:
                Hide_remove = []

            if success_t:
                #---- Detect Moving object --------- https://learnopencv.com/moving-object-detection-with-opencv/
                fgmask = backgroundObject.apply(img_t)
                _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)#applies a binary threshold to the foreground mask. Pixels with values below 20 are set to 0 (black),
                fgmask = cv2.erode(fgmask, kernel, iterations=1)    #Erosion reduces the size of the foreground regions by removing pixels near the boundaries of foreground regions
                fgmask = cv2.dilate(fgmask, kernel, iterations=40)  #Dilation is the opposite of erosion and increases the size of the foreground regions by adding pixels to the boundaries of foreground regions

                # detect contour
                countors, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#identifies contours in the binary foreground mask
                forground = cv2.bitwise_and(img_t, img_t, mask=fgmask) #CHAIN_APPROX_SIMPLE=It removes all redundant points and compresses the contour, thereby saving memory.

            else:
                break

            #---SIDE VIEW-MASK-----------------------------------------------------
            if success_s:
                # Run Yolov8 inference on the frame
                results_s = model(img_s)
                area_px_s = []
                area_px_s_dummy = []
                found_ele_s=[]

                # visulaize the results on the frame without bounding box
                #annotated_frame_s = results_s[0].plot(boxes=False)

                # --- find mask -----------------------------------------
                l = 0
                masks_s = results_s[0].masks
                if results_s[0].masks is not None:
                    l = len(masks_s)
                print(l)
                i = 0
                while i < l:
                    mask1_s = masks_s[i]
                    mask_s = mask1_s.cpu().data[0].numpy()
                    polygon_s = mask1_s.xy[0]
                    shape_s = mask1_s.shape

                    mask_img_s = Image.fromarray(mask_s, "I")
                    pts_s = np.array([polygon_s], np.int32) #converting detected points to polygon shape
                    pts_s = pts_s.reshape((-1, 1, 2))

                    # draw = ImageDraw.Draw(img)
                    # draw.polygon(polygon, outline=(0, 255, 0), width=5)
                    cv2.polylines(img_s, [pts_s], True, color=(0, 0, 255), thickness=2)  # draw polygon

                    # -- Find Calculate --------------------
                    area_px_s_dummy.append(int(cv2.contourArea(polygon_s)))

                    i = i + 1

                # --To find bounding box--------
                #print(results_s)

                # ------SIDE-DETECT----------------------------------------------------
                for r in results_s:
                    boxes = r.boxes
                    i = 0
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        w, h = x2 - x1, y2 - y1
                        cx, cy = x1 + w // 2, y1 + h // 2

                        #cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class Name
                        cls = int(box.cls[0])

                        currentClass_s = classNames[cls]
                        #print("SIDE_Curret class",currentClass_s)

                        if currentClass_s != "person" and conf > 0.3 and 650 > cy:
                            cvzone.putTextRect(img_s, f'{currentClass_s} {conf} Area:{format(area_px_s_dummy[i])}', (max(0, x1), max(35, y1)),
                                                   scale=1, thickness=1)  # Class Name
                            allArray_s.append([x1, y1, x2, y2, currentClass_s])
                            currentArray_s = np.array([x1, y1, x2, y2, conf])
                            detections_s = np.vstack((detections_s, currentArray_s))
                            area_px_s.append(area_px_s_dummy[i])
                            found_ele_s.append([str(currentClass_s),cx,cy])
                        i = i+1
            else:
                break

            # ---TOP VIEW-MASK-----------------------------------------------------

            if success_t:
                # Detect movement-----------------
                results_m = model(forground)

                for r in results_m:
                    boxes = r.boxes
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        w, h = x2 - x1, y2 - y1
                        cx, cy = x1 + w // 2, y1 + h // 2
                        cv2.circle(forground, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

                        # frameCopy = apply_mask(frameCopy, (x1+50, y1+50), (x2-50, y2-50))
                        # frameCopy = apply_mask(frameCopy, (cx - 50, cy - 50), (cx + 50, cy + 50))
                        # cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class Name
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]
                        # print("<-----POINTS---->",currentClass, cx, cy)

                        if currentClass != "person" and conf > 0.3:
                            cvzone.putTextRect(forground, f'{currentClass} {conf} ',
                                               (max(0, x1), max(35, y1)),
                                               scale=1, thickness=1)  # Class Name
                            movement_detector.append([currentClass, cx, cy])

                # -----------------------------------------------------------

                # Run Yolov8 inference on the frame
                results_t = model(img_t)
                area_px_t = []
                area_px_t_dummy = []
                found_ele_t = []

                # visulaize the results on the frame without bounding box
                annotated_frame = results_t[0].plot(boxes=False)

                # --- find mask -----------------------------------------
                l = 0
                masks = results_t[0].masks
                if results_t[0].masks is not None:
                    l = len(masks)
                print(l)
                i = 0
                while i < l:
                    mask1 = masks[i]
                    mask = mask1.cpu().data[0].numpy()
                    polygon = mask1.xy[0]
                    shape = mask1.shape

                    mask_img = Image.fromarray(mask, "I")
                    pts = np.array([polygon], np.int32)
                    pts = pts.reshape((-1, 1, 2))

                    # draw = ImageDraw.Draw(img)
                    # draw.polygon(polygon, outline=(0, 255, 0), width=5)
                    cv2.polylines(img_t, [pts], True, color=(0, 0, 255), thickness=2)  # draw polygon

                    # -- Find Calculate --------------------
                    area_px_t_dummy.append(int(cv2.contourArea(polygon)))

                    i = i + 1

                # --To find bounding box--------
                # print(results)

                # print("-----movement----",movement_detector)
                # print("-----area_px_t_dummy----",area_px_t_dummy)
                # ------TOP-DETECT----------------------------------------------------
                for r in results_t:
                    boxes = r.boxes
                    # print("-----len--------",len(boxes))
                    i = 0
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        w, h = x2 - x1, y2 - y1
                        cx, cy = x1 + w // 2, y1 + h // 2
                        cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        cv2.circle(img_t, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class Name
                        cls = int(box.cls[0])
                        currentClass_t = classNames[cls]
                        # print("------TOP--cx,cy------------",cx,cy)
                        # print("TOP_Curret class", currentClass_t)
                        for mv in movement_detector:
                            if currentClass_t == mv[0] and conf > 0.3 and cx - 100 < mv[1] < cx + 100 and cy - 100 < mv[
                                2] < cy + 100:
                                cvzone.putTextRect(img_t, f'{currentClass_t} {conf} Area:{format(area_px_t_dummy[i])}',
                                                   (max(0, x1), max(35, y1)),
                                                   scale=1, thickness=1)  # Class Name
                                allArray_t.append([x1, y1, x2, y2, currentClass_t])
                                currentArray_t = np.array([x1, y1, x2, y2, conf])
                                detections_t = np.vstack((detections_t, currentArray_t))
                                area_px_t.append(area_px_t_dummy[i])
                                found_ele_t.append([str(currentClass_t), cx, cy])

                        i = i + 1
                        # print("----area_px_top_____",area_px_t)
            else:
                break

            resultsTracker_s = tracker_s.update(detections_s) #update the tracker with new detections obtained from the current frame.
            resultsTracker_t = tracker_t.update(detections_t)

            cnt_s = 0
            cnt_t = 0

            print("removed t,s",Products_removed_t,Products_removed_s,Hide_remove)
            print("added t,s",Products_added_t,Products_added_s,Hide)
            print("Left", Left)
            print("Right", Right)
            print("Top", Top)
            print("Bottom", Bottom)





            # --TOP-VIEW-TRACK----------------------------------------------
            for result in resultsTracker_t:

                # print(cnt)
                # print(classArray[cnt])

                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # print(x1, y1, x2, y2)

                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img_t, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

                # -------To Get the CurrentClass for the Objects detected--------------------------------
                for r in allArray_t:
                    if (r[0] - 50 < x1 < r[0] + 50 and r[1] - 50 < y1 < r[1] + 50 and r[2] - 50 < x2 < r[2] + 50 and r[
                        3] - 50 < y2 < r[3] + 50):
                        currentClass_t = r[4]

                # -------------- Bounding Box for the objects inside the CART ----------------------------
                if left_limits1[0] < cx < right_limits1[0] and top_limits1[1] < cy < bottom_limits1[1]:
                    cvzone.putTextRect(img_t, f' {int(id)}', (max(0, cx), max(35, cy)), scale=1, thickness=1)
                    cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # left-LINE---------------------------------------------------------------------------
                # ------LEFT OUTER LIMIT-----------------------------------------
                if left_limits1[0] - 25 < cx < left_limits1[2] + 25 and left_limits1[1] < cy < left_limits1[3]:
                    cv2.line(img_t, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 255, 0), 5)

                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1")

                    else:
                        # ---------- REMOVE ITEM ------------------

                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (
                                        x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    #    break
                                    #        else:
                                    #    Products_added_t.append(currentClass)
                                    break

                            if ovr_flg_t == 0:
                                Products_removed_t.append(currentClass_t)

                            print("remmove")

                            Result = []
                            # -----------------Union Products----------------
                            temp_top = []
                            temp_side = []
                            union_pro = []
                            temp_top.extend(Products_removed_s)
                            temp_side.extend(Products_removed_t)
                            union_pro.extend(temp_top)
                            for element in temp_side:
                                if element in temp_top:
                                    temp_top.remove(element)
                                else:
                                    union_pro.append(element)
                            print(union_pro)
                            # union_pro = list(set(Products_removed_t + Products_removed_s))

                            print(Products_removed_t, Products_removed_s, Hide_remove)
                            print(Final, Free)

                            # ------------Repeated Deletion ----------------
                            if currentClass_t in Products_added_t:
                                Products_added_t.remove(currentClass_t)

                            else:

                                # ----------------DISCOUNT OFFER-ALONE-----------------------------
                                if Original_Product in union_pro and Discount_Product in union_pro and len(union_pro) == 2:
                                    Final.remove(Original_Product)
                                    Free.remove(Discount_Product)
                                    Hide_remove.extend(Products_removed_s)
                                    Hide_remove_time = int(fps)
                                    Result.extend(union_pro)
                                # -----------------------------------------------
                                else:
                                    if len(Products_removed_s) == 0:

                                        if currentClass_t in Hide_remove:
                                            Hide_remove.remove(currentClass_t)
                                        else:
                                            for ele in Products_removed_t:
                                                if ele in Final:
                                                    Final.remove(ele)
                                                    Result.append(ele)
                                                else:
                                                    Free.remove(ele)
                                                    Result.append(ele)

                                    else:
                                        intersection = list(set(Products_removed_s) & set(Products_removed_t))
                                        print(intersection)
                                        if len(intersection) == 0:

                                            if currentClass_t in Hide_remove:
                                                Hide_remove.remove(currentClass_t)
                                            else:
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time = int(fps)
                                                for ele in union_pro:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                        Result.append(ele)
                                                    else:
                                                        Free.remove(ele)
                                                        Result.append(ele)


                                        else:
                                            if len(Products_removed_s) >= len(Products_removed_t):

                                                Result.extend(union_pro)

                                                for ele in Result:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                    else:
                                                        Free.remove(ele)

                                                for H_ele in Products_removed_t:
                                                    if H_ele in Products_removed_s:
                                                        Products_removed_s.remove(H_ele)
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time = int(fps)
                            Products_removed_t = []
                            Products_removed_s = []

                            # ----------------DISCOUNT-OFFER + OTHER---------------
                            Free_tmp = []
                            print(Result)
                            Org_Cnt = 0
                            for ele in Result:
                                print(ele, Original_Product)
                                if ele == Original_Product:
                                    Org_Cnt = Org_Cnt + 1

                            print(Org_Cnt)

                            for ele in Result:
                                if ele == Discount_Product and Org_Cnt > 0:
                                    Org_Cnt = Org_Cnt - 1
                                    Free_tmp.append(Discount_Product)
                            print(Free)

                            #--UPDATE DATABASE----------
                            for ele in Result:
                                value_to_select = ele

                                for r in Segment_remove:
                                    if value_to_select in r:
                                        value_to_select = r

                                    query = "UPDATE products SET Stock = Stock+1 WHERE Name=%s"
                                    cursor.execute(query, (value_to_select,))
                                    query = "select Stock,Amount from products where Name=%s"
                                    cursor.execute(query, (value_to_select,))
                                    table = cursor.fetchall()
                                    connection.commit()
                                    for row in table:
                                        kpi4_text.write(f"<h1  style='color:red;'>{value_to_select}</h1>",
                                                        unsafe_allow_html=True)
                                        kpi3_text.write(f"<h1  style='color:red;'>{row[0]}</h1>",
                                                        unsafe_allow_html=True)
                                        kpi2_text.write(
                                            f"<h1 style='color:red;'>{'{:.1f}'.format(row[1])}</h1>",
                                            unsafe_allow_html=True)
                                        if len(Free_tmp)>0 and ele==Discount_Product:
                                            Free_tmp.remove(Discount_Product)
                                        else:
                                            current_total = current_total - row[1]
                                            U_Final.remove(value_to_select)
                                        kpi1_text.write(
                                            f"<h1 style='color:black;'>{'{:.1f}'.format(current_total)}</h1>",
                                            unsafe_allow_html=True)

                # -------LEFT INNER LIMIT------------------------------------------------------

                if left_limits2[0] - 25 < cx < left_limits2[2] + 25 and left_limits2[1] < cy < left_limits2[3]:
                    cv2.line(img_t, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]),
                                 (0, 255, 0), 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                            # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:

                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            # if currentClass=="Mango":
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                        #    if (ovr_cls=="Maa_Juice"):
                                        ovr_flg_t = 1
                                        #    break
                                        #        else:
                                        #    Products_added_t.append(currentClass)
                                        break

                            if ovr_flg_t == 0:
                                Products_added_t.append(currentClass_t)
                                # -----Directions Added-----------------
                                Left.append(currentClass_t)
                                Left.append(fps)

                # right----------------------------------------------
                # ------RIGHT OUTER LIMIT-----------------------------------------
                if right_limits1[0] + 25 > cx > right_limits1[2] - 25 and right_limits1[1] < cy < right_limits1[3]:
                    cv2.line(img_t, (right_limits1[0], right_limits1[1]), (right_limits1[2], right_limits1[3]),
                                 (0, 255, 0),
                                 5)
                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                            # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1-right")

                    else:
                        # ---------- REMOVE ITEM ------------------
                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    ovr_flg_t = 1
                                    break

                            if ovr_flg_t == 0:
                                Products_removed_t.append(currentClass_t)

                            print("remmove")
                            Result = []
                            # -----------------Union Products----------------
                            temp_top = []
                            temp_side = []
                            union_pro = []
                            temp_top.extend(Products_removed_s)
                            temp_side.extend(Products_removed_t)
                            union_pro.extend(temp_top)
                            for element in temp_side:
                                if element in temp_top:
                                    temp_top.remove(element)
                                else:
                                    union_pro.append(element)
                            print(union_pro)

                            # ------------Repeated Deletion ----------------
                            if currentClass_t in Products_added_t:
                                Products_added_t.remove(currentClass_t)

                            else:

                                # ----------------DISCOUNT OFFER-ALONE-----------------------------
                                if Original_Product in union_pro and Discount_Product in union_pro and len(union_pro) == 2:
                                    Final.remove(Original_Product)
                                    Free.remove(Discount_Product)
                                    Hide_remove.extend(Products_removed_s)
                                    Hide_remove_time = int(fps)
                                    Result.extend(union_pro)
                                # -----------------------------------------------
                                else:
                                    if len(Products_removed_s) == 0:
                                        if currentClass_t in Hide_remove:
                                            Hide_remove.remove(currentClass_t)
                                        else:
                                            for ele in Products_removed_t:
                                                if ele in Final:
                                                    Final.remove(ele)
                                                    Result.append(ele)
                                                else:
                                                    Free.remove(ele)
                                                    Result.append(ele)

                                    else:
                                        intersection = list(set(Products_removed_s) & set(Products_removed_t))
                                        print(intersection)
                                        if len(intersection) == 0:

                                            if currentClass_t in Hide_remove:
                                                Hide_remove.remove(currentClass_t)
                                            else:
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time = fps
                                                for ele in union_pro:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                        Result.append(ele)
                                                    else:
                                                        Free.remove(ele)
                                                        Result.append(ele)


                                        else:
                                            if len(Products_removed_s) >= len(Products_removed_t):
                                                Result.extend(union_pro)

                                                for ele in Result:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                    else:
                                                        Free.remove(ele)

                                                for H_ele in Products_removed_t:
                                                    if H_ele in Products_removed_s:
                                                        Products_removed_s.remove(H_ele)
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time = fps

                            Products_removed_t = []
                            Products_removed_s = []

                            # ----------------DISCOUNT-OFFER + OTHER---------------
                            Free_tmp = []
                            print(Result)
                            Org_Cnt = 0
                            for ele in Result:
                                print(ele, Original_Product)
                                if ele == Original_Product:
                                    Org_Cnt = Org_Cnt + 1

                            print(Org_Cnt)

                            for ele in Result:
                                if ele == Discount_Product and Org_Cnt > 0:
                                    Org_Cnt = Org_Cnt - 1
                                    Free_tmp.append(Discount_Product)
                            print(Free)

                            # --UPDATE DATABASE----------
                            for ele in Result:
                                value_to_select = ele

                                for r in Segment_remove:
                                    if value_to_select in r:
                                        value_to_select = r

                                query = "UPDATE products SET Stock = Stock+1 WHERE Name=%s"
                                cursor.execute(query, (value_to_select,))
                                query = "select Stock,Amount from products where Name=%s"
                                cursor.execute(query, (value_to_select,))
                                table = cursor.fetchall()
                                connection.commit()
                                for row in table:
                                    kpi4_text.write(f"<h1  style='color:red;'>{value_to_select}</h1>",
                                                        unsafe_allow_html=True)
                                    kpi3_text.write(f"<h1  style='color:red;'>{row[0]}</h1>",
                                                        unsafe_allow_html=True)
                                    kpi2_text.write(f"<h1 style='color:red;'>{'{:.1f}'.format(row[1])}</h1>",
                                            unsafe_allow_html=True)
                                    if len(Free_tmp) > 0 and ele == Discount_Product:
                                        Free_tmp.remove(Discount_Product)
                                    else:
                                        current_total = current_total - row[1]
                                        U_Final.remove(value_to_select)
                                    kpi1_text.write(f"<h1 style='color:black;'>{'{:.1f}'.format(current_total)}</h1>",
                                            unsafe_allow_html=True)

                # --------RIGHT INNER LIMIT ---------------------------------------
                if right_limits2[0] + 25 > cx > right_limits2[2] - 25 and right_limits2[1] < cy < right_limits2[3]:
                    cv2.line(img_t, (right_limits2[0], right_limits2[1]), (right_limits2[2], right_limits2[3]),
                                 (0, 255, 0),
                                 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:
                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            # if currentClass=="Mango":
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    break

                            if ovr_flg_t == 0:
                                Products_added_t.append(currentClass_t)
                                # -----Directions Added-----------------
                                Right.append(currentClass_t)
                                Right.append(fps)

                # ----- TOP----------------------------------------------------------------------------------------------------------
                # --------------------TOP-OUTER LINE-----------------------------------------

                if top_limits1[0] < cx < top_limits1[2] and top_limits1[1] - 25 < cy < top_limits1[3] + 25:
                    cv2.line(img_t, (top_limits1[0], top_limits1[1]), (top_limits1[2], top_limits1[3]), (0, 255, 0),
                                 5)
                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                            # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1")

                    else:
                        # ---------- REMOVE ITEM ------------------
                        if out_line_t.count(id) == 0 and in_line_t.count(
                                    id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    break

                                ############################TESTING###########################################################
                                # if ovr_flg_t == 0 and currentClass!='Patanjali_Dant_Kanti':

                            if ovr_flg_t == 0:
                                Products_removed_t.append(currentClass_t)

                            print("remmove")

                            Result = []
                            # -----------------Union Products----------------
                            temp_top = []
                            temp_side = []
                            union_pro = []
                            temp_top.extend(Products_removed_s)
                            temp_side.extend(Products_removed_t)
                            union_pro.extend(temp_top)
                            for element in temp_side:
                                if element in temp_top:
                                    temp_top.remove(element)
                                else:
                                    union_pro.append(element)
                            print(union_pro)

                            # ------------Repeated Deletion ----------------
                            if currentClass_t in Products_added_t:
                                Products_added_t.remove(currentClass_t)

                            else:

                                # ----------------DISCOUNT OFFER-ALONE-----------------------------
                                if Original_Product in union_pro and Discount_Product in union_pro and len(union_pro) == 2:
                                    Final.remove(Original_Product)
                                    Free.remove(Discount_Product)
                                    Hide_remove.extend(Products_removed_s)
                                    Hide_remove_time=fps
                                    Result.extend(union_pro)
                                # -----------------------------------------------
                                else:
                                    if len(Products_removed_s) == 0:

                                        if currentClass_t in Hide_remove:
                                            Hide_remove.remove(currentClass_t)
                                        else:
                                            for ele in Products_removed_t:
                                                if ele in Final:
                                                    Final.remove(ele)
                                                    Result.append(ele)
                                                else:
                                                    Free.remove(ele)
                                                    Result.append(ele)

                                    else:
                                        intersection = list(set(Products_removed_s) & set(Products_removed_t))
                                        print(intersection)
                                        if len(intersection) == 0:

                                            if currentClass_t in Hide_remove:
                                                Hide_remove.remove(currentClass_t)
                                            else:
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time=fps
                                                for ele in union_pro:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                        Result.append(ele)
                                                    else:
                                                        Free.remove(ele)
                                                        Result.append(ele)


                                        else:
                                            if len(Products_removed_s) >= len(Products_removed_t):

                                                Result.extend(union_pro)

                                                for ele in Result:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                    else:
                                                        Free.remove(ele)

                                                for H_ele in Products_removed_t:
                                                    if H_ele in Products_removed_s:
                                                        Products_removed_s.remove(H_ele)
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time=fps

                            Products_removed_t = []
                            Products_removed_s = []

                            # ----------------DISCOUNT-OFFER + OTHER---------------
                            Free_tmp = []
                            print(Result)
                            Org_Cnt = 0
                            for ele in Result:
                                print(ele, Original_Product)
                                if ele == Original_Product:
                                    Org_Cnt = Org_Cnt + 1

                            print(Org_Cnt)

                            for ele in Result:
                                if ele == Discount_Product and Org_Cnt > 0:
                                    Org_Cnt = Org_Cnt - 1
                                    Free_tmp.append(Discount_Product)
                            print(Free)

                            # --UPDATE DATABASE----------
                            for ele in Result:
                                value_to_select = ele

                                for r in Segment_remove:
                                    if value_to_select in r:
                                        value_to_select = r

                                query = "UPDATE products SET Stock = Stock+1 WHERE Name=%s"
                                cursor.execute(query, (value_to_select,))
                                query = "select Stock,Amount from products where Name=%s"
                                cursor.execute(query, (value_to_select,))
                                table = cursor.fetchall()
                                connection.commit()
                                for row in table:
                                    kpi4_text.write(f"<h1  style='color:red;'>{value_to_select}</h1>",
                                                        unsafe_allow_html=True)
                                    kpi3_text.write(f"<h1  style='color:red;'>{row[0]}</h1>",
                                                        unsafe_allow_html=True)
                                    kpi2_text.write(f"<h1 style='color:red;'>{'{:.1f}'.format(row[1])}</h1>",
                                            unsafe_allow_html=True)
                                    if len(Free_tmp) > 0 and ele == Discount_Product:
                                        Free_tmp.remove(Discount_Product)
                                    else:
                                        current_total = current_total - row[1]
                                        U_Final.remove(value_to_select)
                                        kpi1_text.write(f"<h1 style='color:black;'>{'{:.1f}'.format(current_total)}</h1>",
                                            unsafe_allow_html=True)

                # -------TOP INNER LIMIT------------------------------------------------------
                if top_limits2[0]  < cx < top_limits2[2]  and top_limits2[1] - 25 < cy < top_limits2[3] + 25:
                    cv2.line(img_t, (top_limits2[0], top_limits2[1]), (top_limits2[2], top_limits2[3]),
                                 (0, 255, 0), 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:
                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            # if currentClass=="Mango":
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    break

                            if ovr_flg_t == 0:
                                Products_added_t.append(currentClass_t)
                                # -----Directions Added-----------------
                                Top.append(currentClass_t)
                                Top.append(fps)

                # -----BOTTOM -------------------------------------------
                # ---------------------BOTTOM OUTER LIMIT------------------------------------------------------------
                if bottom_limits1[0] < cx < bottom_limits1[2] and bottom_limits1[1] - 25 < cy < bottom_limits1[3] + 25:
                    cv2.line(img_t, (bottom_limits1[0], bottom_limits1[1]), (bottom_limits1[2], bottom_limits1[3]),
                                 (0, 255, 0), 5)
                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1")

                    else:
                        # ---------- REMOVE ITEM ------------------
                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    break

                            if ovr_flg_t == 0:
                                Products_removed_t.append(currentClass_t)

                            print("remmove")

                            Result = []
                            # -----------------Union Products----------------
                            temp_top = []
                            temp_side = []
                            union_pro = []
                            temp_top.extend(Products_removed_s)
                            temp_side.extend(Products_removed_t)
                            union_pro.extend(temp_top)
                            for element in temp_side:
                                if element in temp_top:
                                    temp_top.remove(element)
                                else:
                                    union_pro.append(element)
                            print(union_pro)
                            # union_pro = list(set(Products_removed_t + Products_removed_s))

                            print(Products_removed_t, Products_removed_s, Hide_remove)
                            print(Final, Free)

                            # ------------Repeated Deletion ----------------
                            if currentClass_t in Products_added_t:
                                Products_added_t.remove(currentClass_t)

                            else:

                                # ----------------DISCOUNT OFFER-ALONE-----------------------------
                                if Original_Product in union_pro and Discount_Product in union_pro and len(union_pro) == 2:
                                    Final.remove(Original_Product)
                                    Free.remove(Discount_Product)
                                    Hide_remove.extend(Products_removed_s)
                                    Hide_remove_time = fps
                                    Result.extend(union_pro)
                                # -----------------------------------------------
                                else:
                                    if len(Products_removed_s) == 0:

                                        if currentClass_t in Hide_remove:
                                            Hide_remove.remove(currentClass_t)
                                        else:
                                            for ele in Products_removed_t:
                                                if ele in Final:
                                                    Final.remove(ele)
                                                    Result.append(ele)
                                                else:
                                                    Free.remove(ele)
                                                    Result.append(ele)

                                    else:
                                        intersection = list(set(Products_removed_s) & set(Products_removed_t))
                                        print(intersection)
                                        if len(intersection) == 0:
                                            if currentClass_t in Hide_remove:
                                                Hide_remove.remove(currentClass_t)
                                            else:
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time=fps
                                                for ele in union_pro:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                        Result.append(ele)
                                                    else:
                                                        Free.remove(ele)
                                                        Result.append(ele)


                                        else:
                                            if len(Products_removed_s) >= len(Products_removed_t):

                                                Result.extend(union_pro)

                                                for ele in Result:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                    else:
                                                        Free.remove(ele)

                                                for H_ele in Products_removed_t:
                                                    if H_ele in Products_removed_s:
                                                        Products_removed_s.remove(H_ele)
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time=fps

                            Products_removed_t = []
                            Products_removed_s = []

                            # ----------------DISCOUNT-OFFER + OTHER---------------
                            Free_tmp = []
                            print(Result)
                            Org_Cnt = 0
                            for ele in Result:
                                print(ele, Original_Product)
                                if ele == Original_Product:
                                    Org_Cnt = Org_Cnt + 1
                            print(Org_Cnt)

                            for ele in Result:
                                if ele == Discount_Product and Org_Cnt > 0:
                                    Org_Cnt = Org_Cnt - 1
                                    Free_tmp.append(Discount_Product)
                            print(Free)

                            # --UPDATE DATABASE----------
                            for ele in Result:
                                value_to_select = ele

                                for r in Segment_remove:
                                    if value_to_select in r:
                                        value_to_select = r

                                query = "UPDATE products SET Stock = Stock+1 WHERE Name=%s"
                                cursor.execute(query, (value_to_select,))
                                query = "select Stock,Amount from products where Name=%s"
                                cursor.execute(query, (value_to_select,))
                                table = cursor.fetchall()
                                connection.commit()
                                for row in table:
                                    kpi4_text.write(f"<h1  style='color:red;'>{value_to_select}</h1>",
                                                        unsafe_allow_html=True)
                                    kpi3_text.write(f"<h1  style='color:red;'>{row[0]}</h1>",
                                                        unsafe_allow_html=True)
                                    kpi2_text.write(f"<h1 style='color:red;'>{'{:.1f}'.format(row[1])}</h1>",
                                            unsafe_allow_html=True)

                                    if len(Free_tmp) > 0 and ele == Discount_Product:
                                        Free_tmp.remove(Discount_Product)
                                    else:
                                        current_total = current_total - row[1]
                                        U_Final.remove(value_to_select)
                                    kpi1_text.write(f"<h1 style='color:black;'>{'{:.1f}'.format(current_total)}</h1>",
                                            unsafe_allow_html=True)

                # -------BOTTOM INNER LIMIT------------------------------------------------------
                if bottom_limits2[0] - 25 < cx < bottom_limits2[2] + 25 and bottom_limits2[1] < cy < bottom_limits2[3]:
                    cv2.line(img_t, (bottom_limits2[0], bottom_limits2[1]), (bottom_limits2[2], bottom_limits2[3]),
                             (0, 255, 0), 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:
                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)

                            # ---------Overlap Bounding Box---------
                            # if currentClass=="Mango":
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (
                                        x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    break

                            if ovr_flg_t == 0:
                                Products_added_t.append(currentClass_t)
                                # -----Directions Added-----------------
                                Bottom.append(currentClass_t)
                                Bottom.append(fps)

            # ----SIDE VIEW------------------------------------------------------------------------------------------------------
            for result in resultsTracker_s:

                    # print(cnt_s)
                    # print(classArray[cnt])

                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    print("-----area-top------", area_px_t)
                    print("-----area-side-----", area_px_s)
                    print("-----found-Side----", found_ele_s)

                    for r in allArray_s:
                        if (r[0] - 50 < x1 < r[0] + 50 and r[1] - 50 < y1 < r[1] + 50 and r[2] - 50 < x2 < r[2] + 50 and r[3] - 50 < y2 < r[3] + 50):
                            currentClass_s = r[4]

                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img_s, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

                    # if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:

                    #-----BOUNDING BOX FOR OBJECTS INSIDE CART--------------------------
                    if top_limits1_s[1] < cy:
                        cvzone.putTextRect(img_s, f' {int(id)}', (max(0, cx), max(35, cy)), scale=1.5, thickness=2,offset=10)
                        cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 255, 0), 2)


                    # TOP-OUTER LIMIT-----------------------------------------------------------

                    if top_limits1_s[0] < cx < top_limits1_s[2] and top_limits1_s[1] - 25 < cy < top_limits1_s[3] + 25:
                        cv2.line(img_s, (top_limits1_s[0], top_limits1_s[1]), (top_limits1_s[2], top_limits1_s[3]),(0, 255, 0), 5)

                        if out_line_s.count(id) == 0 and in_line_s.count(id) == 0:
                            # if totalCount.count(id) == 0:
                            out_line_s.append(id)
                            print("out-1")
                        else:
                            # REMOVE --------------------------------
                            if out_line_s.count(id) == 0 and in_line_s.count(id) == 1:
                                print("out-2")
                                Total_products_s = Total_products_s - 1
                                in_line_s.remove(id)

                                # ---------Overlap Bounding Box---------
                                ovr_flg_s = 0
                                for ovr_ele in allArray_s:
                                    lx1 = ovr_ele[0]
                                    ly1 = ovr_ele[1]
                                    lx2 = ovr_ele[2]
                                    ly2 = ovr_ele[3]
                                    ovr_cls = ovr_ele[4]
                                    if (x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass):
                                        print("overlap", currentClass)

                                        #    if (ovr_cls=="Maa_Juice"):
                                        ovr_flg_s = 1
                                        break

                                if ovr_flg_s == 0:
                                    print("NO NO NO Overlap", currentClass_s)
                                    Products_removed_s.append(currentClass_s)

                                    value_to_select = currentClass_s
                                    ele = value_to_select
                                    for r in results_t:
                                        boxes = r.boxes
                                        lt = len(area_px_t)
                                        i = 0

                                        for box in boxes:
                                            x1, y1, x2, y2 = box.xyxy[0]
                                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                            w, h = x2 - x1, y2 - y1
                                            cx, cy = x1 + w // 2, y1 + h // 2

                                            if cy < 650:

                                                cls = int(box.cls[0])

                                                # query = "select Name,High,Low from products where Name LIKE %s"
                                                # cursor.execute(query, (value_to_select,))

                                                for i in range(0, len(area_px_t)):

                                                    top_fnd = found_ele_t[i][0]
                                                    cxt = found_ele_t[i][1]
                                                    cyt = found_ele_t[i][2]

                                                    if top_fnd == value_to_select and value_to_select == classNames[cls] and cx - 50 < cxt < cx + 50 and cy - 50 < cyt < cy + 50:

                                                        print("---------segment_top-REMOVE---------", classNames[cls])
                                                        print(area_px_t[i])

                                                        query = "select Name,High,Low from products where Name LIKE %s"
                                                        cursor.execute(query, ("%" + value_to_select + "%",))
                                                        table = cursor.fetchall()
                                                        area_intersection = 0
                                                        area_value = ""
                                                        for row in table:
                                                            print("!!!sql!!! ", row[0])
                                                            if classNames[cls] == ele and (
                                                                    row[2] + 5000 < area_px_t[i] < row[1] + 5000 or row[
                                                                2] - 5000 <
                                                                    area_px_t[i] < row[1] - 5000):
                                                                area_intersection += 1
                                                                tmp = str(row[0])
                                                                temp_area_t = area_px_t[i]

                                                        if area_intersection > 1:

                                                            ls = len(area_px_s)
                                                            j = 0

                                                            for j in range(0, ls):
                                                                side_fnd = found_ele_s[j][0]
                                                                cxs = found_ele_s[j][1]
                                                                cys = found_ele_s[j][2]
                                                                if classNames[cls] == side_fnd and side_fnd == ele:
                                                                    print(
                                                                        "!!!!!!!!!!!!!!!!!!!!!!!!!!EEERRRRORRRRRR!!!!!!!!!!!!!!!!!!!!!!!",
                                                                        side_fnd, area_px_s[j])
                                                                    # ----------------Calculating area based on the distance-------------------
                                                                    a2 = area_px_s[j]
                                                                    d2 = cx - 50
                                                                    d1 = 1150
                                                                    a1 = int(a2 * (pow(d2 / d1, 2)))

                                                                    query = "select Name,Side_Low,Side_High from products where Name LIKE %s"
                                                                    cursor.execute(query,
                                                                                   ("%" + value_to_select + "%",))
                                                                    table = cursor.fetchall()

                                                                    new_intersection = 0
                                                                    new_area = 1000000000000000
                                                                    new_tmp = ""

                                                                    for row in table:
                                                                        print("!!!side!!! ", row[0])
                                                                        print("-----A1-----", a1)
                                                                        print("-----RANGE------", row[1], row[2])
                                                                        if row[1] - 5000 < a1 < row[2] + 5000:
                                                                            new_tmp = str(row[0])
                                                                            new_intersection += 1

                                                                            if str(new_tmp[
                                                                                       -1]) == "L" and new_area > int(
                                                                                    abs(a1 - row[1])):
                                                                                print("11111111111111111111")
                                                                                new_area = int(abs(a1 - row[1]))
                                                                                value_to_select = new_tmp

                                                                            if str(new_tmp[
                                                                                       -1]) == "S" and new_area > int(
                                                                                    abs(row[2] - a1)):
                                                                                print("22222222222222222222")
                                                                                new_area = int(abs(row[2] - a1))
                                                                                value_to_select = new_tmp

                                                        else:
                                                            value_to_select = str(tmp)

                                                        print("!!!!!!!!!sql-remove!!!!!!!!!1!! ", value_to_select,
                                                              temp_area_t)
                                                        Segment_remove.append(value_to_select)


                                print("remove-Side", Products_removed_s)

                    # TOP-INNER-LIMIT----------------------------------------------------------------

                    if top_limits2_s[0] < cx < top_limits2_s[2] and top_limits2_s[1] - 25 < cy < top_limits2_s[
                        3] + 25:
                        cv2.line(img_s, (top_limits2_s[0], top_limits2_s[1]), (top_limits2_s[2], top_limits2_s[3]),
                                 (0, 255, 0), 5)

                        if in_line_s.count(id) == 0 and out_line_s.count(id) == 0:
                            # if totalCount.count(id) == 0:
                            print("in-1")
                            in_line_s.append(id)

                        else:
                            # ADD --------------------------------
                            if in_line_s.count(id) == 0 and out_line_s.count(id) == 1:
                                print("in-2")
                                Total_products_s = Total_products_s + 1
                                out_line_s.remove(id)

                                Products_added_t_dummy = []
                                for i in Products_added_t:
                                    Products_added_t_dummy.append(i)

                                print("----dummmy-----", Products_added_t_dummy)
                                # ---------Overlap Bounding Box---------
                                ovr_flg_s = 0
                                for ovr_ele in allArray_s:
                                    lx1 = ovr_ele[0]
                                    ly1 = ovr_ele[1]
                                    lx2 = ovr_ele[2]
                                    ly2 = ovr_ele[3]
                                    ovr_cls = ovr_ele[4]
                                    print(x1, lx1, x2, lx2, y1, ly1, y2, ly2, ovr_cls, currentClass_s)
                                    if (
                                            x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_s):
                                        #    if (ovr_cls=="Maa_Juice"):
                                        ovr_flg_s = 1
                                        #    break
                                        #        else:
                                        #    Products_added_t.append(currentClass)
                                        break

                                if ovr_flg_s == 0:
                                    Products_added_s.append(currentClass_s)

                                print("presennt", Products_added_t, Products_added_s, Hide)
                                Result = []

                                # -----------------UNION Products----------------
                                temp_top = []
                                temp_side = []
                                union_pro = []
                                temp_top.extend(Products_added_s)
                                temp_side.extend(Products_added_t)
                                union_pro.extend(temp_top)
                                for element in temp_side:
                                    if element in temp_top:
                                        temp_top.remove(element)
                                    else:
                                        union_pro.append(element)
                                print(union_pro)

                                intersection = list(set(Products_added_t) & set(Products_added_s))
                                # union_pro = list(set(Products_added_t + Products_added_s))

                                # -----------------Repeated Addition--------------
                                if currentClass_s in Products_removed_s:
                                    Products_removed_s.remove(currentClass_s)

                                else:
                                    # ------------------- UNION PRODUCTS-------------------------------
                                    if Original_Product in union_pro and Discount_Product in union_pro and (
                                            len(intersection) == 0 or (
                                            Original_Product in Left and Discount_Product in Left) or (
                                                    Original_Product in Right and Discount_Product in Right) or (
                                                    Original_Product in Top and Discount_Product in Top) or (
                                                    Original_Product in Bottom and Discount_Product in Bottom)):
                                        if len(intersection) == 0:
                                            if len(union_pro) == 2:  # ------------ONLY COMBO PACK--------------------
                                                Final.append(Original_Product)
                                                Free.append(Discount_Product)
                                                Hide.extend(Products_added_t)
                                                Hide_add_time=fps
                                                Result.extend(union_pro)
                                            # else: ---------- COMBO + MULTI HAND -------------------------------

                                        else:
                                            if (Original_Product in Left and Discount_Product in Left and abs(
                                                    int(Left[Left.index(Original_Product) + 1]) - int(
                                                            Left[Left.index(Discount_Product) + 1])) < 25) or (
                                                    Original_Product in Right and Discount_Product in Right and abs(
                                                    int(Right[Right.index(Original_Product) + 1]) - int(
                                                            Right[Right.index(Discount_Product) + 1])) < 25) or (
                                                    Original_Product in Top and Discount_Product in Top and abs(
                                                    int(Top[Top.index(Original_Product) + 1]) - int(
                                                            Top[Top.index(Discount_Product) + 1])) < 25) or (
                                                    Original_Product in Bottom and Discount_Product in Bottom and abs(
                                                    int(Bottom[Bottom.index(Original_Product) + 1]) - int(
                                                            Bottom[Bottom.index(Discount_Product) + 1])) < 25):
                                                # -----Considering NOT removed before added-----------------
                                                Final.append(Original_Product)
                                                Free.append(Discount_Product)

                                                # only remove need-------------------------------------
                                                Result.extend(union_pro)
                                                for H_ele in union_pro:
                                                    if H_ele == Original_Product or H_ele == Discount_Product:
                                                        union_pro.remove(H_ele)

                                                Hide.extend(union_pro)
                                                Hide_add_time=fps
                                            else:
                                                if len(Products_added_t) == 0:
                                                    if currentClass_s in Hide:
                                                        Hide.remove(currentClass_s)
                                                    else:
                                                        Final.extend(Products_added_s)
                                                        Result.extend(Products_added_s)
                                                else:
                                                    # intersection = list(set(Products_added_t) & set(Products_added_s))
                                                    print(intersection)
                                                    if len(intersection) == 0:
                                                        if currentClass_s in Hide:
                                                            Hide.remove(currentClass_s)
                                                        else:
                                                            Final.extend(union_pro)
                                                            Result.extend(union_pro)
                                                            Hide.extend(Products_added_t)
                                                            Hide_add_time = fps
                                                            # Result.extend(Products_added_t)

                                                    else:
                                                        if len(Products_added_t) >= len(Products_added_s):

                                                            Result.extend(union_pro)
                                                            Final.extend(Result)

                                                            # ----- Removing elements seen in side from top-----------
                                                            for H_ele in Products_added_s:
                                                                if H_ele in Products_added_t:
                                                                    Products_added_t.remove(H_ele)
                                                            Hide.extend(Products_added_t)
                                                            Hide_add_time = fps
                                            # ---------Direction remove--------------------------------
                                            Left = []
                                            Right = []
                                            Top = []
                                            Bottom = []
                                    else:
                                        if len(Products_added_t) == 0:
                                            if currentClass_s in Hide:
                                                Hide.remove(currentClass_s)
                                            else:
                                                Final.extend(Products_added_s)
                                                Result.extend(Products_added_s)
                                        else:
                                            # intersection = list(set(Products_added_t) & set(Products_added_s))
                                            print(intersection)
                                            if len(intersection) == 0:
                                                if currentClass_s in Hide:
                                                    Hide.remove(currentClass_s)
                                                else:
                                                    Final.extend(union_pro)
                                                    Result.extend(union_pro)
                                                    Hide.extend(Products_added_t)
                                                    Hide_add_time = fps
                                                    # Result.extend(Products_added_t)

                                            else:
                                                if len(Products_added_t) >= len(Products_added_s):

                                                    Result.extend(union_pro)
                                                    Final.extend(Result)

                                                    # ----- Removing elements seen in side from top-----------
                                                    for H_ele in Products_added_s:
                                                        if H_ele in Products_added_t:
                                                            Products_added_t.remove(H_ele)
                                                    Hide.extend(Products_added_t)
                                                    Hide_add_time = fps

                                # ----------------DISCOUNT OFFER + OTHER-----------------
                                Free_tmp = []
                                print(Result)
                                Org_Cnt = 0
                                for ele in Result:
                                    print(ele, Original_Product)
                                    if ele == Original_Product:
                                        Org_Cnt = Org_Cnt + 1

                                        print(Org_Cnt)

                                for ele in Result:
                                    if ele == Discount_Product and Org_Cnt > 0:
                                        Org_Cnt = Org_Cnt - 1
                                        Free_tmp.append(Discount_Product)
                                print(Free)

                                print("products T,S,H", Products_added_t_dummy, Products_added_s, Hide)

                                #----------------DIRECTIONS----------------------------
                                Left = []
                                Right = []
                                Top = []
                                Bottom = []


                                # --UPDATE DATABASE----------
                                for ele in Result:
                                    value_to_select = ele

                                    # ----BOTH PRESENT-----------

                                    if ele in Products_added_t_dummy and ele in Products_added_s:

                                        print("-----PRESENT IN BOTH-----------")

                                        for r in results_t:
                                            boxes = r.boxes
                                            lt = len(area_px_t)

                                            i = 0

                                            for box in boxes:
                                                x1, y1, x2, y2 = box.xyxy[0]
                                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                                w, h = x2 - x1, y2 - y1
                                                cx, cy = x1 + w // 2, y1 + h // 2

                                                cls = int(box.cls[0])

                                                # query = "select Name,High,Low from products where Name LIKE %s"
                                                # cursor.execute(query, (value_to_select,))

                                                for i in range(0, len(area_px_t)):
                                                    print(found_ele_t)
                                                    print(found_ele_t[i])
                                                    top_fnd = found_ele_t[i][0]
                                                    cxt = found_ele_t[i][1]
                                                    cyt = found_ele_t[i][2]

                                                    if top_fnd == value_to_select and value_to_select == classNames[
                                                        cls] and cx - 50 < cxt < cx + 50 and cy - 50 < cyt < cy + 50:

                                                        print("---------segment_top----------", classNames[cls])
                                                        print(area_px_t[i])

                                                        query = "select Name,High,Low from products where Name LIKE %s"
                                                        cursor.execute(query, ("%" + value_to_select + "%",))
                                                        table = cursor.fetchall()
                                                        area_intersection = 0
                                                        area_value = ""
                                                        for row in table:
                                                            print("!!!sql!!! ", row[0])
                                                            if classNames[cls] == ele and (
                                                                    row[2] + 5000 < area_px_t[i] < row[1] + 5000 or row[
                                                                2] - 5000 < area_px_t[i] < row[1] - 5000):
                                                                area_intersection += 1
                                                                tmp = str(row[0])
                                                                temp_area_t = area_px_t[i]

                                                        if area_intersection > 1:

                                                            ls = len(area_px_s)
                                                            j = 0

                                                            for j in range(0, ls):
                                                                side_fnd = found_ele_s[j][0]
                                                                cxs = found_ele_s[j][1]
                                                                cys = found_ele_s[j][2]
                                                                if classNames[cls] == side_fnd and side_fnd == ele:
                                                                    print(
                                                                        "!!!!!!!!!!!!!!!!!!!!!!!!!!EEERRRRORRRRRR!!!!!!!!!!!!!!!!!!!!!!!",
                                                                        side_fnd, area_px_s[j])
                                                                    # ----------------Calculating area based on the distance-------------------
                                                                    a2 = area_px_s[j]
                                                                    d2 = cx - 50
                                                                    d1 = 1150
                                                                    a1 = int(a2 * (pow(d2 / d1, 2)))

                                                                    query = "select Name,Side_Low,Side_High from products where Name LIKE %s"
                                                                    cursor.execute(query,
                                                                                   ("%" + value_to_select + "%",))
                                                                    table = cursor.fetchall()

                                                                    new_intersection = 0
                                                                    new_area = 1000000000000000
                                                                    new_tmp = ""

                                                                    for row in table:
                                                                        print("!!!side!!! ", row[0])
                                                                        print("-----A1-----", a1)
                                                                        print("-----RANGE------", row[1], row[2])
                                                                        if row[1] - 5000 < a1 < row[2] + 5000:
                                                                            new_tmp = str(row[0])
                                                                            new_intersection += 1

                                                                            if str(new_tmp[
                                                                                       -1]) == "L" and new_area > int(
                                                                                    abs(a1 - row[1])):
                                                                                print("11111111111111111111")
                                                                                new_area = int(abs(a1 - row[1]))
                                                                                value_to_select = new_tmp

                                                                            if str(new_tmp[
                                                                                       -1]) == "S" and new_area > int(
                                                                                    abs(row[2] - a1)):
                                                                                print("22222222222222222222")
                                                                                new_area = int(abs(row[2] - a1))
                                                                                value_to_select = new_tmp

                                                        else:
                                                            value_to_select = str(tmp)

                                                        print("new area, value to sel", value_to_select)

                                    # -----------PRESENT IN TOP-----------------------------------
                                    elif ele in Products_added_t_dummy and ele not in Products_added_s:

                                        print("-------PRESENT IN TOP-----------------")

                                        for r in results_t:
                                            boxes = r.boxes
                                            lt = len(area_px_t)

                                            i = 0

                                            for box in boxes:
                                                x1, y1, x2, y2 = box.xyxy[0]
                                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                                w, h = x2 - x1, y2 - y1
                                                cx, cy = x1 + w // 2, y1 + h // 2

                                                cls = int(box.cls[0])

                                                # query = "select Name,High,Low from products where Name LIKE %s"
                                                # cursor.execute(query, (value_to_select,))

                                                for i in range(0, len(area_px_t)):
                                                    top_fnd = found_ele_t[i][0]
                                                    cxt = found_ele_t[i][1]
                                                    cyt = found_ele_t[i][2]

                                                    if top_fnd == value_to_select and value_to_select == classNames[
                                                        cls] and cx - 50 < cxt < cx + 50 and cy - 50 < cyt < cy + 50:

                                                        print("---------segment_top----------", classNames[cls])
                                                        print(area_px_t[i])

                                                        query = "select Name,Low,High from products where Name LIKE %s"
                                                        cursor.execute(query, ("%" + value_to_select + "%",))
                                                        table = cursor.fetchall()

                                                        new_intersection = 0
                                                        new_area = 1000000000000000
                                                        new_tmp = ""
                                                        a1 = area_px_t[i]

                                                        for row in table:
                                                            print("!!!sql!!!!")
                                                            print("!!!side!!! ", row[0])
                                                            print("-----A1-----", a1)
                                                            print("-----RANGE------", row[1], row[2])
                                                            if row[1] - 5000 < a1 < row[2] + 5000:
                                                                new_tmp = str(row[0])
                                                                new_intersection += 1

                                                                if str(new_tmp[-1]) == "L" and new_area > int(
                                                                        abs(a1 - row[1])):
                                                                    print("11111111111111111111")
                                                                    new_area = int(abs(a1 - row[1]))
                                                                    value_to_select = new_tmp

                                                                if str(new_tmp[-1]) == "S" and new_area > int(
                                                                        abs(row[2] - a1)):
                                                                    print("22222222222222222222")
                                                                    new_area = int(abs(row[2] - a1))
                                                                    value_to_select = new_tmp

                                    # -----------PRESENT IN SIDE ------------------------
                                    elif ele not in Products_added_t_dummy and ele in Products_added_s:

                                        print("-----PRESENT IN SIDE-------------")
                                        for r in results_s:
                                            boxes = r.boxes
                                            ls = len(area_px_s)

                                            j = 0

                                            for box in boxes:
                                                x1, y1, x2, y2 = box.xyxy[0]
                                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                                w, h = x2 - x1, y2 - y1
                                                cx, cy = x1 + w // 2, y1 + h // 2

                                                cls = int(box.cls[0])

                                                # query = "select Name,High,Low from products where Name LIKE %s"
                                                # cursor.execute(query, (value_to_select,))

                                                for j in range(0, ls):
                                                    top_fnd = found_ele_s[j][0]
                                                    cxt = found_ele_s[j][1]
                                                    cyt = found_ele_s[j][2]

                                                    if top_fnd == value_to_select and value_to_select == classNames[
                                                        cls] and cx - 50 < cxt < cx + 50 and cy - 50 < cyt < cy + 50 and 400 < cy < 550:

                                                        print("---------segment_top----------", classNames[cls])
                                                        print(area_px_s[j])

                                                        query = "select Name,Side_low,Side_high from products where Name LIKE %s"
                                                        cursor.execute(query, ("%" + value_to_select + "%",))
                                                        table = cursor.fetchall()

                                                        new_intersection = 0
                                                        new_area = 1000000000000000
                                                        new_tmp = ""
                                                        a1 = area_px_s[j]

                                                        for row in table:
                                                            print("!!!sql!!!")
                                                            print("!!!side!!! ", row[0])
                                                            print("-----A1-----", a1)
                                                            print("-----RANGE------", row[1], row[2])
                                                            if row[1] - 5000 < a1 < row[2] + 5000:
                                                                new_tmp = str(row[0])
                                                                new_intersection += 1

                                                                if str(new_tmp[-1]) == "L" and new_area > int(
                                                                        abs(a1 - row[1])):
                                                                    print("11111111111111111111")
                                                                    new_area = int(abs(a1 - row[1]))
                                                                    value_to_select = new_tmp

                                                                if str(new_tmp[-1]) == "S" and new_area > int(
                                                                        abs(row[2] - a1)):
                                                                    print("22222222222222222222")
                                                                    new_area = int(abs(row[2] - a1))
                                                                    value_to_select = new_tmp

                                    print("current element ", value_to_select)

                                    #-----------------UPDATE----------------------------------------------

                                    query = "UPDATE products SET Stock = Stock-1 WHERE Name=%s"
                                    cursor.execute(query, (value_to_select,))
                                    query = "select Stock,Amount from products where Name=%s"
                                    cursor.execute(query, (value_to_select,))
                                    table = cursor.fetchall()
                                    connection.commit()
                                    for row in table:
                                        kpi4_text.write(f"<h1  style='color:red;'>{value_to_select}</h1>",unsafe_allow_html=True)
                                        kpi3_text.write(f"<h1  style='color:red;'>{row[0]}</h1>",unsafe_allow_html=True)
                                        kpi2_text.write(f"<h1 style='color:red;'>{'{:.1f}'.format(row[1])}</h1>",unsafe_allow_html=True)
                                        if len(Free_tmp)>0 and ele==Discount_Product:
                                            Free_tmp.remove(Discount_Product)
                                        else:
                                            current_total = current_total + row[1]
                                            U_Final.append(value_to_select)
                                        kpi1_text.write(f"<h1 style='color:black;'>{'{:.1f}'.format(current_total)}</h1>",unsafe_allow_html=True)

                                        ########## SMS Message ######## ADMIN FOR REFRESH #########################
                                        # SMS exchange ADMIN-------------------------------------------------
                                        if row[0]<=0:
                                            to_phone_number = '+919941815173'  # Replace with recipient's phone number
                                            message = '!!! EMPTY STOCK ALERT !!! Product_name: ' + value_to_select   # Your message
                                            send_sms(to_phone_number, message)
                                        elif row[0]<4:
                                            to_phone_number = '+919941815173'  # Replace with recipient's phone number
                                            message = '!!! LOW STOCK ALERT !!! Product_name: '+value_to_select+' Current_stock '+ str(row[0]) # Your message
                                            send_sms(to_phone_number, message)

                                Products_added_t = []
                                Products_added_s = []

                    cnt_s = cnt_s + 1

                    cnt_t = cnt_t + 1

            no_prod_s = 0
            no_prod_t = 0

            print(U_Final)

            # ---SIDE-VIEW-----------------------------------------
            # cv2.putText(img, str(Products_added), (1000, 100 + (no_prod * 100)), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
            occurrence_s = {item: U_Final.count(item) for item in U_Final}
            # print(occurrence_s.get('e'))
            item_added_s = []

            #----PRINT FINAL DETAILS----------------------
            for p in U_Final:
                if item_added_s.count(p) == 0:
                    cv2.putText(img_s, str(p) + " " + str(occurrence_s.get(p)),
                                    (int(1400 * frame_width), 60 + (no_prod_s * 60)), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (50, 50, 255), 4)
                    cv2.putText(img_t, str(p) + " " + str(occurrence_s.get(p)),
                                    (int(1400 * frame_width), 60 + (no_prod_s * 60)), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (50, 50, 255), 4)

                    # print(p)
                    item_added_s.append(p)
                    no_prod_s = no_prod_s + 1

            item_added_s = []
            no_prod_s = 1

            #-------PRINT FREE DETAILS--------------------
            for p in Free:
                if item_added_s.count(p) == 0:
                    cv2.putText(img_s, str(p),
                                    (int(20 * frame_width), 60 + (no_prod_s * 100)), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 0, 0), 4)
                    cv2.putText(img_t, str(p),
                                    (int(20 * frame_width), 60 + (no_prod_s * 100)), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 0, 0), 4)

                    # print(p)
                    item_added_s.append(p)
                    no_prod_s = no_prod_s + 1

            # ----Time-------------------
            # fps = time.time()
            # fps = fps - start

            cv2.putText(img_s, "TOTAL: " + str(int(current_total)), (20, 360), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            cv2.putText(img_t, "TOTAL: " + str(int(current_total)), (20, 360), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)

            cv2.putText(img_s, "Count: " + str(len(U_Final)), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            cv2.putText(img_s, "Free: " + str(len(Free)), (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)

            cv2.putText(img_t, "Count: " + str(len(U_Final)), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            cv2.putText(img_t, "Free: " + str(len(Free)), (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)

            # cv2.putText(img_t, "FPS: " + str(int(fps)), (20, 180), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            # cv2.putText(img_s, "FPS: " + str(int(fps)), (20, 180), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            #
            # cv2.putText(img_t, "Hide_add: " + str(int(Hide_add_time)), (20, 240), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            # cv2.putText(img_t, "Hide_remove: " + str(int(Hide_remove_time)), (20, 300), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            #
            # cv2.putText(img_s, "Hide_add: " + str(int(Hide_add_time)), (20, 240), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
            # cv2.putText(img_s, "Hide_remove: " + str(int(Hide_remove_time)), (20, 300), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)

# -------------WINDOW-------------------
            cv2.line(img_t, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 0, 255), 1)
            cv2.line(img_t, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]), (255, 0, 0), 1)

            cv2.line(img_t, (right_limits1[0], right_limits1[1]), (right_limits1[2], right_limits1[3]), (0, 0, 255), 1)
            cv2.line(img_t, (right_limits2[0], right_limits2[1]), (right_limits2[2], right_limits2[3]), (255, 0, 0), 1)

            cv2.line(img_t, (top_limits1[0], top_limits1[1]), (top_limits1[2], top_limits1[3]), (0, 0, 255), 1)
            cv2.line(img_t, (top_limits2[0], top_limits2[1]), (top_limits2[2], top_limits2[3]), (255, 0, 0), 1)

            cv2.line(img_t, (bottom_limits1[0], bottom_limits1[1]), (bottom_limits1[2], bottom_limits1[3]), (0, 0, 255), 1)
            cv2.line(img_t, (bottom_limits2[0], bottom_limits2[1]), (bottom_limits2[2], bottom_limits2[3]), (255, 0, 0), 1)

            cv2.line(img_s, (top_limits1_s[0], top_limits1_s[1]), (top_limits1_s[2], top_limits1_s[3]), (0, 0, 255), 1)
            cv2.line(img_s, (top_limits2_s[0], top_limits2_s[1]), (top_limits2_s[2], top_limits2_s[3]), (255, 0, 0), 1)

            cv2.line(img_s, (top_limits3_s[0], top_limits3_s[1]), (top_limits3_s[2], top_limits3_s[3]), (255, 0, 0), 1)


            #resize_img_s = cv2.resize(img, (screen_width, screen_height), interpolation=cv2.INTER_NEAREST_EXACT)
            stframe_s.image(img_s, channels='BGR', use_column_width=True)

            stframe_t.image(img_t, channels='BGR', use_column_width=True)


    #print(success)

    # connection = mysql.connector.connect(host="localhost", user="root", password="", database="shop2")
    #
    # cursor = connection.cursor()

    occurrence = {item: U_Final.count(item) for item in U_Final}

    # fetch all columns
    print('\n Table Data:')

    print("Product ", U_Final)

    value = ""
    value = value + "--------------WELCOME TO XYZ Shop--------------\n\n"
    value = value + "RECIEPT\n\n"
    # value=value+"Item_Name Count Amount\n\n"
    total = 0
    #result = PrettyTable([' Item Name ', ' Count ', ' Amount '])
    value=value+"Item Name\t\t\t\tCount\t\t\t\tAmount\n\n"
    item_added=[]

    st.title("WELCOME TO XYZ SHOP")
    st.title("RECIEPT")
    #-----------------
    # st.markdown("<table><tr><th>Item Name</th><th>Count</th><th>Amount</th><th>Calculated</th></tr>",unsafe_allow_html=True)
    #
    # for p in U_Final:
    #     if item_added.count(p) == 0:
    #
    #         value_to_select = str(p)
    #         query = "select Amount from products where Name=%s"
    #         cursor.execute(query, (value_to_select,))
    #         table = cursor.fetchall()
    #         for row in table:
    #             placed = occurrence.get(p)
    #             #item_added.append(str(p) + " " + str(placed) + " " + str(row[0]))
    #             item_added.append(str(p))
    #             #value+=str(p)+"\t\t\t\t"+str(placed)+"\t\t\t\t"+str(row[0])+"\n\n"---
    #
    #             st.markdown(f"<tr><td>{str(p)}</td><td>{str(placed)}</td><td>{'{:.1f}'.format(row[0])}</td><td>{'{:.1f}'.format(row[0]*placed)}</td></tr>",unsafe_allow_html=True)
    #
    #             #result.add_row([str(p), placed, str(row[0])])
    #             total += (row[0] * occurrence.get(p))
    # print(item_added)

    table_style = """
    <style>
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th, td {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: black;
    }
    </style>
    """

    # Create the table header
    table_html = "<table><tr><th>Item Name</th><th>Count</th><th>Amount</th><th>Calculated</th></tr>"

    for p in U_Final:
                if item_added.count(p) == 0:
                    value_to_select = str(p)
                    query = "SELECT Amount FROM products WHERE Name=%s"
                    cursor.execute(query, (value_to_select,))
                    table = cursor.fetchall()
                    for row in table:
                        placed = occurrence.get(p)
                        item_added.append(str(p))
                        # Add rows to the table
                        table_html += f"<tr><td>{str(p)}</td><td>{str(placed)}</td><td>{'{:.1f}'.format(row[0])}</td><td>{'{:.1f}'.format(row[0] * placed)}</td></tr>"
                        total += (row[0] * occurrence.get(p))
    # Close the table
    table_html += "</table>"

    # Render the styled HTML table
    st.write(table_style, unsafe_allow_html=True)
    st.write(table_html, unsafe_allow_html=True)

    #------FREE PRODUCT----------------
    st.markdown(f"<h1 style='color:black;'>FREE\t</h1>",
                unsafe_allow_html=True)
    occurrence = {item: Free.count(item) for item in Free}
    item_added = []
    saved_amt = 0
    for p in Free:
        if item_added.count(p) == 0:

            value_to_select = str(p)
            query = "select Amount from products where Name=%s"
            cursor.execute(query, (value_to_select,))
            table = cursor.fetchall()
            for row in table:
                placed = occurrence.get(p)
                # item_added.append(str(p) + " " + str(placed) + " " + str(row[0]))
                item_added.append(str(p))
                # value+=str(p)+"\t\t\t\t"+str(placed)+"\t\t\t\t"+str(row[0])+"\n\n"---

                st.markdown(
                    f"<tr><td>{str(p)}</td><td>{str(placed)}</td><td>{'{:.1f}'.format(row[0])}</td><td>{'{:.1f}'.format(row[0] * placed)}</td></tr>",
                    unsafe_allow_html=True)

                #result.add_row([str(p), placed, str(row[0])])
                saved_amt += (row[0] * occurrence.get(p))
    print(item_added)

    st.markdown(f"<h1 style='color:red;'>TOTAL\t{'{:.1f}'.format(total)}</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:blue;'>SAVED AMOUNT\t{'{:.1f}'.format(saved_amt)}</h1>",
                unsafe_allow_html=True)

    st.markdown("</table>",unsafe_allow_html=True)



    # QR CODE ######################################################################

    upi_id = "9941815173@kotak"
    amt_to_pay = total
    upi_url = f"upi://pay?pa={upi_id}&am={amt_to_pay}"
    # Generate QR code
    qr = qrcode.QRCode(version=1,error_correction=qrcode.constants.ERROR_CORRECT_Q,box_size=10,border=4,)
    qr.add_data(upi_url)
    qr.make(fit=True)

    # Create QR code image --------------------------------------------------------
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Convert PIL.Image to bytes
    img_bytes = io.BytesIO()
    qr_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    st.info("Scan this QR code with your UPI app to make the payment.")
    st.image(img_bytes, caption="UPI Payment QR Code", use_column_width=True)

    ###################################################################################

    # ----------- PHNO ADD -----------------------
    phno = "91"+phno
    query = "INSERT INTO `tally`(`Time`, `mobile`, `Total`) VALUES(%s,%s,%s)"
    today_time = datetime.now()
    cursor.execute(query, (today_time, phno, total))
    connection.commit()

    # SMS exchange-------------------------------------------------

    to_phone_number = '+919941815173'  # Replace with recipient's phone number
    message = '!!! You Final Price Amount is '+str(total)+ ' and Saved Amount is '+ str(saved_amt) +' Thank You For Shopping with us!!!'  # Your message
    send_sms(to_phone_number, message)

    # result.add_row(['TOTAL',"",total])
    #value += str(result)
    value += "\n\nTOTAL\t" + str(total)
    #st.markdown(f"TOTAL {total}", unsafe_allow_html=True)
    #st.markdown(f"<h1 style='color:red;'>TOTAL\t{'{:.1f}'.format(total)}</h1>",unsafe_allow_html=True)
    #connection.commit()
    # print(result)

    cursor.close()

    # closing connection object
    connection.close()

    Products_added.clear()

    value = value.replace("(", "")
    value = value.replace(")", "")
    value = value.replace(",", "")
    value = value.replace("'", "")
    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.replace("+", "")
    value = value.replace("-", "")

    #st.markdown(value)
    print(value)