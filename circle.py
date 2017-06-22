# coding=<utf-8>
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from numpy import *
import copy
import time
import sys
import math
import operator

pic_path = 'dataset/true/1.png'
rect_scale = 5
rect_area = 0
rect_min_area = 0.0010
color_range = 20
hvs_luminance = 190
angle_limit = 8
top_ext_dist = 4
cluster_dist = 20
#---------------------------------------------------------
pic_width = 0
pic_height = 0
#---------------------------------------------------------
def CalculateOneLineAndOnePointMinDistance(a,b,c):
    u = np.array([b[0] - a[0], (pic_height-b[1]) - (pic_height-a[1])])
    v = np.array([c[0] - a[0], (pic_height-c[1]) - (pic_height-a[1])])
    if (linalg.norm(u) > 0):
        L = abs(cross(u, v) / linalg.norm(u))
    else:
        L = int()
    return L

def CalculateTwoPointDistance(src, dst):
    a = np.array(src)
    b = np.array(dst)
    return np.linalg.norm(b-a)

def PointConvertDegree(center, point1):
    angle = math.degrees(math.atan2((pic_height-point1[1]) - (pic_height-center[1]), point1[0] - center[0]))
    if (angle < 0) :
        angle = 360 + angle
    return angle

def DegreeCompare(angleRef, angleDst):
    result = angleDst - angleRef
    if result > 180:
        result = 180 - result
    if result < -180:
        result = result + 360
    return result

def DegreeMirror(angle):
    if angle > 180:
        angle += 180
        if angle >= 360:
            angle -= 360
    return angle

def GetRectColor(img, rect):
    m1 = np.zeros(img.shape, np.uint8)
    cv2.drawContours(m1, rect, 0, (255, 255, 255), -1)
    m1 = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
    m2 = cv2.bitwise_and(img, img, mask=m1)
    hist0 = cv2.calcHist([m2], [0], None, [256], [0.0, 255.0])
    hist1 = cv2.calcHist([m2], [1], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([m2], [2], None, [256], [0.0, 255.0])
    hist0[0:10] = 0
    hist1[0:10] = 0
    hist2[0:10] = 0
    maxidx0, maxval0 = max(enumerate(hist0), key=operator.itemgetter(1))
    maxidx1, maxval1 = max(enumerate(hist1), key=operator.itemgetter(1))
    maxidx2, maxval2 = max(enumerate(hist2), key=operator.itemgetter(1))
    #return (maxidx0, maxidx1, maxidx2)
    return (maxval0, maxval1, maxval2)

'''
def GetRectColorHsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist0 = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist1 = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist2 = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    hist1[0:10] = 0
    hist2[0:10] = 0
    hist2[0:10] = 0
    maxidx0, maxval0 = max(enumerate(hist0), key=operator.itemgetter(1))
    maxidx1, maxval1 = max(enumerate(hist1), key=operator.itemgetter(1))
    maxidx2, maxval2 = max(enumerate(hist2), key=operator.itemgetter(1))
    return (maxidx0, maxidx1, maxidx2)
'''

def drawPoint(img, point, size=1, color=(0, 0, 255)):
    cv2.circle(img, point, size, color, -1)

def FindCluster(cluster, idx1, idx2, rectWH):
    ret_cluster = []
    for i in range(0, cluster.__len__()):
        pos = cluster[i]
        if pos != cluster[idx1] and pos != cluster[idx2]:
            dist = CalculateOneLineAndOnePointMinDistance(cluster[idx1], cluster[idx2], pos)
            limitDist = (rectWH[i][0]/(pic_width/4))
            if limitDist < cluster_dist:
                limitDist = cluster_dist
            angle = abs(DegreeCompare(rectWH[i][2], rectWH[idx1][2]))
            if dist < limitDist and angle < angle_limit:
                ret_cluster.append(i)
    return ret_cluster

def CheckCluster(rectCenter, rectWH):
    maxNum = 0
    max_pos = []
    dst_pos = []
    dst_rect = []
    dst_idx = []
    for pos1 in range(0, rectCenter.__len__()):
        for pos2 in range(0, rectCenter.__len__()):
            if pos1 != pos2:
                angle3 = abs(DegreeCompare(rectWH[pos1][2], rectWH[pos2][2]))
                if angle3 < angle_limit:
                    tmp = FindCluster(rectCenter, pos1, pos2, rectWH)
                    if tmp.__len__() > maxNum:
                        maxNum = tmp.__len__()
                        max_pos = [pos1, pos2, angle3]
                        dst_rect = tmp

    dst_pos.append(rectCenter[max_pos[0]])
    dst_idx.append(max_pos[0])
    dst_pos.append(rectCenter[max_pos[1]])
    dst_idx.append(max_pos[1])
    for pos in dst_rect:
        dst_pos.append(rectCenter[pos])
        dst_idx.append(pos)

    #drawPoint(image, dst_pos[0], 5, (0, 255, 0))
    #cv2.drawContours(image, [rectPos[dst_idx[0]]], 0, (0, 0, 255), 1)
    #drawPoint(image, dst_pos[1], 5, (0, 0, 255))
    #cv2.drawContours(image, [rectPos[dst_idx[1]]], 0, (0, 0, 255), 1)

    '''
    for pos in dst_pos:
        drawPoint(img, pos, 5, (255, 0, 0))
    drawPoint(img, dst_pos[0], 5, (0, 255, 0))
    drawPoint(img, dst_pos[1], 5, (0, 0, 255))
    for pos in dst_idx:
        print rectWH[pos][2]
        cv2.drawContours(img, [rectPos[pos]], 0, (0, 0, 255), 1)
    '''
    return dst_idx

def findFourSide(contour):
    param = 0.001
    approx = []
    while param < 1:
        epsilon = param * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        param += 0.001
        if approx.__len__() == 4:
            break
    return approx

def findBottomSide(rect, angle):
    boxRect = []
    for pos in rect:
        boxRect.append(pos[0])
    #
    bottomPos = boxRect[0]
    bottomIdx = 0
    idxTmp = 0
    for pos in boxRect:
        if pos[1] > bottomPos[1]:
            bottomIdx = idxTmp
        idxTmp += 1
    #
    bottomIdxNext = bottomIdx + 1
    if bottomIdxNext >= 4:
        bottomIdxNext = 0
    #
    bottomIdxPrev = bottomIdx - 1
    if bottomIdxPrev < 0:
        bottomIdxPrev = 3
    #
    angle1 = PointConvertDegree(boxRect[bottomIdx], boxRect[bottomIdxNext])
    angleCmp = abs(DegreeCompare(angle, DegreeMirror(angle1)))
    if angleCmp < 60:
        return [boxRect[bottomIdx], boxRect[bottomIdxPrev]]
    else:
        return [boxRect[bottomIdx], boxRect[bottomIdxNext]]

def findTopSide(rect, angle):
    boxRect = []
    for pos in rect:
        boxRect.append(pos[0])
    #
    TopPos = boxRect[0]
    TopIdx = 0
    idxTmp = 0
    for pos in boxRect:
        if pos[1] < TopPos[1]:
            TopIdx = idxTmp
        idxTmp += 1
    #
    TopIdxNext = TopIdx + 1
    if TopIdxNext >= 4:
        TopIdxNext = 0
    #
    TopIdxPrev = TopIdx - 1
    if TopIdxPrev < 0:
        TopIdxPrev = 3
    #
    angle1 = DegreeMirror(PointConvertDegree(boxRect[TopIdx], boxRect[TopIdxNext]))
    angleCmp = abs(DegreeCompare(angle, angle1))
    if angleCmp < 60:
        return [boxRect[TopIdx], boxRect[TopIdxPrev]]
    else:
        return [boxRect[TopIdx], boxRect[TopIdxNext]]

def rotatePoint(origin, point, angle):
    angle = math.radians(360 - angle)
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
'''
def checkTheSamePoint(src1, src2):
    if src1[0] == src2[0]:
        if src1[1] == src2[1]:
            return 1
    return 0
def findNextPoint(pos1, pos2):
    idx = pos1 + 1
    if idx > 3:
        idx = 0
    if idx == pos2:
        idx = pos1 - 1
        if idx < 0:
            idx = 3
    return idx

def fixTopSide(img, rect, bottom):
    boxRect = []
    for pos in rect:
        boxRect.append(pos[0])
    #
    for pos in range(0, 4):
        if checkTheSamePoint(boxRect[pos], bottom[0]) > 0:
            idx1 = pos
    for pos in range(0, 4):
        if checkTheSamePoint(boxRect[pos], bottom[1]) > 0:
            idx2 = pos
    #
    idx1_1 = findNextPoint(idx1, idx2)
    idx2_1 = findNextPoint(idx2, idx1)
    angle = DegreeMirror(PointConvertDegree(boxRect[idx1], boxRect[idx2]))
    l1 = CalculateTwoPointDistance(boxRect[idx1], boxRect[idx1_1])
    l2 = CalculateTwoPointDistance(boxRect[idx2], boxRect[idx2_1])
    print l1, l2
    if l1 > l2:
        max_idx = idx1_1
    else:
        max_idx = idx2_1

    print l1, l2
    PointA = boxRect[max_idx]
    origin = boxRect[max_idx]

    NewPointA = np.int0(rotatePoint(origin, PointA, angle))

    print NewPointA

    drawPoint(img, tuple(boxRect[idx2]), 5, (0, 255, 255))
    drawPoint(img, tuple(boxRect[idx2_1]), 5, (0, 255, 255))
'''


def findSide(contour, angle):
    approx = findFourSide(contour)
    if approx.__len__() != 4:
        return None
    sideAngle = angle

    bottom = np.array(findBottomSide(approx, sideAngle))
    top = np.array(findTopSide(approx, sideAngle))

    #angle1 = DegreeMirror(PointConvertDegree(top[0], top[1]))
    #angle2 = DegreeMirror(PointConvertDegree(bottom[0], bottom[1]))

    #print "diff:", abs(DegreeCompare(angle1, angle2))

    #if abs(DegreeCompare(angle1, angle2)) > 5:
    #    fixTopSide(img, approx, bottom)

    #cv2.drawContours(img, [approx], 0, (0, 0, 255), 1)
    #cv2.drawContours(img, [bottom], 0, (255, 0, 255), 2)
    #cv2.drawContours(img, [top], 0, (255, 0, 0), 2)

    #drawPoint(img, tuple(top[0]), 5, (0, 255, 0))

    return [top, bottom]

def fixPoint(pos):
    x = pos[0]
    y = pos[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    #if x > pic_width:
    #    x = pic_width - 1
    #if y > pic_height:
    #    y = pic_height - 1
    return [x, y]

def getTopSideRect(pos):
    if pos[0][0] > pos[1][0]:
        pos1 = pos[1]
        pos2 = pos[0]
    else:
        pos1 = pos[0]
        pos2 = pos[1]
    angle = PointConvertDegree(pos1, pos2)
    dist = CalculateTwoPointDistance(pos1, pos2)

    if top_ext_dist > 0:
        addDist = dist / top_ext_dist
    else:
        addDist = 0
    '''
    posT1 = fixPoint(extendPoint(pos1[0], pos1[1], addDist, angle))
    posT2 = fixPoint(extendPoint(pos1[0], pos1[1], addDist, angle - 180))
    a1 = CalculateTwoPointDistance(posT1, pos2)
    a2 = CalculateTwoPointDistance(posT2, pos2)
    if a1 > a2:
        pos1 = posT1
    else:
        pos1 = posT2

    posT1 = fixPoint(extendPoint(pos2[0], pos2[1], addDist, angle))
    posT2 = fixPoint(extendPoint(pos2[0], pos2[1], addDist, angle + 180))
    a1 = CalculateTwoPointDistance(posT1, pos1)
    a2 = CalculateTwoPointDistance(posT2, pos1)
    if a1 > a2:
        pos2 = posT1
    else:
        pos2 = posT2
    '''
    pos1 = fixPoint(extendPoint(pos1, addDist, angle - 180))
    pos2 = fixPoint(extendPoint(pos2, addDist, angle))

    #pos2 = fixPoint(extendPoint(pos2[0], pos2[1], dist / top_ext_dist, angle + 90))
    #
    NewP1 = extendPoint(pos1, dist / 2, angle)
    NewPointA = np.int0(rotatePoint(pos1, NewP1, angle+90))
    NewPointA = fixPoint(NewPointA)
    #
    NewP2 = extendPoint(pos2, dist / 2, angle)
    NewPointB = np.int0(rotatePoint(pos2, NewP2, angle+90))
    NewPointB = fixPoint(NewPointB)
    #
    dst_rect = []
    dst_rect.append(pos1)
    dst_rect.append(NewPointA)
    dst_rect.append(NewPointB)
    dst_rect.append(pos2)
    dst_rect = np.array(dst_rect)

    return dst_rect


def getBopttomSideRect(pos):
    if pos[0][0] > pos[1][0]:
        pos1 = pos[1]
        pos2 = pos[0]
    else:
        pos1 = pos[0]
        pos2 = pos[1]
    angle = PointConvertDegree(pos1, pos2)
    dist = CalculateTwoPointDistance(pos1, pos2)
    #
    NewP1 = extendPoint(pos1, dist / 2, angle)
    NewPointA = np.int0(rotatePoint(pos1, NewP1, angle - 90))
    NewPointA = fixPoint(NewPointA)
    #
    NewP2 = extendPoint(pos2, dist / 2, angle)
    NewPointB = np.int0(rotatePoint(pos2, NewP2, angle - 90))
    NewPointB = fixPoint(NewPointB)
    #
    dst_rect = []
    dst_rect.append(pos1)
    dst_rect.append(NewPointA)
    dst_rect.append(NewPointB)
    dst_rect.append(pos2)
    dst_rect = np.array(dst_rect)

    return dst_rect

def extendPoint(pos, d, theta):
    theta_rad = pi/2 - radians(theta + 90)
    return np.int0([pos[0] + d*cos(theta_rad), pos[1] + d*sin(theta_rad)])
#---------------------------------------------------------
def FindZebraCrossing(filePath):
    srcImg = image = cv2.imread(filePath)  #original
    pic_width = image.shape[1]
    pic_height = image.shape[0]

    rect_area = np.int((pic_width * pic_height * 1.0) * rect_min_area)

    # Color Filter
    hsv = cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV) #hsv
    low_color = np.array([0, 0, hvs_luminance])
    # low_color = np.array([0, 0, 180])
    upper_color = np.array([180, 43, 255])
    mask = cv2.inRange(hsv, low_color, upper_color)
    res = cv2.bitwise_and(srcImg, srcImg, mask=mask) #filter image
    
    # Fix Image Color
    image = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)

    # canny
    img_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(img_gray, 150, 220, apertureSize=3) #canny

    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # for CV2
    print("Contours:{} ", len(contours))
    print("\n\n\n***************************************************************************\n")

    area_pos = []
    rect_pos = []
    rect_center = []
    rect_wh = []
    for i in range(0, len(contours)):
        hull = cv2.convexHull(contours[i])
        if len(hull) < 5:
            continue

    return None

#---------------------------------------------------------
def main():
    fig = plt.figure()
    srcImg = image = cv2.imread(pic_path)

    pic_width = image.shape[1]
    pic_height = image.shape[0]

    rect_area = np.int((pic_width * pic_height * 1.0) * rect_min_area)


    # Color Filter
    hsv = cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV)
    low_color = np.array([0, 0, hvs_luminance])
    #low_color = np.array([0, 0, 180])
    upper_color = np.array([180, 43, 255])
    mask = cv2.inRange(hsv, low_color, upper_color)
    res = cv2.bitwise_and(srcImg, srcImg, mask=mask)

    # Fix Image Color
    image = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)

    #canny
    img_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(img_gray, 150, 220, apertureSize=3)

    _,contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # for CV2
    #印出輪廓數量
    print("Contours: ",len(contours))
    print("\n\n\n***************************************************************************\n")

    area_pos = []
    rect_pos = []
    rect_center = []
    rect_wh = []

    for i in range(0, len(contours)):
        hull = cv2.convexHull(contours[i])
        if len(hull) < 5:
            continue

        # 計算重心
        moments = cv2.moments(hull)
        m00 = moments['m00']
        centroid_x, centroid_y = None, None
        if m00 != 0:
            centroid_x = int(moments['m10'] / m00)  # Take X coordinate
            centroid_y = int(moments['m01'] / m00)  # Take Y coordinate
        circle_pos = (centroid_x, centroid_y)

        if len(circle_pos) != 2:
            continue
        if (circle_pos[0] == None) or (circle_pos[1] == None):
            continue
        #print circle_pos

        #x, y, w, h = cv2.boundingRect(hull)
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

        a1 = CalculateTwoPointDistance(box[0], box[1])
        a2 = CalculateTwoPointDistance(box[1], box[2])
        box_w = max(a1, a2)
        box_h = min(a1, a2)
        if box_h <= 0:
            continue
        box_scale = (box_w / box_h)
        box_area = (box_w * box_h)
        if box_w == a1:
            box_angle = PointConvertDegree(box[0], box[1])
        else:
            box_angle = PointConvertDegree(box[1], box[2])

        if box_scale > rect_scale and box_area > rect_area:
            box_color = GetRectColor(image, [box])
            if box_color[0] > color_range and box_color[1] > color_range and box_color[2] > color_range:
                # cv2.drawContours(image, [box], 0, (0, 0, 255), 1)
                # drawPoint(image, circle_pos, 5, (255, 0, 0))
                rect_pos.append(hull)
                rect_center.append(circle_pos)
                rect_wh.append([box_w, box_h, box_angle])

    if not rect_pos:
        exit()
    idx_pos = CheckCluster(rect_center, rect_wh)


    for idx in idx_pos:
        for pos in rect_pos[idx]:
            area_pos.append(pos)

    area_pos = np.array(area_pos)

    hull = cv2.convexHull(area_pos)
    rect = cv2.minAreaRect(area_pos)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(hull)
    print(x,y,w,h)
    im = image[y:y+h,x:x+w]
    cv2.imwrite('test.png',im)

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.drawContours(image, [hull], -1, (255, 0, 255), 1)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

    #print hull
    #print rect_wh[idx_pos[0]][2], rect_wh[idx_pos[0]][0]
    #print rect_wh[idx_pos[1]][2], rect_wh[idx_pos[1]][0]

    # line_dir = PointConvertDegree(rect_center[idx_pos[0]], rect_center[idx_pos[1]])
    # line_dir = DegreeMirror(line_dir)
    #
    # dst = findSide(hull, line_dir)
    # topRect = getTopSideRect(dst[0])
    # bottomRect = getBopttomSideRect(dst[1])
    #
    #
    # cv2.drawContours(image, [topRect], 0, (255, 0, 0), 2)

    #cv2.drawContours(image, [bottomRect], 0, (255, 0, 0), 2)
    # print ("Top", topRect)
    # print ("Bottom", bottomRect)

    #---------------------------------------------------------
    # Escape Keyboard Event
    def press(event):
        if event.key == u'escape':
            plt.close()
            cv2.destroyAllWindows()
    fig.canvas.mpl_connect('key_press_event', press)



    #顯示原圖 & output
    # plt.subplot(1, 2, 1), plt.imshow(image)
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    #
    # #顯示canny圖
    # plt.subplot(1, 2, 2), plt.imshow(canny_img, cmap = 'gray')
    # plt.title('Canny'), plt.xticks([]), plt.yticks([])
    #
    #
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    # plt.show()
    #
    # print("\n***************************************************************************")
    # print(" End")
    # print("***************************************************************************")

if __name__ == '__main__':
    main()