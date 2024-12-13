import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import requests 
import imutils 

url = "http://172.18.84.72:8080/shot.jpg"

# HSV ranges for color detection and corresponding BGR color values
myColors = [['Purple', 110, 46, 50, 140, 255, 255]]
colorValue = [[255, 0, 0]]


def get_ip_img():
        img_resp = requests.get(url) 
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
        img = cv2.imdecode(img_arr, -1) 
        img = imutils.resize(img, width=1000, height=1800) 
        return True, img 

def drawSmoothCurve(segment, imgResult):
    if len(segment) > 3:  # Spline requires at least three points
        pts = np.array(segment)
        # Ensure the points are not collinear or too close
        if np.std(pts[:, 0]) > 1 and np.std(pts[:, 1]) > 1:
            # print(pts.shape)
            tck, u = splprep(pts.T, s=1)  # s is the smoothness parameter
            x_new, y_new = splev(np.linspace(0, 1, max(50, len(segment) * 10)), tck)
            for i in range(len(x_new) - 1):
                cv2.line(
                    imgResult, 
                    (int(x_new[i]), int(y_new[i])), 
                    (int(x_new[i + 1]), int(y_new[i + 1])), 
                    colorValue[0],  # Use the color of the first point
                    thickness=2
                )

def contours(img):
    x, w, y, h = 0, 0, 0, 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
    return x + w // 2, y

def findColor(img, imgResult):
    count = 0
    new_points = []
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[1:4])
        upper = np.array(color[4:])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = contours(mask)
        cv2.circle(imgResult, (x, y), 10, colorValue[count], cv2.FILLED)
        if x != 0 and y != 0:
            new_points.append([x, y])
        count += 1  # To decide the color
    print(new_points)
    return new_points


def main():
    all_segments = []  # List of all segments
    current_segment = []  # Current segment being drawn

    cam = cv2.VideoCapture(0)
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 480))
    cam.set(3, 640)
    cam.set(4, 480)
    cam.set(10, 130)  # Brightness

    while True:
        # success, img = cam.read()
        success, img = get_ip_img()
        if success:
            imgResult = img.copy()

            # Detect new points
            new_points = findColor(img, imgResult)
            if len(new_points) != 0:
                current_segment.extend(new_points)

            # Draw all segments
            for segment in all_segments:
                try:
                    drawSmoothCurve(segment, imgResult)
                except Exception as e:
                    current_segment = []
                    print(e)

            if current_segment:
                try:
                    
                    drawSmoothCurve(current_segment, imgResult)
                except Exception as e:
                    current_segment = []

                    print(e)
            imgResult = cv2.flip(imgResult, 1)
            cv2.imshow("Result", imgResult)
            out.write(imgResult)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Start a new segment
                if current_segment:
                    all_segments.append(current_segment)
                    current_segment = []
            elif key == ord('q'):  # Quit the program
                if current_segment:
                    all_segments.append(current_segment)
                break
            elif key == ord('c'):  # clear lists
                current_segment.clear()
                all_segments.clear()

    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
