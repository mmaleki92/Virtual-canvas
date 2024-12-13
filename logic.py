import cv2
import numpy as np
import requests
import imutils
from scipy.interpolate import splprep, splev
from abc import ABC, abstractmethod
from typing import List
from copy import copy
from pynput import keyboard
from keyboard_manager import KeyboardManager

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Segment:
    def __init__(self):
        self.points = []

    def add_point(self, point: Point):
        self.points.append(point)

    def get_points(self):
        return self.points

def get_ip_img():
    url = "http://172.18.84.72:8080/shot.jpg"  # Replace with your URL
    try:
        img_resp = requests.get(url, timeout=100)
        img_resp.raise_for_status()
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
        # img = cv2.flip(img, 1)  # Correct horizontal flip
        return True, img
    except Exception as e:
        print(f"Error fetching image: {e}")
        return False, None

def create_color_mask(hsv_img, color, tolerance):
    lower_bound = np.array([
        max(color[0] - tolerance, 0),
        max(color[1] - tolerance, 0),
        max(color[2] - tolerance, 0),
    ])
    upper_bound = np.array([
        min(color[0] + tolerance, 179),
        min(color[1] + tolerance, 255),
        min(color[2] + tolerance, 255),
    ])
    return cv2.inRange(hsv_img, lower_bound, upper_bound)

def find_contours(img, img_result, tolerance, sampled_color, last_point=None, max_distance=50):
    draw_it = False
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = create_color_mask(hsv_img, sampled_color, tolerance)

    # Find contours of the masked regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the center of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Check if the point is too far from the previous point
            if last_point is not None:
                distance = np.sqrt((cx - last_point[0]) ** 2 + (cy - last_point[1]) ** 2)
                if distance > max_distance:
                    return draw_it, None, None

            # Draw the contour and the center point
            cv2.drawContours(img_result, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(img_result, (cx, cy), 10, (255, 0, 0), -1)
            cv2.putText(img_result, "Tracking Center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            draw_it = True
            return draw_it, cx, cy

    return draw_it, None, None

class Curve:
    def __init__(self):
        self.points = []

    def add_point(self, point: Point):
        self.points.append(point)
    def smooth(self):
        """Smooth the curve using spline interpolation."""
        if len(self.points) < 4:
            print("Not enough points to smooth.")
            return self.points  # Not enough points to smooth

        # Extract points and ensure proper shape for splprep
        pts = np.array([[p.x, p.y] for p in self.points]).T

        if np.all(pts[0] == pts[0][0]) or np.all(pts[1] == pts[1][0]):
            print("Singular point data; cannot fit a curve.")
            return self.points

        try:
            # Fit spline to the points
            tck, u = splprep(pts, s=5)
            x_new, y_new = splev(np.linspace(0, 1, max(50, len(pts[0]) * 10)), tck)
            return [Point(int(x), int(y)) for x, y in zip(x_new, y_new)]
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return self.points


class Layer:
    def __init__(self, name: str):
        self.name = name
        self.curves: List[Curve] = []
        self.segments = []

    def add_curve(self, curve: Curve):
        self.curves.append(curve)

    def add_segment(self, segment: Segment):
        self.segments.append(segment)


    def draw(self, img):
        """Draw all elements of the layer on the provided image."""
        for segment in self.segments:
            for i in range(len(segment.points) - 1):
                p1 = segment.points[i]
                p2 = segment.points[i + 1]
                # cv2.line(img, (p1.x, p1.y), (p2.x, p2.y), (0, 255, 0), 2)
        
        for curve in self.curves:
            try:
                smoothed_points = curve.smooth()
                for i in range(len(smoothed_points) - 1):
                    p1 = smoothed_points[i]
                    p2 = smoothed_points[i + 1]
                    cv2.circle(img, (p1.x, p1.y), 10, (0, 0, 255), cv2.FILLED)

                    # cv2.line(img, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 0), 2)
            except Exception as e:
                print(e)

class Canvas:
    def __init__(self):
        self.layers: List[Layer] = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def draw(self, img):
        for layer in self.layers:
            # try:
            layer.draw(img)
            # except:
            #     pass

    def get_active_layer(self):
        """Returns the top-most visible layer."""
        return self.layers[-1] if self.layers else None


class Tool(ABC):
    @abstractmethod
    def apply(self, canvas: Canvas, event, x, y):
        pass

class BrushTool(Tool):
    def __init__(self):
        self.active_segment = Segment()

    def apply(self, canvas: Canvas, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
            point = Point(x, y)
            self.active_segment.add_point(point)
            layer = canvas.get_active_layer()
            if layer:
                layer.add_segment(self.active_segment)

def draw_circles(points, img_result):
    for i in points:
        cv2.circle(img_result, i, 10, (0, 0, 255), cv2.FILLED)

def drawlines(original_img, lines):
        try:
            isClosed = True
            color = (0, 255, 0)
            thickness = 2

            points = []
            for line in lines:
                points.append([line[0][0], line[0][1]])
                points.append([line[0][2], line[0][3]])
                points = np.array(points, np.int32).reshape((-1, 1, 2))

                img = cv2.polylines(original_img, [points], isClosed, color, thickness)
            return img
        except:
            return original_img

class Application:
    def __init__(self):
        self.canvas = Canvas()
        self.current_tool = BrushTool()
        self.sampled_color = None
        self.sampling_mode = True
        self.sampled_points = []
        self.all_points = {"current": [],
                           "previous": [],
                           "segments": []
                           }
        self.ip_camera = False
        self.draw = False
        # Initialize the image
        if not self.ip_camera:
            self.cam = cv2.VideoCapture(0)
            # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 480))
            self.cam.set(3, 640)
            self.cam.set(4, 480)
            self.cam.set(10, 130)  # Brightness
            self.update_image()
        else:
            self.success, self.img = get_ip_img()

        # Initialize the keyboard manager
        self.kb_manager = KeyboardManager()
        self.kb_manager.start()
        
    def update_image(self):
        if self.ip_camera:
            self.success, self.img = get_ip_img()
        else:
            self.success, self.img = self.cam.read()

    def run(self):
        cv2.namedWindow("Canvas")
        cv2.setMouseCallback("Canvas", self.click_to_sample)
        tolerance = 20
        last_point = None  # To track the previous valid point

        while True:
            self.update_image()
            if not self.success:
                return
            img_copy = self.img.copy()

            if self.sampling_mode:
                cv2.putText(img_copy, "Click to sample a color", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                draw_it, x, y = find_contours(self.img, img_copy, tolerance, self.sampled_color, last_point)
                if draw_it and x is not None and y is not None:
                    self.all_points["current"].append([x, y])
                    last_point = (x, y)  # Update the last valid point

            key = cv2.waitKey(1) & 0xFF

            if self.kb_manager.is_key_pressed(keyboard.Key.esc):
                print("Exiting...")
                break
            elif self.kb_manager.is_key_pressed('d'):
                self.draw = True
            elif self.kb_manager.is_key_pressed('n'):
                self.canvas.add_layer(Layer(f"Layer {len(self.canvas.layers) + 1}"))

            if self.draw:
                self.draw_segments_and_curves()
            else:
                self.all_points["segments"].append(self.all_points["current"])
                self.all_points["previous"] = copy(self.all_points["current"])
                self.all_points["current"] = []
            self.canvas.draw(img_copy)

            cv2.imshow("Canvas", img_copy)

        self.cam.release()
        cv2.destroyAllWindows()


    def draw_segments_and_curves(self):
        # Add a new layer to draw lines if none exists
        if not self.canvas.layers:
            self.canvas.add_layer(Layer("Default Layer"))

        active_layer = self.canvas.get_active_layer()

        if active_layer:
            # Add sampled points as a segment
            if self.all_points["current"]:
                segment = Segment()
                for x, y in self.all_points["current"]:
                    segment.add_point(Point(x, y))
                active_layer.add_segment(segment)

            # Convert the segment into a curve and smooth it
            curve = Curve()
            for x, y in self.all_points["current"]:
                curve.add_point(Point(x, y))
            active_layer.add_curve(curve)



    def handle_mouse(self, event, x, y, flags, param):
        if self.current_tool:
            self.current_tool.apply(self.canvas, event, x, y)

    def click_to_sample(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.sampling_mode:
                # success, img = get_ip_img()
                self.update_image()
                if self.success:
                    self.sampled_color = self.detect_color(self.img, x, y)
                    self.sampling_mode = False
                    print(f"Sampled color HSV: {self.sampled_color}")
            # else:
            #     self.all_points[].append((x, y))
            #     print(f"Point added at ({x}, {y}). Current points: {self.sampled_points}")

    def detect_color(self, img, x, y):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv_img[y, x]  # Return HSV value of the clicked pixel

if __name__ == "__main__":
    app = Application()
    app.run()
