import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from logic import Application

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

def draw_smooth_curve(points, img_result, segment_length=10):
    if len(points) > 3:  # Spline requires at least three points
        # Only use the most recent `segment_length` points
        recent_points = points[-segment_length:]
        pts = np.unique(recent_points, axis=0)  # Remove duplicate points
        if len(pts) < 3:
            print("Not enough unique points for a smooth curve.")
            return

        if np.std(pts[:, 0]) > 1 and np.std(pts[:, 1]) > 1:  # Check for variance
            try:
                tck, u = splprep(pts.T, s=1)  # s is the smoothness parameter
                x_new, y_new = splev(np.linspace(0, 1, max(50, len(pts) * 10)), tck)

                # Draw the curve segment by connecting points
                for i in range(len(x_new) - 1):
                    pt1 = (int(x_new[i]), int(y_new[i]))
                    pt2 = (int(x_new[i + 1]), int(y_new[i + 1]))
                    cv2.line(img_result, pt1, pt2, (0, 0, 255), thickness=2)  # Red color
            except ValueError as e:
                print(f"Error in drawing smooth curve: {e}")
        else:
            print("Points lack sufficient variance to draw a curve.")

if __name__ == "__main__":
    app = Application()
    app.run()