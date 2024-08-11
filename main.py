import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import svgwrite
import cairosvg
import numpy as np
import cv2
from PIL import Image, ImageFilter

# Define colors
colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
           'orange', 'purple', 'brown', 'pink', 'gray', 'teal', 'navy',
           '#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2',
           '#FF4500', '#2E8B57', '#D2691E', '#A52A2A', '#7FFF00']

# Normalize CSV data and scale it
def normalize_and_scale_csv(csv_path, output_path, scale_factor=1000):
    df = pd.read_csv(csv_path, header=None)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = pd.DataFrame(scaler.fit_transform(df))
    df_scaled = df_normalized * scale_factor
    df_scaled.to_csv(output_path, index=False, header=False)
    print(df_scaled.head())

# Read CSV data into paths
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Regularize and beautify shapes in the image
def regularize_shapes(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Detect and regularize shapes
        if len(approx) == 3:
            shape = "Triangle"
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        elif len(approx) == 4:
            # Detect rectangle or square
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
        elif len(approx) > 4:
            # Detect circle
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * (area / (cv2.arcLength(contour, True) ** 2))

            # If the circularity is close to 1, it's a circle
            if 0.7 <= circularity <= 1.3:
                shape = "Circle"
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            else:
                shape = "Ellipse"
                cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
        else:
            # For irregular shapes, just draw the contour
            shape = "Polygon"
            cv2.drawContours(img, [approx], -1, (0, 255, 255), 2)

    # Apply smoothing or any additional beautification
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Save the output image
    cv2.imwrite(output_path, img)

# Convert polylines to SVG and then to PNG
def polylines2svg(paths_XYs, svg_path, color='blue'):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for path in paths_XYs:
        path_data = []
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))

        path_str = ''
        for cmd, coord in path_data:
            x, y = coord
            path_str += f" {cmd} {x},{y}"

        group.add(dwg.path(d=path_str.strip(), fill='none', stroke=color, stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                     parent_width=W, parent_height=H,
                     output_width=fact * W, output_height=fact * H,
                     background_color='white')

    return png_path

# File paths
file_path = "/content/frag0.csv"
normalized_csv_path = "normalized_file.csv"
svg_path = "new.svg"
regularized_image_path = "regularized_image.png"

# Normalize, scale CSV, and generate SVG and PNG
normalize_and_scale_csv(file_path, normalized_csv_path, scale_factor=1000)
path_XYs = read_csv(normalized_csv_path)
png_path = polylines2svg(path_XYs, svg_path, color=colours[0])

# Regularize and beautify shapes in the generated image
regularize_shapes(png_path, regularized_image_path)

print(f"Regularized image saved at: {regularized_image_path}")
