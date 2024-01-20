from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO("/home/dslab/Documents/Face detection/runs/detect/train2/weights/best.pt")

# Set the confidence and IoU thresholds
model.conf = 0.80
model.iou = 0.10

# Get the list of all the image files in the directory
image_files = os.listdir('/home/dslab/Documents/Face detection/test/images')

for file in image_files:
    print(file)

num_image_files = len(image_files)

# Print the result
print("Number of image files:", num_image_files)

# Define the output directory
output_directory = "detected images/output/"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over the image files
for image_file in image_files:

    # Read the image file
    image = cv2.imread(os.path.join('/home/dslab/Documents/Face detection/test/images', image_file))

    # Predict objects in the image using the YOLOv8 model
    results = model([image])

    # Iterate over the predicted bounding boxes and their classes
    for result in results:
        for i, data in enumerate(result.boxes.data.tolist()):
            classs = result.boxes.cls.tolist()[i]

            # Get the confidence score for the current prediction
            confidence = data[4]

            # If the confidence score is greater than the threshold, draw the bounding box and label the object
            if confidence > 0.5:
                xmin, ymin, xmax, ymax = (
                    int(data[0]),
                    int(data[1]),
                    int(data[2]),
                    int(data[3]),
                )
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                image = cv2.putText(
                    image,
                    f"{result.names[classs]}",
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    # Resize the image and display it in a window
    image = cv2.resize(image, (900, 1000))
    #cv2.imshow("ff", image)


    img = result.plot()
    img = Image.fromarray(img[:, :, ::-1])  
    img.save(output_directory + os.path.basename(image_file))
    # Press `q` to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Destroy all windows
cv2.destroyAllWindows()