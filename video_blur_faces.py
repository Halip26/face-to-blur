import cv2
import numpy as np
import time
import sys
import os

# Jalur prototxt model Caffe
prototxt_path = "weights/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# berikan video path sebagai argumen
video_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# capture frames from video
capture = cv2.VideoCapture(video_path)

# it creates a four-character code (fourcc) used to specify the video codec.
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video, image = capture.read()
print(image.shape)
output_video_path = cv2.VideoWriter(
    "output.avi", fourcc, 20.0, (image.shape[1], image.shape[0])
)

while True:  # forever looping
    start = time.time()
    captured, image = capture.read()

    # get width and height of the image
    if not captured:
        break
    h, w = image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1

    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(model.forward())
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # get the confidence
        # if confidence is above 40%, then blur the bounding box (face)
        if confidence > 0.4:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # get the face image
            face = image[start_y:end_y, start_x:end_x]
            # apply gaussian blur to this face
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # put the blurred face into the original image
            image[start_y:end_y, start_x:end_x] = face

    # menampilkan hasilnya
    cv2.imshow("The results", image)

    # tombol close jika tekan "q"
    if cv2.waitKey(1) == ord("q"):
        break
    time_elapsed = time.time() - start
    fps = 1 / time_elapsed
    print("FPS:", fps)
    output_video_path.write(image)

cv2.destroyAllWindows()
capture.release()
output_video_path.release()
