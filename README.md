# [How to Blur Faces in Images using OpenCV in Python](https://www.thepythoncode.com/article/blur-faces-in-images-using-opencv-in-python)

To run this:

- `pip3 install -r requirements.txt`
- To blur faces of the image `father-and-daughter.jpg`:

    ```bash
    python blur_faces.py father-and-daughter.jpg
    ```

    This should show the blurred image and save it of the name `image_blurred.jpg` in your current directory.

- To blur faces using your live camera:

    ```bash
    python blur_faces_live.py
    ```

- To blur faces of a video:

    ```bash
    python blur_faces_video.py video.3gp
    ```

## Output faces become blurred

  ![father-and-daughter_blurred.jpg](output/father-and-daughter_blurred.jpg)

## Path prototxt model Caffe

```python
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "weights/deploy.prototxt.txt"
```

## Path model Caffe

```python
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
```
