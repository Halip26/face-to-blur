import cv2
import numpy as np
import sys
import os

# Jalur prototxt model Caffe
prototxt_path = "weights/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# berikan video path sebagai argumen
image_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# memuat gambar yang akan diuji
image = cv2.imread(image_path)

# Ekstrak nama file dari image_path
filename = os.path.basename(image_path)

# Memisahkan nama file dan ekstensi
name, extension = os.path.splitext(filename)

# Gabungkan direktori output & nama file yg ditambah akhiran "_blurred"
output_image_path = os.path.join(output_directory, f"{name}_blurred{extension}")

# dapatkan lebar dan tinggi gambar
height, width = image.shape[:2]

# ukuran kernel gaussian blur tergantung pada lebar dan tinggi gambar asli
kernel_width = (width // 7) | 1
kernel_height = (height // 7) | 1

# memproses gambar: mengubah ukuran dan melakukan pengurangan rata-rata
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# mengatur gambar ke dalam input jaringan saraf
model.setInput(blob)

# melakukan inferensi dan mendapatkan hasilnya
output = np.squeeze(model.forward())

# looping dgn parameters
for i in range(0, output.shape[0]):
    face_accuracy = output[i, 2]
    # get high akurasi wajah
    # jika akurasi wajah lebih dari 40%, maka buramkan rectangle blur pembatas (wajah)
    if face_accuracy > 0.4:
        # dapatkan koordinat rectangle blur sekitarnya dan tingkatkan ukurannya ke gambar asli
        box = output[i, 3:7] * np.array([width, height, width, height])
        # ubah menjadi bilangan bulat
        start_x, start_y, end_x, end_y = box.astype(np.int64)
        # dapatkan gambar wajah
        face = image[start_y:end_y, start_x:end_x]
        # terapkan gaussian blur ke wajah ini
        face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
        # masukkan wajah yang kabur ke gambar asli
        image[start_y:end_y, start_x:end_x] = face

# mengatur lebar & tinggi gambar di window
width = 1080
height = 540

# mengatur ukuran jendela sesuai dengan gambar asli
cv2.namedWindow("The results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("The results", width, height)

cv2.imshow("The results", image)
cv2.waitKey(0)
cv2.imwrite(output_image_path, image)
