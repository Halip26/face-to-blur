import cv2
import numpy as np
import sys

# Jalur prototxt model Caffe
prototxt_path = "weights/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# dapatkan nama file gambar dari baris perintah
image_file = sys.argv[1]

# membaca gambar yang diinginkan
image = cv2.imread(image_file)

# dapatkan lebar dan tinggi gambar
h, w = image.shape[:2]

# ukuran kernel gaussian blur tergantung pada lebar dan tinggi gambar asli
kernel_width = (w // 7) | 1
kernel_height = (h // 7) | 1

# memproses gambar: mengubah ukuran dan melakukan pengurangan rata-rata
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# mengatur gambar ke dalam input jaringan saraf
model.setInput(blob)

# melakukan inferensi dan mendapatkan hasilnya
output = np.squeeze(model.forward())
for i in range(0, output.shape[0]):
    face_accuracy = output[i, 2]
    # dapatkan tingkat akurasi wajah
    # jika akurasi wajah lebih dari 40%, maka buramkan kotak pembatas (wajah)
    if face_accuracy > 0.4:
        # dapatkan koordinat kotak sekitarnya dan tingkatkan ukurannya ke gambar asli
        box = output[i, 3:7] * np.array([w, h, w, h])
        # ubah menjadi bilangan bulat
        start_x, start_y, end_x, end_y = box.astype(np.int)
        # dapatkan gambar wajah
        face = image[start_y:end_y, start_x:end_x]
        # terapkan gaussian blur ke wajah ini
        face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
        # masukkan wajah yang kabur ke gambar asli
        image[start_y:end_y, start_x:end_x] = face
cv2.imshow("gambar", image)
cv2.waitKey(0)
cv2.imwrite("gambar_diblur.jpg", image)
