import cv2
import numpy as np
import time

# Jalur prototxt model Caffe
prototxt_path = "weights/deploy.prototxt.txt"

# Jalur model Caffe
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# memuat model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# untuk menyambungkan ke kamera default
cap = cv2.VideoCapture(0)

# forever looping untuk deteksi wajah & buat blur
while True:
    start = time.time()
    live, image = cap.read()
    # mendapatkan lebar dan tinggi gambar
    h, w = image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    # praproses gambar: ubah ukuran dan lakukan pengurangan rata-rata
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # mengatur gambar menjadi input jaringan saraf tiruan
    model.setInput(blob)
    # lakukan inferensi dan dapatkan hasilnya
    output = np.squeeze(model.forward())

    # looping untuk manggambar blur padaarea wajah
    for i in range(0, output.shape[0]):
        face_accuracy = output[i, 2]
        # dapatkan tingkat akurasi wajah
        # jika akurasi wajah lebih dari 40%, maka buramkan kotak pembatas (wajah)
        if face_accuracy > 0.4:
            # dapatkan koordinat kotak di sekitarnya dan tingkatkan ke gambar asli
            box = output[i, 3:7] * np.array([w, h, w, h])
            # ubah menjadi bilangan bulat
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # dapatkan gambar wajah
            face = image[start_y:end_y, start_x:end_x]
            # terapkan gaussian blur ke wajah terdeteksi
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # taruh wajah yang telah diburamkan ke dalam gambar asli
            image[start_y:end_y, start_x:end_x] = face

    cv2.imshow("Faces blurred v1.0", image)
    if cv2.waitKey(1) == ord("q"):
        break
    time_elapsed = time.time() - start
    fps = 1 / time_elapsed
    print("FPS:", fps)

cv2.destroyAllWindows()
cap.release()
