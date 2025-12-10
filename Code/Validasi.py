import cv2
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
BEST_MODEL_PATH = r'C:\Users\Barru\Documents\Universitas Brawijaya\SEMESTER 5\Computer Vision\Project_Akhir_Compvis\Facial-Asymmetry-Analysis--YOLOv8-for-Bell-s-Palsy-Detection\bell_palsy_project\train_result_v12\weights\best.pt'
DATASET_PATH = r'C:\Users\Barru\Documents\Universitas Brawijaya\SEMESTER 5\Computer Vision\Project_Akhir_Compvis\Facial-Asymmetry-Analysis--YOLOv8-for-Bell-s-Palsy-Detection\Dataset\images\test'   # <-- Ubah ke folder dataset kamu
BELL_PALSY_THRESHOLD = 5.0

# ===============================
# LOAD YOLO MODEL
# ===============================
try:
    print("⏳ Memuat YOLOv8...")
    yolo_model = YOLO(BEST_MODEL_PATH)
    print("✅ YOLOv8 Berhasil Dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat YOLO: {e}")
    exit()

# ===============================
# INIT MEDIAPIPE
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# FUNGSI PERHITUNGAN MEDIAPIPE
# ===============================
def calculate_mediapipe_asymmetry(landmarks, image_w, image_h):
    def get_coords(index):
        lm = landmarks[index]
        return np.array([lm.x * image_w, lm.y * image_h])

    LEFT_EYE = 468
    RIGHT_EYE = 473
    LEFT_BROW = 105
    RIGHT_BROW = 334
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    l_eye = get_coords(LEFT_EYE)
    r_eye = get_coords(RIGHT_EYE)
    l_brow = get_coords(LEFT_BROW)
    r_brow = get_coords(RIGHT_BROW)
    l_mouth = get_coords(LEFT_MOUTH)
    r_mouth = get_coords(RIGHT_MOUTH)

    brow_dist_left = np.linalg.norm(l_brow - l_eye)
    brow_dist_right = np.linalg.norm(r_brow - r_eye)

    mouth_dist_left = np.linalg.norm(l_mouth - l_eye)
    mouth_dist_right = np.linalg.norm(r_mouth - r_eye)

    diff_brow = abs(brow_dist_left - brow_dist_right)
    diff_mouth = abs(mouth_dist_left - mouth_dist_right)

    eye_inter_dist = np.linalg.norm(l_eye - r_eye)
    if eye_inter_dist == 0:
        return 0, {}

    total_score = ((diff_brow + diff_mouth) / 2) / eye_inter_dist * 100.0

    points_data = {
        'l_eye': l_eye, 'r_eye': r_eye,
        'l_brow': l_brow, 'r_brow': r_brow,
        'l_mouth': l_mouth, 'r_mouth': r_mouth
    }

    return total_score, points_data


# ===============================
# LOOP TESTING DATASET
# ===============================
print("\n📁 Memulai Testing Dataset...\n")

image_files = [f for f in os.listdir(DATASET_PATH)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if len(image_files) == 0:
    print("❌ Tidak ada file gambar dalam folder dataset!")
    exit()

for img_name in image_files:
    img_path = os.path.join(DATASET_PATH, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"⚠️ Gagal membaca gambar: {img_name}")
        continue

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO DETECTION
    yolo_results = yolo_model.predict(source=frame, conf=0.25, verbose=False)

    # MEDIAPIPE
    mp_results = face_mesh.process(rgb_frame)

    # TAMPILKAN YOLO BOXES
    for r in yolo_results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "YOLO Detect", (x1, y1 - 10), 0, 0.6, (0, 255, 0), 2)

    # ANALISIS MEDIAPIPE
    if mp_results.multi_face_landmarks:
        for face_landmarks in mp_results.multi_face_landmarks:
            score, pts = calculate_mediapipe_asymmetry(face_landmarks.landmark, w, h)

            status = "SIMETRIS"
            color = (0, 255, 0)
            if score > BELL_PALSY_THRESHOLD:
                status = "POTENSI BELL'S PALSY"
                color = (0, 0, 255)

            if pts:
                for key in pts:
                    cv2.circle(frame, pts[key].astype(int), 3, color, -1)

            cv2.putText(frame, f"Asimetri: {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, status, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    print(f"✔️ Selesai menganalisis: {img_name}")

    cv2.imshow("Hasil Testing Dataset", frame)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("\n🎉 Testing dataset selesai!")
