import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

BEST_MODEL_PATH = r'C:\Users\Barru\Documents\Universitas Brawijaya\SEMESTER 5\Computer Vision\Project_Akhir_Compvis\bell_palsy_project\train_result_v12\weights\best.pt'
BELL_PALSY_THRESHOLD = 5.0

try:
    print("⏳ Memuat YOLOv8...")
    yolo_model = YOLO(BEST_MODEL_PATH)
    print("✅ YOLOv8 Berhasil Dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat YOLO: {e}")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    NOSE_TIP = 1

    l_eye = get_coords(LEFT_EYE)
    r_eye = get_coords(RIGHT_EYE)
    l_brow = get_coords(LEFT_BROW)
    r_brow = get_coords(RIGHT_BROW)
    l_mouth = get_coords(LEFT_MOUTH)
    r_mouth = get_coords(RIGHT_MOUTH)
    nose = get_coords(NOSE_TIP)

    brow_dist_left = np.linalg.norm(l_brow - l_eye)
    brow_dist_right = np.linalg.norm(r_brow - r_eye)
    
    mouth_dist_left = np.linalg.norm(l_mouth - l_eye)
    mouth_dist_right = np.linalg.norm(r_mouth - r_eye)

    diff_brow = abs(brow_dist_left - brow_dist_right)
    diff_mouth = abs(mouth_dist_left - mouth_dist_right)

    eye_inter_dist = np.linalg.norm(l_eye - r_eye)
    
    if eye_inter_dist == 0: return 0, {}

    total_score = ((diff_brow + diff_mouth) / 2) / eye_inter_dist * 100.0

    points_data = {
        'l_eye': l_eye, 'r_eye': r_eye,
        'l_brow': l_brow, 'r_brow': r_brow,
        'l_mouth': l_mouth, 'r_mouth': r_mouth
    }
    
    return total_score, points_data

cap = cv2.VideoCapture(0)

print("\n🚀 Analisis Bell's Palsy (MediaPipe + YOLO) Dimulai...")
print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    yolo_results = yolo_model.predict(source=frame, conf=0.25, verbose=False)

    mp_results = face_mesh.process(rgb_frame)

    for r in yolo_results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, "YOLO Detect", (x1, y1-5), 0, 0.5, (0, 255, 0), 1)

    if mp_results.multi_face_landmarks:
        for face_landmarks in mp_results.multi_face_landmarks:
            
            score, pts = calculate_mediapipe_asymmetry(face_landmarks.landmark, w, h)

            status = "SIMETRIS"
            color = (0, 255, 0)
            
            if score > BELL_PALSY_THRESHOLD:
                status = "POTENSI BELL'S PALSY"
                color = (0, 0, 255)

            if pts:
                cv2.line(frame, pts['l_brow'].astype(int), pts['l_eye'].astype(int), (0, 255, 255), 1)
                cv2.line(frame, pts['r_brow'].astype(int), pts['r_eye'].astype(int), (0, 255, 255), 1)
                
                cv2.line(frame, pts['l_mouth'].astype(int), pts['l_eye'].astype(int), (255, 0, 255), 1)
                cv2.line(frame, pts['r_mouth'].astype(int), pts['r_eye'].astype(int), (255, 0, 255), 1)

                for key in pts:
                    cv2.circle(frame, pts[key].astype(int), 3, (0, 0, 255), -1)

            cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Skor Asimetri: {score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Analisis Bell's Palsy - MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()