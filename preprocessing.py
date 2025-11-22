import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path
from affine_transform import estimate_affine_partial_2d

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.9
)

def normalization(image_bgr: np.ndarray) -> dict | None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    if len(results.multi_face_landmarks) != 1:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    h, w = image_bgr.shape[:2]
    pixel_landmarks = np.array(
        [
            [landmark.x * w, landmark.y * h] 
            for landmark in face_landmarks.landmark
        ],
        dtype=np.float32
    )

    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    left_iris_points = pixel_landmarks[LEFT_IRIS_INDICES]
    right_iris_points = pixel_landmarks[RIGHT_IRIS_INDICES]

    left_iris_center = np.mean(left_iris_points, axis=0)
    right_iris_center = np.mean(right_iris_points, axis=0)

    src_points = [
        left_iris_center,
        right_iris_center,
    ]

    target_eye_distance = 96
    nose_center_x = 128
    
    dst_points = [
        [nose_center_x - target_eye_distance / 2, 50],
        [nose_center_x + target_eye_distance / 2, 50],
    ]

    affine_matrix = estimate_affine_partial_2d(src_points, dst_points)
    affine_matrix = np.array(affine_matrix)

    normalized_image = cv2.warpAffine(
        image_bgr, 
        affine_matrix, 
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    ones = np.ones((pixel_landmarks.shape[0], 1))
    homogeneous_landmarks = np.hstack([pixel_landmarks, ones])  # (478, 3)
    transformed = (affine_matrix @ homogeneous_landmarks.T).T  # (478, 2)
    normalized_landmarks = transformed.astype(np.float32)
    return {
        "normalized_image": normalized_image,
        "normalized_landmarks": normalized_landmarks,
        "original_landmarks": pixel_landmarks,
    }

if __name__ == "__main__":
    MAIN_FACIAL_LANDMARKS = {
        'eyebrow_right': [336, 296, 334, 293],
        'eyebrow_left': [63, 105, 66, 107],
        'eyelid_right': [362, 398, 384, 385, 386, 387, 388, 466, 263],
        'eyelid_left': [33, 246, 161, 160, 159, 158, 157, 173, 133],
        'upper_lip_outer': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        'upper_lip_inner': [78, 146, 91, 181, 84, 17, 314, 405, 321, 375, 308],
        'lower_lip_outer': [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 291],
        'lower_lip_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    }

    IMAGES = Path("datasets").glob("*/*.jpg")
    if not IMAGES:
        print("No images found")
        exit()

    for image_path in IMAGES:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        
        indices = list(MAIN_FACIAL_LANDMARKS.values())
        indices = np.concatenate(indices)

        normalized_data = normalization(image_bgr)
        if normalized_data is not None:
            frame = normalized_data["normalized_image"]
            landmarks = normalized_data["normalized_landmarks"]
            original_landmarks = normalized_data["original_landmarks"]
            
            for landmark in landmarks[indices]:
                x, y = landmark
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)            
        
            for landmark in original_landmarks[indices]:
                x, y = landmark
                cv2.circle(image_bgr, (int(x), int(y)), 1, (0, 255, 0), -1)

            cv2.imshow("Original Image", image_bgr)
            cv2.imshow("Normalized Image", frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    face_mesh.close()