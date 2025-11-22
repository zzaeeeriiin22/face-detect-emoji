import cv2
import torch

from preprocessing import normalization
from train import ExpressionMLP, CLASSES

MODEL_PATH = "best_model.pth"

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    input_dim = int(checkpoint.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError(
            "input_dim is missing in checkpoint. Retrain the model with the updated training script."
        )

    model = ExpressionMLP(input_dim=input_dim, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        normalized_data = normalization(frame)
        if normalized_data is None:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        features = normalized_data["normalized_landmarks"]  # (478, 2)

        features_tensor = torch.tensor(
            features.flatten(), dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = model(features_tensor)  # shape: (1, num_classes)
            predicted_class = int(torch.argmax(output, dim=1).item())

        cv2.putText(
            frame,
            f"Predicted: {CLASSES[predicted_class]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()