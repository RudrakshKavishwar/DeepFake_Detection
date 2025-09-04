from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2

class AdvancedVideoCaptioner:
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

    def generate_detailed_captions(self, video_path, frame_step=30):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0
        captions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_step == 0:  # sample frames
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inputs)
                caption = self.processor.decode(out[0], skip_special_tokens=True)

                captions.append({
                    "timestamp": frame_id / fps if fps > 0 else frame_id,
                    "caption": caption
                })

            frame_id += 1

        cap.release()
        return {"captions": captions, "actions": []}  # keep same structure
