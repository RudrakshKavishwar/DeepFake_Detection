import sys
from pathlib import Path
import argparse
from deepfake.processor import AdvancedDeepfakeProcessor   # CNN–RNN video model
from audio.processor import AdvancedAudioDeepfakeProcessor # CNN–RNN audio model


# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from deepfake.processor import AdvancedDeepfakeProcessor   # CNN–RNN for video
from captioning.captioner import AdvancedVideoCaptioner
from knowledge_graph.pipeline import build_and_visualize_kg
from utils.dataloader import DataLoader
from audio.processor import AdvancedAudioDeepfakeProcessor  # CNN–RNN for audio


# ------------------- VIDEO PIPELINE -------------------
def run_pipeline(video_path: str):
    print("=" * 60)
    print(f"🎬 Processing video: {video_path}")
    print("=" * 60)

    # Step 1: Deepfake Detection (CNN–RNN hybrid)
    df_proc = AdvancedDeepfakeProcessor(device="cpu")
    df_result = df_proc.detect_deepfake_advanced(video_path)
    print("\n[🔍 Deepfake Detection Result]")
    print(df_result)

    # Step 2: Captioning (real captions extracted)
    cap_proc = AdvancedVideoCaptioner(device="cpu")
    cap_result = cap_proc.generate_detailed_captions(video_path, frame_step=30)
    print("\n[📝 Captions]")
    for c in cap_result["captions"]:
        ts = c.get("timestamp", 0.0)
        txt = c.get("caption", "").strip()
        print(f" - {ts:.2f}s: {txt}")

    # Step 3: Knowledge Graph
    segments = [
        {
            "timestamp": c.get("timestamp", 0.0),
            "caption": c.get("caption", "").strip(),
            "actions": []
        }
        for c in cap_result["captions"]
    ]

    stats, msg = build_and_visualize_kg(segments, "data/graph.html")
    print("\n[🧠 Knowledge Graph]")
    print(stats)
    print(f"Graph visualization saved to {msg}")


# ------------------- AUDIO PIPELINE -------------------
def run_pipeline_audio(audio_path: str):
    print("=" * 60)
    print(f"🎤 Processing audio: {audio_path}")
    print("=" * 60)

    # Step 1: Deepfake Detection (CNN–RNN hybrid)
    audio_proc = AdvancedAudioDeepfakeProcessor(device="cpu")
    df_result = audio_proc.detect_deepfake_audio(audio_path)
    print("\n[🔍 Deepfake Detection Result]")
    print(df_result)


# ------------------- MAIN -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--audio", type=str, help="Path to a single audio file")
    parser.add_argument("--all", action="store_true", help="Process all videos in data/samples/")
    args = parser.parse_args()

    data = DataLoader("data")
    videos = data.list_videos()

    if args.video:
        run_pipeline(args.video)
    elif args.audio:
        run_pipeline_audio(args.audio)
    elif args.all:
        if not videos:
            print("⚠️ No videos found in data/. Please add videos to data/samples/")
        else:
            for v in videos:
                run_pipeline(v)
    else:
        if not videos:
            print("⚠️ No videos found in data/. Please add videos to data/samples/")
        else:
            run_pipeline(videos[0])  # default: process first video
