

import os
import sys
import cv2 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


from src.utils.audio_processor import AudioProcessor

if __name__ == "__main__":
    audio_processor = AudioProcessor()

    wav_path = "./assets/test.mp3"

    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(
        wav_path,
    )

    print("audio feature segments:", len(audio_feature))

    if len(audio_feature) > 0:
        print("first audio feature shape:", audio_feature[0].shape)

    print("librosa_feature_length:", librosa_feature_length)