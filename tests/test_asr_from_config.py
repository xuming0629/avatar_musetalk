#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import os
import sys
import cv2 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.nets.whisper.audio2feature import Audio2Feature
from services.asr.base import MuseTalkWhisperASRService


def main():
    audio_processor = Audio2Feature()
    asr = MuseTalkWhisperASRService(
        audio_processor=audio_processor,
    )

    text = asr.transcribe("./assets/data/audio/sun.wav")

    print("ASR:", text)


if __name__ == "__main__":
    main()