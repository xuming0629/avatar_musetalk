#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from services.config import load_yaml_config
from services.llm import build_llm_service


def main():
    cfg = load_yaml_config("configs/realtime_services.yaml")
    llm = build_llm_service(cfg)

    history = []

    user_text = "你好，我叫李雷，1+1等于多少？"
    answer = llm.chat(user_text, history=history)

    print("========== Answer ==========")
    print(answer)


if __name__ == "__main__":
    main()