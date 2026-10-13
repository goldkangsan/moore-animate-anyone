# Moore-AnimateAnyone 실행 안내

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goldkangsan/moore-animate-anyone/blob/main/colab_animate_anyone.ipynb)

이 저장소는 [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)의 추론 스크립트와 설정을 담고 있습니다. 원본 `src/` 코드와 모델 가중치는 포함하지 않습니다. NVIDIA CUDA 환경을 권장합니다. 전체 실행은 이 저장소에서 검증된 것으로 표시하지 않습니다.

## 설치

```bash
git clone https://github.com/goldkangsan/moore-animate-anyone.git
cd moore-animate-anyone
mkdir -p upstream
git clone https://github.com/MooreThreads/Moore-AnimateAnyone.git upstream/Moore-AnimateAnyone
ln -s upstream/Moore-AnimateAnyone/src src

conda create -n animate python=3.10 -y
conda activate animate
pip install -r upstream/Moore-AnimateAnyone/requirements.txt
```

`tools/vid2pose.py`와 `scripts/pose2vid.py`가 저장소 루트의 `src/`를 import하므로 심볼릭 링크가 필요합니다. 원본 저장소의 의존성 설치 방법이 바뀌었다면 [원본 README](https://github.com/MooreThreads/Moore-AnimateAnyone)를 먼저 확인하세요.

## 입력과 가중치

```bash
mkdir -p inputs
cp /path/to/driving.mp4 inputs/driving.mp4
python tools/prepare_inputs.py --ref /path/to/character.jpg --video inputs/driving.mp4 -W 512 -H 784
python tools/download_weights.py
python tools/check_env.py
```

`prepare_inputs.py`는 참조 이미지를 `inputs/ref.png`에 저장하지만 driving video는 복사하지 않습니다. `run.sh`는 `inputs/ref.png`와 `inputs/driving.mp4`가 이미 있다고 가정합니다. 가중치는 `pretrained_weights/`에 내려받습니다. 다운로드에 접근 권한이 필요한 모델은 배포처의 절차를 따라야 합니다.

## 실행

한 번에 실행하려면:

```bash
bash run.sh
# 메모리 부족 시: bash run.sh --low_vram
# 프레임 수 변경: bash run.sh --frames 64
```

단계별로 실행하려면:

```bash
python tools/vid2pose.py --video_path inputs/driving.mp4 --output_path outputs/pose/driving_kps.mp4
```

`configs/prompts/animation.yaml`의 `test_cases`가 `./inputs/ref.png`와 `./outputs/pose/driving_kps.mp4`를 가리키는지 확인한 뒤:

```bash
python scripts/pose2vid.py --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 32
```

포즈 영상은 `outputs/pose/`에, 최종 생성 영상은 `output/` 아래의 날짜별 폴더에 저장됩니다. 브라우저 UI는 `python run_gradio.py`로 시작합니다.

## 자주 확인할 점

- CUDA 메모리가 부족하면 `-L`, `-W`, `-H`를 줄입니다.
- `src` import 오류가 나면 위 심볼릭 링크와 원본 저장소의 파일 구성을 확인합니다.
- ONNX 오류가 나면 설치된 `onnxruntime` 또는 `onnxruntime-gpu`가 환경에 맞는지 확인합니다.
- `pretrained_weights/` 누락은 `python tools/check_env.py`로 확인합니다.

원본 코드 및 가중치의 라이선스는 이 저장소의 문서와 별개입니다. 각 배포처의 조건을 확인하세요.
