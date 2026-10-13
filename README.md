# Moore-AnimateAnyone inference wrapper

Reference image and driving video를 입력으로 사용하는 [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) 추론용 스크립트 모음입니다. 이 저장소에는 원본 모델의 `src/` 코드나 가중치가 들어 있지 않습니다.

한국어 실행 안내: [README_KR.md](README_KR.md)

## 구성

- `tools/prepare_inputs.py`: 참조 이미지 크기 조정 및 입력 확인
- `tools/vid2pose.py`: driving video에서 포즈 영상 생성
- `scripts/pose2vid.py`: 모델 추론
- `run.sh`: 환경 확인, 가중치 다운로드, 포즈 추출, 추론 순서로 실행
- `run_gradio.py`: 브라우저 UI

원본 코드와 모델 가중치는 별도로 준비해야 합니다. 모델 및 가중치의 이용 조건은 각각의 배포처에서 확인하세요. 이 저장소의 명령은 코드 구성을 기준으로 정리했으며, 전체 파이프라인 실행 결과는 이 README에서 보증하지 않습니다.
