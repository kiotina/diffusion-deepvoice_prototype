# Diffusion Deepvoice Prototype

Real 음성만으로 diffusion 기반 deepvoice 이상 탐지를 검증하기 위한 최소 프로토타입입니다.

현재 단계에서는 원본 WAV를 검사하고, 학습에 바로 사용할 수 있는 고정 크기 log-Mel spectrogram으로 전처리합니다. Git 저장소 초기화와 원격 저장소 연결은 프로젝트 구조가 안정된 뒤 진행합니다.

## 확인된 데이터

- 원본: `D:/Workspace/Mel-spectrogram_MFCC/data/raw/real`
- 파일: WAV 1,819개
- 형식: 16 kHz, mono, 16-bit PCM
- 길이: 1.48–10.88초, 중앙값 4.28초
- 사용량: 기본값은 전체 1,819개이며, 최소 1,000개 미만이면 중단

## 전처리 규칙

- sample rate: 16 kHz
- 길이: 2초 (앞에서부터 겹치지 않게 모든 완전한 구간 사용)
- 잔여 구간: 1초 이상이면 오른쪽 zero padding, 1초 미만이면 제외
- padding mask: 실제 음성과 zero padding 프레임을 구분
- 데이터 분할: 원본 파일 기준 train/validation/test = 80/10/10
- 긴 사용자 입력: 2초 창을 1초씩 겹쳐 전체 구간을 검사하도록 분할
- Mel bins: 80
- FFT / hop: 1024 / 256
- log-Mel 범위: -80–0 dB
- 저장 범위: `[-1, 1]`
- 출력 shape: `(1, 80, 126)`

## 환경 준비

Python 3.11을 권장합니다. 기존 `Mel-spectrogram_MFCC/.venv`는 원래 Python 실행 파일이 제거되어 현재 사용할 수 없습니다.

```powershell
cd D:\Workspace\diffusion-deepvoice_prototype
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

PyTorch는 다음 모델 단계에서 사용하는 장치(CPU/CUDA)에 맞춰 별도로 설치합니다.

## 실행

```powershell
python scripts\inspect_dataset.py
python scripts\preprocess.py
python scripts\preprocess.py --limit 8 --output-dir data/processed/smoke
pytest
```

생성 결과는 `data/processed/`와 `artifacts/` 아래에 저장되며 Git 추적 대상에서 제외됩니다.

## 길이 처리 원칙

- 원본 WAV를 먼저 train/validation/test로 나눈 뒤 각 그룹 안에서 2초 구간을 생성합니다.
- 같은 원본에서 나온 모든 구간은 항상 같은 split에 들어갑니다.
- 완전한 2초 구간은 모두 사용하고, 마지막 잔여 구간이 1초 이상이면 오른쪽을 zero padding합니다.
- 마지막 잔여 구간이 1초 미만이면 학습 정보가 부족하다고 보고 제외합니다.
- 모든 구간은 `(1, 1, 126)` frame mask를 함께 저장합니다.
- 이후 학습 loss에 mask를 곱하면 padding에 해당하는 시간 프레임을 학습에서 제외할 수 있습니다.
- 긴 사용자 음성은 `make_inference_segments()`로 50% 겹치는 2초 구간들로 나눌 수 있습니다.
- 파일별 최대값 기준 dB 정규화는 녹음 환경 보존 방향을 결정할 때 다시 검토합니다.
