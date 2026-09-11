# Diffusion 학습 안내

## 구현 범위와 코드 읽는 순서

기존 전처리 결과를 읽어 real 음성의 Mel에 추가한 noise를 예측하는 학습 프로토타입입니다. 기존 manifest의 split을 그대로 사용합니다.

| 순서 | 파일 | 역할 |
| --- | --- | --- |
| 1 | `src/deepvoice_diffusion/training_data.py` | Mel/mask 로딩, shape·범위·split 누수 검사 |
| 2 | `src/deepvoice_diffusion/diffusion.py` | 선형 noise schedule, noisy Mel 생성, masked MSE |
| 3 | `src/deepvoice_diffusion/model.py` | timestep embedding과 skip connection을 사용하는 U-Net |
| 4 | `src/deepvoice_diffusion/training.py` | AdamW, 역전파, gradient clipping, 검증, early stopping |
| 5 | `src/deepvoice_diffusion/checkpoint.py` | 가중치·optimizer·진행 상태 저장과 복원 |

설정은 `configs/train.yaml`, 실행 진입점은 `scripts/train.py`입니다.

입력 Mel은 `(B, 1, 80, 126)`, mask는 `(B, 1, 1, 126)`입니다. U-Net은 noisy Mel과 주파수 방향으로 확장한 mask를 두 채널로 입력받습니다. 내부에서 시간축을 128로 맞추고, 채널 수를 `32 → 64 → 128 → 64 → 32`로 처리한 후 원래 크기의 noise 예측을 반환합니다. 기본 모델은 940,993개 파라미터를 가집니다.

매 학습 epoch마다 샘플별 timestep과 Gaussian noise를 바꿉니다. 1,000개 diffusion timestep 중 하나를 뽑아 바로 noisy Mel을 만들므로, 매 배치마다 U-Net을 1,000번 실행하지 않습니다. 검증에서는 학습과 다른 seed로 샘플마다 고정한 timestep/noise를 사용하여 epoch 사이 loss를 비교합니다.

패딩 위치의 noisy Mel은 정규화된 무음 값 `-1`로 고정합니다. loss는 실제 음성에 해당하는 Mel 원소의 오차 합을 유효 원소 수로 나눕니다. epoch 집계도 배치 평균의 단순 평균이 아닌 전체 유효 원소 기준입니다. mask는 패딩의 직접적인 loss 기여를 제외하지만 convolution/GroupNorm 내부의 영향을 완전히 제거하는 것은 아닙니다. STFT 경계 프레임도 기존 전처리 mask의 근사 규칙을 따릅니다.

## 설치

현재 로컬 환경에서 확인한 버전은 Python 3.11.16, PyTorch 2.14.0+cpu입니다. 프로젝트 루트에서 실행합니다.

```powershell
.\.venv\python.exe -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\python.exe -m pip install -e ".[dev,training]"
```

다른 환경에서는 해당 환경의 Python 경로를 사용하세요. CUDA 환경 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 장치에 맞게 선택합니다. `device: auto`는 CUDA 사용이 가능하면 CUDA를, 그 외에는 CPU를 선택합니다. 이번 검증은 CPU에서 수행했습니다.

## 짧은 실행과 테스트

```powershell
.\.venv\python.exe -m pytest -q
.\.venv\python.exe scripts/train.py --smoke --output-dir artifacts/training_smoke_02
```

`--smoke`는 manifest 순서의 train 16개, validation 8개만 사용하며 최대 20회 optimizer update를 실행합니다. 기본 batch size 8이면 작은 subset을 10 epoch 학습합니다. 전체 데이터 10 epoch 학습과는 다릅니다. 데이터가 더 적으면 사용 가능한 수만 사용합니다. `--max-steps 4`처럼 더 짧게 제한할 수 있습니다. 기존 결과가 있는 출력 폴더는 덮어쓰지 않으므로 새 폴더를 지정하세요.

저장 결과:

- `best.pt`: 완료된 epoch 중 검증 loss가 가장 낮은 체크포인트
- `last.pt`: 마지막 안전한 저장 지점의 체크포인트
- `run_config.json`: 설정, 장치, 데이터 수와 지문
- `history.csv`, `history.json`, `loss_curve.png`: 완료된 epoch별 train/validation loss
- `summary.json`: 초기/최저 검증 loss, 종료 이유와 실행 시간

체크포인트에는 모델, optimizer, epoch, 다음 배치 위치, 부분 epoch의 loss 누적값, best loss, early stopping 상태, RNG 상태, 설정과 데이터 지문이 포함됩니다. 주기적인 저장 및 epoch 종료/step 제한 시 저장을 수행합니다. 강제 종료 시 마지막 저장 이후의 update는 다시 실행합니다.

중간 종료한 실행을 **같은 폴더의 `last.pt`**에서 이어가는 예:

```powershell
.\.venv\python.exe scripts/train.py --resume artifacts/training_smoke_02/last.pt --epochs 12 --max-steps 4
```

`--max-steps`는 이번 실행에서 추가할 update의 상한입니다. 재개 시 `--smoke`, `--config`는 함께 지정하지 않으며 저장된 설정을 사용합니다. epoch 상한은 늘릴 수 있고, 데이터 내용이나 학습 설정이 바뀌면 재개를 거부합니다. 이미 early stopping 조건에 도달한 실행은 자동으로 추가 학습하지 않습니다. CPU에서 중간 재개와 연속 실행의 가중치/loss가 정확히 일치하는지 테스트했습니다. 다른 장치나 PyTorch 버전 사이의 수치 일치는 보장하지 않습니다.

전체 학습은 준비가 됐을 때 아래 명령으로 별도 실행합니다. 이번 구현 확인에서는 실행하지 않았습니다.

```powershell
.\.venv\python.exe scripts/train.py --config configs/train.yaml
```

기본은 batch 8, AdamW learning rate `1e-4`, 최대 30 epoch, 검증 loss가 5 epoch 연속 개선되지 않으면 중단입니다. early stopping과 best 선택에는 validation만 사용하며 test 배열은 읽거나 평가하지 않습니다.

## 이번 smoke 결과와 해석

실제 전처리 manifest는 train 3,218개, validation 433개, test 403개입니다. 이 중 train 16개, validation 8개로 CPU에서 20회 update를 완료했습니다. 결과는 `artifacts/training_smoke/`에 있습니다.

- 초기 validation masked MSE: `1.079924`
- 10번째 subset epoch validation masked MSE: `0.215916`
- train masked MSE: 첫 epoch `1.032176` → 마지막 epoch `0.262504`
- 초기 로딩/검증을 제외한 실행 구간: 약 14초

이 결과는 데이터 로딩부터 loss 감소와 저장까지 동작함을 확인합니다. 작은 subset의 noise 예측 loss 감소가 실제 deepvoice 판별 성능을 뜻하지는 않습니다. 화자 ID가 없어 파일 간 같은 화자에 의한 누수는 아직 배제할 수 없습니다. 현재는 원본 파일 단위 분리만 보장합니다.

역방향 sampling, 음성 복원, 이상 점수와 판별 임계값, fake 데이터 비교 및 최종 test 평가는 다음 단계입니다. 학습 시간 추정치는 작은 subset의 측정값을 확대한 참고값이며 전체 파일 읽기와 시스템 부하에 따라 달라집니다.
