# WAV 이상 점수와 평가

이 기능은 real 음성으로 학습한 diffusion U-Net의 **noise 예측 오차**를 이상 점수로 쓴다. 점수가 높을수록 fake일 것이라는 방향은 아직 검증되지 않았다. 출력은 확률이 아니며, 기본 판정 문구는 `기준 범위 내`와 `이상 점수 기준 초과`다.

## 현재 모델로 기능 확인

PowerShell에서 프로젝트 루트를 기준으로 실행한다. smoke는 기존 `artifacts/training_smoke/best.pt`를 사용해 real validation 4개로 임계값을 만들고, 겹치지 않는 validation 2개와 외부 음성 2개를 점수화한다. real test는 열지 않는다. 실행마다 **새 출력 폴더**를 지정한다.

```powershell
.\.venv\python.exe scripts\evaluate.py --output-dir artifacts\evaluation\smoke_001 smoke
```

결과의 `purpose=functional_smoke`는 연결 확인용이다. 이 임계값은 일반 `evaluate` 또는 `predict`에서 거부된다. `metrics.json`의 탐지 성능은 `null`이다.

## 전체 데이터 모델 평가

전체 real 데이터로 3 epoch 학습한 v2 체크포인트는 `artifacts/training_full_3epoch/best.pt`에 있다. 이 모델로 validation 임계값 산출과 real test·외부 WAV 점수화까지 실행한 결과는 `artifacts/evaluation/full_3epoch_001/`에 있다. real test 182개 중 7개가 임계값을 넘어 real FPR은 약 3.85%였다. 외부 파일은 라벨이 검증되지 않아 fake 탐지 정확도·F1은 계산하지 않았다.

1. `configs/evaluate.yaml`의 경로를 확인한다. `prepare --verify-contract`는 기존 train/validation의 **모든** 저장 Mel·mask를 원본 WAV에서 재현해 비교하고, 성공한 경우에만 `preprocess_contract.json`을 남긴다. 재전처리나 원본 변경은 하지 않는다. `prepare`는 manifest의 원본별 split을 유지한 `inventory.csv`도 만든다.

   검증이 불가능하거나 값이 다르면 계약을 만들지 않는다. 원본·설정·저장 배열의 차이를 확인한 다음, 필요한 경우에만 **새 출력 폴더**에 전처리해 다시 검증한다.

   ```powershell
   .\.venv\python.exe scripts\evaluate.py --output-dir artifacts\evaluation\run_001 prepare --verify-contract
   ```

2. 별도 학습 출력 폴더에 전체 학습을 실행한다. 검증된 전처리 계약이 있는 manifest를 쓰면 새 checkpoint는 v2로 저장된다. 기존 v1 `last.pt`의 재개 동작은 유지된다.

   ```powershell
   .\.venv\python.exe scripts\train.py --manifest data\processed\real_mels_2s\manifest.csv --output-dir artifacts\training_full
   ```

3. 새 `best.pt`를 지정해 validation 원본 WAV 전체로 임계값을 만든 뒤, 같은 모델로 real test와 외부 음성을 평가한다. 일반 calibration은 유효한 real validation 파일이 20개 미만이면 실패한다. 예상치 못한 읽기·계산 오류가 있으면 임계값을 저장하지 않는다.

   ```powershell
   .\.venv\python.exe scripts\evaluate.py --output-dir artifacts\evaluation\run_001 --checkpoint artifacts\training_full_3epoch\best.pt calibrate
   .\.venv\python.exe scripts\evaluate.py --output-dir artifacts\evaluation\run_001 --checkpoint artifacts\training_full_3epoch\best.pt evaluate
   ```

4. 임의 WAV만 점수화하려면 기존 `inventory.csv`와 `threshold.json`을 가리키고 새 출력 폴더를 사용한다.

   ```powershell
   .\.venv\python.exe scripts\evaluate.py --output-dir artifacts\evaluation\single_001 --checkpoint artifacts\training_full_3epoch\best.pt --inventory artifacts\evaluation\run_001\inventory.csv --threshold artifacts\evaluation\run_001\threshold.json predict --wav D:\path\sample.wav
   ```

기존 checkpoint v1에는 충분한 전처리 정보가 없어 `smoke`에서만 사용할 수 있다. 모델·전처리·점수 설정이나 calibration 파일 지문이 바뀌면 저장된 임계값은 거부된다. 기존 결과 파일은 자동으로 덮어쓰지 않는다.

## 점수와 데이터 해석

- WAV를 16 kHz mono로 읽는다. 1초 미만 및 리샘플 후 정확히 0 신호는 별도 제외한다. 1~2초는 오른쪽 패딩과 mask를 쓰고, 긴 음성은 2초 창을 1초씩 이동하며 끝에 맞춘 마지막 창을 추가한다. 각 **구간별** 최대 에너지를 Mel의 0 dB 기준으로 유지한다.
- 기본 timestep은 0부터 세는 `[100, 300, 500]`, noise seed는 `[1729, 2718]`이다. 두 CPU Gaussian 텐서를 파일·구간·단계 전체에서 재사용한다. 유효 Mel 위치의 masked MSE를 6회 측정해 구간 점수를 만들고, 구간 점수의 산술평균을 파일 대표 점수로 삼는다. 최고 구간 점수와 단계별 점수는 보조 정보다.
- 임계값은 real validation **파일 점수**의 95백분위수(`inverted_cdf`)다. `score > threshold`만 기준 초과이며 동점은 범위 안이다. 저장된 실제 calibration 초과율은 그 집합의 관측치일 뿐, 새 음성의 오탐률 보장이 아니다.
- `inventory.csv`의 외부 `fake` 폴더는 `declared_label=1`, `label_status=unverified`, 실제 `label` 공란이다. 폴더명만으로 fake 정답을 만들지 않는다. 출처를 직접 검증한 경우에만 별도 CSV `path,evidence`를 작성하고 `configs/evaluate.yaml`의 `verified_fake_manifest`에 지정한다. 이때 각 경로는 외부 폴더 안의 WAV여야 한다. 이 표시는 사용자 제공 근거에 의존하므로 파일의 합성 이력을 코드가 스스로 증명하지는 않는다.
- 검증된 real과 fake가 둘 다 있을 때 파일 단위 혼동행렬(`[real, fake]`), accuracy, precision, recall, F1, real FPR, specificity, balanced accuracy, ROC-AUC를 계산한다. 현재처럼 검증된 fake가 없으면 fake 탐지 지표는 `null`과 사유를 남기고, real FPR은 계산할 수 있다. ROC-AUC에는 선택 의존성 `pip install -e '.[evaluation]'`이 필요하다.
- `predictions.csv`는 파일 점수·최고 점수·판정·제외/오류 상태, `segments.csv`는 구간별·timestep별 점수, `metrics.json`은 라벨이 검증된 집합의 지표, `run.json`은 실행 목적과 설정·모델 지문을 담는다. JSON의 미정의 수치는 `null`이다.

원본 파일의 경로와 SHA-256으로 중복을 막고, inventory의 real 경로와 split을 원본 manifest와 다시 대조한다. 따라서 test 파일을 validation으로 바꿔 임계값 산출에 넣는 변경은 거부된다. 화자 ID가 확인되지 않았으므로 화자 독립 성능으로 해석할 수 없다. real과 외부 음성의 출처·녹음 조건 차이도 결과에 영향을 줄 수 있다. 점수 방향이나 집계법을 바꿔야 한다면 별도 개발 집합에서 검토하고, 최종 test 결과에 맞춰 수정하지 않는다.

## 외부 폴더 라벨을 가정한 임계값 실험

`scripts/experiment_threshold.py`는 **탐색적 실험 전용**이다. 앞서 실제 모델 추론으로 저장한 `full_smoke_inference_001/scores.csv`를 읽는다. 모델과 noise 설정이 같으므로 임계값만 바꾸기 위해 WAV 추론을 반복할 필요는 없다.

```powershell
.\.venv\python.exe scripts\experiment_threshold.py --scores-dir artifacts\evaluation\full_smoke_inference_001 --output-dir artifacts\evaluation\labeled_threshold_experiment_001
```

real validation 전체와 외부 파일의 절반을 개발용으로 사용하고, 개발용 **균형 정확도**가 최대가 되는 임계값을 고른다. 동점이면 더 높은 임계값을 선택한다. 남은 외부 파일과 real test는 임계값 선택에 넣지 않는다. 파일 ID 해시와 seed로 분할하므로 반복 실행 결과가 같다. 출력의 `label_verified=false`는 필수 해석 조건이다.

이 폴더의 파일명은 모두 같은 접두부를 공유하므로 개발/시험 분할이 화자 독립임을 보장하지 않는다. 이미 전체 외부 점수의 분포를 확인한 뒤 만든 실험이기도 하다. 따라서 이 실험의 높은 F1이나 정확도를 실제 deepfake 탐지 성능으로 발표해서는 안 된다. 기존 real-only 모델은 재학습하지 않았으며, 현재 v1 smoke 체크포인트의 점수를 사용한다.
