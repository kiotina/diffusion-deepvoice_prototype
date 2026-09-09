import numpy as np

from deepvoice_diffusion.audio import (
    fit_duration,
    make_inference_segments,
    make_segment,
    make_training_segment,
    sample_mask_to_frame_mask,
    waveform_to_logmel,
)
from deepvoice_diffusion.config import AudioConfig, MelConfig


AUDIO = AudioConfig(
    sample_rate=16_000,
    duration_seconds=4.0,
    crop_mode="random",
    random_seed=42,
    inference_overlap_seconds=2.0,
    trim_silence=False,
    trim_top_db=35.0,
)
MEL = MelConfig(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    f_min=20.0,
    f_max=8000.0,
    top_db=80.0,
)


def test_fit_duration_center_crops() -> None:
    # 10개 중 중앙 4개인 index 3~6이 선택돼야 한다.
    waveform = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(fit_duration(waveform, 4), [3, 4, 5, 6])


def test_fit_duration_pads_symmetrically() -> None:
    # 부족한 3칸을 왼쪽 1칸, 오른쪽 2칸으로 나눠 채운다.
    waveform = np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(fit_duration(waveform, 6), [0, 1, 2, 3, 0, 0])


def test_logmel_shape_dtype_and_range() -> None:
    # 외부 파일 없이 440 Hz 사인파로 Mel 변환 규격을 확인한다.
    time = np.arange(AUDIO.target_samples, dtype=np.float32) / AUDIO.sample_rate
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    mel = waveform_to_logmel(waveform, AUDIO, MEL)

    assert mel.shape == (1, 80, 251)
    assert mel.dtype == np.float32
    assert float(mel.min()) >= -1.0
    assert float(mel.max()) <= 1.0


def test_short_audio_gets_padding_mask() -> None:
    # 2초 음성을 4초로 만들면 mask에는 실제 구간과 padding 구간이 함께 있어야 한다.
    waveform = np.ones(32_000, dtype=np.float32)
    segment = make_segment(waveform, AUDIO.target_samples)
    frame_mask = sample_mask_to_frame_mask(
        segment.sample_mask,
        frame_count=251,
        mel_config=MEL,
    )

    assert segment.waveform.shape == (64_000,)
    assert segment.valid_samples == 32_000
    assert frame_mask.shape == (1, 1, 251)
    assert 0 < int(frame_mask.sum()) < 251


def test_random_training_crop_is_reproducible() -> None:
    # 같은 난수 seed라면 crop 시작점과 결과 waveform이 완전히 같아야 한다.
    waveform = np.arange(160_000, dtype=np.float32)
    first = make_training_segment(waveform, AUDIO, np.random.default_rng(7))
    second = make_training_segment(waveform, AUDIO, np.random.default_rng(7))

    assert first.source_start_sample == second.source_start_sample
    np.testing.assert_array_equal(first.waveform, second.waveform)
    assert first.valid_samples == AUDIO.target_samples


def test_ten_second_audio_is_fully_covered_for_inference() -> None:
    # 10초 전체를 50% 겹치는 네 개의 4초 창이 빠짐없이 덮어야 한다.
    waveform = np.ones(160_000, dtype=np.float32)
    segments = make_inference_segments(waveform, AUDIO)

    assert [segment.source_start_sample for segment in segments] == [
        0,
        32_000,
        64_000,
        96_000,
    ]
    assert all(segment.waveform.shape == (64_000,) for segment in segments)
