import numpy as np

from deepvoice_diffusion.audio import (
    fit_duration,
    make_inference_segments,
    make_segment,
    make_training_segments,
    sample_mask_to_frame_mask,
    waveform_to_logmel,
)
from deepvoice_diffusion.config import AudioConfig, MelConfig


AUDIO = AudioConfig(
    sample_rate=16_000,
    duration_seconds=2.0,
    segment_hop_seconds=2.0,
    minimum_remainder_seconds=1.0,
    inference_overlap_seconds=1.0,
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


def test_fit_duration_pads_only_on_the_right() -> None:
    waveform = np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(fit_duration(waveform, 6), [1, 2, 3, 0, 0, 0])


def test_logmel_shape_dtype_and_range() -> None:
    # 외부 파일 없이 440 Hz 사인파로 Mel 변환 규격을 확인한다.
    time = np.arange(AUDIO.target_samples, dtype=np.float32) / AUDIO.sample_rate
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    mel = waveform_to_logmel(waveform, AUDIO, MEL)

    assert mel.shape == (1, 80, 126)
    assert mel.dtype == np.float32
    assert float(mel.min()) >= -1.0
    assert float(mel.max()) <= 1.0


def test_short_audio_gets_padding_mask() -> None:
    waveform = np.ones(16_000, dtype=np.float32)
    segment = make_segment(waveform, AUDIO.target_samples)
    frame_mask = sample_mask_to_frame_mask(
        segment.sample_mask,
        frame_count=126,
        mel_config=MEL,
    )

    assert segment.waveform.shape == (32_000,)
    assert segment.valid_samples == 16_000
    assert frame_mask.shape == (1, 1, 126)
    assert 0 < int(frame_mask.sum()) < 126
    assert np.all(segment.waveform[:16_000] == 1)
    assert np.all(segment.waveform[16_000:] == 0)
    assert np.all(segment.sample_mask[:16_000] == 1)
    assert np.all(segment.sample_mask[16_000:] == 0)


def test_remainder_of_at_least_one_second_is_padded_on_the_right() -> None:
    waveform = np.arange(84_800, dtype=np.float32)  # 5.3초
    segments = make_training_segments(waveform, AUDIO)

    assert [segment.source_start_sample for segment in segments] == [0, 32_000, 64_000]
    assert [segment.valid_samples for segment in segments] == [32_000, 32_000, 20_800]
    np.testing.assert_array_equal(segments[-1].waveform[:20_800], waveform[64_000:])
    assert np.all(segments[-1].sample_mask[:20_800] == 1)
    assert np.all(segments[-1].sample_mask[20_800:] == 0)


def test_remainder_shorter_than_one_second_is_discarded() -> None:
    waveform = np.ones(75_200, dtype=np.float32)  # 4.7초
    segments = make_training_segments(waveform, AUDIO)
    assert [segment.source_start_sample for segment in segments] == [0, 32_000]


def test_exactly_one_second_remainder_is_kept() -> None:
    waveform = np.ones(80_000, dtype=np.float32)  # 5초
    segments = make_training_segments(waveform, AUDIO)
    assert len(segments) == 3
    assert segments[-1].valid_samples == 16_000


def test_six_second_audio_is_fully_covered_for_inference() -> None:
    waveform = np.ones(96_000, dtype=np.float32)
    segments = make_inference_segments(waveform, AUDIO)

    assert [segment.source_start_sample for segment in segments] == [
        0,
        16_000,
        32_000,
        48_000,
        64_000,
    ]
    assert all(segment.waveform.shape == (32_000,) for segment in segments)
