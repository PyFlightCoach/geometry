import numpy as np
import pytest
from pytest import raises

from geometry.utils import apply_index_slice, get_index, get_value, make_smoothing_spline


def test_get_index():
    arr = np.arange(10)
    assert get_index(arr, 5) == 5
    assert get_index(arr, 5.5) == 5.5
    assert get_index(arr, 0) == 0
    assert get_index(arr, 9) == 9
    with raises(ValueError):
        get_index(arr, -1)
    with raises(ValueError):
        get_index(arr, 10)


def test_get_index_repeat():
    arr = np.array([0, 1, 1, 1, 2, 3, 4])
    assert get_index(arr, 3) == 5
    assert get_index(arr, 1) == 1
    assert get_index(arr, 1, direction="backward") == 3
    assert get_index(arr, 0.5) == 0.5
    assert get_index(arr, 1.5) == 3.5


def test_get_value():   
    arr = np.arange(10)
    assert get_value(arr, 5) == 5
    assert get_value(arr, 5.5) == 5.5
    assert get_value(arr, 0) == 0
    assert get_value(arr, 9) == 9
    with raises(ValueError):
        get_value(arr, 10)


def test_apply_index_slice():
    assert np.all(apply_index_slice(np.arange(10), slice(1, 5)) == np.array([1, 2, 3, 4, 5]))
    assert np.all(apply_index_slice(np.arange(10), slice(0.5, 1.5)) == np.array([0.5, 1, 1.5]))
    assert np.all(apply_index_slice(np.arange(10), slice(0.5, 3.5)) == np.array([0.5, 1,  2, 3, 3.5]))
    assert np.all(apply_index_slice(np.arange(1), slice(0, 5)) == np.array([0]))
    

def _noisy_signal(fs, duration=20.0, seed=0):
    t = np.arange(0, duration, 1 / fs)
    clean = np.sin(2 * np.pi * 0.2 * t) + 0.5 * np.cos(2 * np.pi * 0.05 * t)
    noisy = clean + np.random.default_rng(seed).normal(0, 0.05, len(t))
    return t, clean, noisy


def _original_auto_s_spline(data, index, cutoff=10, min_s=1e-4):
    """The 1.0.0 algorithm with an unclamped cutoff, valid above 2 * cutoff Hz."""
    from scipy.interpolate import make_splrep
    from scipy.signal import butter, sosfiltfilt

    sos = butter(
        N=4, Wn=cutoff, btype="highpass", fs=1 / np.median(np.diff(index)), output="sos"
    )
    noise = sosfiltfilt(sos, data)
    trim = int(len(data) * 0.05)
    s = max(np.var(noise[1 + trim : -2 - trim]) * len(index), min_s)
    w = np.ones(len(index))
    w[:trim] = 0.5
    w[len(index) - trim :] = 0.5
    return make_splrep(index, data, w=w, s=s)


@pytest.mark.parametrize("fs", [10.0, 9.99, 20.0, 5.0])
def test_make_smoothing_spline_low_rate_auto_s(fs):
    # the default 10 Hz cutoff is at or above Nyquist for these rates
    t, clean, noisy = _noisy_signal(fs)
    spline = make_smoothing_spline(noisy, t)
    fitted = spline(t)
    assert np.all(np.isfinite(fitted))
    assert np.max(np.abs(fitted - clean)) < 0.3


@pytest.mark.parametrize("fs", [100.0, 50.0])
def test_make_smoothing_spline_high_rate_unchanged(fs):
    # cutoff 10 Hz < 0.8 * Nyquist, so the clamp must not alter the result
    t, _, noisy = _noisy_signal(fs)
    new = make_smoothing_spline(noisy, t)
    old = _original_auto_s_spline(noisy, t)
    np.testing.assert_array_equal(new.t, old.t)
    np.testing.assert_array_equal(new.c, old.c)
    assert new.k == old.k


def test_make_smoothing_spline_clamp_boundary():
    # 25 Hz: 0.8 * Nyquist is 10 Hz up to rounding, so the cutoff is not reduced
    t, _, noisy = _noisy_signal(25.0)
    new = make_smoothing_spline(noisy, t)
    old = _original_auto_s_spline(noisy, t)
    np.testing.assert_allclose(new.c, old.c, rtol=1e-6)
