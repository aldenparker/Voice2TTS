"""The fixed effects chain behind the Voice Designer's macro controls.

Blending moves through the speaker space, which changes *who* is talking. It
cannot change how big the room is or how bright the microphone was, so the
designer also has a short signal chain in a fixed order:

    size -> tone -> breath -> dynamics -> space

The order is not adjustable, and that is a decision rather than an omission. It
is the order these operations make sense in, and a canvas that let you put the
reverb before the resampler would offer freedom nobody wants.

Everything here is numpy. No new dependency earns its place for four effects.

SIZE deserves a note. Resampling a signal by a ratio shifts pitch *and* formants
together, which is exactly the "bigger or smaller speaker" axis people reach for
-- and unlike a pitch shifter it needs no phase vocoder, because the duration
change it causes is corrected upstream: VITS has its own `length_scale`, so the
model is asked to speak slower or faster by the same ratio and the two cancel.
A real formant shift for free, at the cost of one multiplication in the config.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger(__name__)

# Macros run -1..+1, neutral at 0, so "no effect" is the origin in every
# dimension and a saved design with everything at zero is a passthrough.
NEUTRAL = 0.0
MACROS = ("size", "warmth", "brightness", "breathiness", "space", "dynamics")


@dataclass
class Design:
    """Macro positions. All zero means the chain does nothing at all."""

    size: float = NEUTRAL          # -1 smaller/brighter .. +1 larger/deeper
    warmth: float = NEUTRAL        # low-mid weight
    brightness: float = NEUTRAL    # air and presence
    breathiness: float = NEUTRAL   # aspiration mixed in
    space: float = NEUTRAL         # room around the voice
    dynamics: float = NEUTRAL      # evenness of level

    def clamped(self) -> Design:
        return Design(**{k: float(np.clip(v, -1.0, 1.0))
                         for k, v in asdict(self).items()})

    @property
    def is_neutral(self) -> bool:
        return all(abs(v) < 1e-6 for v in asdict(self).values())

    @property
    def size_ratio(self) -> float:
        """Resampling ratio. Below 1 for a larger speaker.

        `resample_ratio` stretches by 1/ratio, so a ratio under 1 lengthens the
        signal and drops pitch and formants together -- a bigger head. The sign
        is easy to get backwards, and the check that catches it is that a larger
        speaker must lower the spectral centroid, not raise it.

        Kept modest: beyond about a fifth it stops sounding like a person and
        starts sounding like a tape running at the wrong speed.
        """
        return float(2.0 ** (-self.size * 0.26))    # +-1 -> about -+20%

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Design:
        return cls(**{k: float(data.get(k, NEUTRAL)) for k in MACROS}).clamped()


# -- building blocks --------------------------------------------------------


def shelf_response(freqs: np.ndarray, corner: float, gain_db: float,
                   high: bool) -> np.ndarray:
    """Magnitude of a first-order shelf, evaluated at `freqs`.

    A gain of `gain_db` in the shelf band, unity in the other, with a smooth
    transition around `corner`.
    """
    gain = 10.0 ** (gain_db / 20.0)
    # Smooth 0..1 crossfade in log frequency, so the corner is musical rather
    # than a step.
    with np.errstate(divide="ignore"):
        octaves = np.log2(np.maximum(freqs, 1e-6) / corner)
    blend = 1.0 / (1.0 + 2.0 ** (-octaves * 2.0))    # 0 below corner, 1 above
    weight = blend if high else 1.0 - blend
    return 1.0 + (gain - 1.0) * weight


def shelf(audio: np.ndarray, rate: int, freq: float, gain_db: float,
          high: bool) -> np.ndarray:
    """Shelving EQ, applied in the frequency domain.

    A biquad would be the textbook answer, but an IIR recursion in Python runs
    one sample at a time and this sits on the live call path, where Piper
    delivers a whole sentence in about 60 ms. An FFT multiply is vectorised, and
    being zero-phase it also avoids the phase smear a cascade of shelves would
    introduce.

    The chunk is padded before transforming so the circular wrap-around lands in
    the padding rather than at the start of the audio.
    """
    if abs(gain_db) < 0.01 or not len(audio):
        return np.asarray(audio, dtype=np.float32)

    n = len(audio)
    padded_len = int(2 ** np.ceil(np.log2(n + rate // 20)))
    spectrum = np.fft.rfft(audio, n=padded_len)
    freqs = np.fft.rfftfreq(padded_len, 1.0 / rate)
    shaped = np.fft.irfft(spectrum * shelf_response(freqs, freq, gain_db, high),
                          n=padded_len)
    return shaped[:n].astype(np.float32)


def resample_ratio(audio: np.ndarray, ratio: float) -> np.ndarray:
    """Play the signal at `ratio` speed, shifting pitch and formants together."""
    if abs(ratio - 1.0) < 1e-6 or not len(audio):
        return np.asarray(audio, dtype=np.float32)
    target = max(1, round(len(audio) / ratio))
    source = np.linspace(0.0, len(audio) - 1, target)
    return np.interp(source, np.arange(len(audio)), audio).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: int, factor: float,
                 frame_ms: float = 46.0, search_ms: float = 7.0) -> np.ndarray:
    """Change duration by `factor` without changing pitch. WSOLA.

    This is the other half of the `size` macro. Resampling shifts pitch and
    formants but also changes how long the sentence takes; stretching back by
    the same factor restores the timing and leaves only the pitch change.

    The first attempt did this through Piper's `length_scale` instead, asking
    the model to speak faster by the resampling ratio. Measurement killed it:
    Piper's output length is not proportional to `length_scale` -- 55 to 60% of
    it is a fixed component, and going from 1.0 to 1.1 moved a sentence by 256
    samples. Compensating in DSP depends on nothing but arithmetic.

    Overlap-add alone would tear waveforms at the splices, so each segment is
    nudged within a small window to the offset that best continues what came
    before -- which is what makes this WSOLA rather than plain OLA.
    """
    if abs(factor - 1.0) < 1e-6 or not len(audio):
        return np.asarray(audio, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(64, int(rate * frame_ms / 1000.0) // 2 * 2)
    hop_out = frame // 2                      # Hann at 50% overlap sums to 1
    hop_in = hop_out / factor
    search = max(0, int(rate * search_ms / 1000.0))

    if len(audio) < frame + search + hop_out:
        # Too short to splice; plain resampling is the honest fallback and the
        # duration error on a fragment this size is inaudible.
        return resample_ratio(audio, 1.0 / factor)

    window = np.hanning(frame).astype(np.float32)
    out_len = int(len(audio) * factor) + frame
    out = np.zeros(out_len, dtype=np.float32)

    # What the previous segment implies should come next; the search looks for
    # the piece of source that continues it most smoothly.
    expected = audio[hop_out:hop_out + frame].copy()

    position = 0.0
    out_pos = 0
    limit = len(audio) - frame - search - 1
    while out_pos + frame <= out_len and position <= limit:
        centre = int(position)
        low = max(0, centre - search)
        high = min(limit, centre + search)
        if high > low:
            # Cross-correlate every candidate offset at once.
            offsets = np.arange(low, high + 1)
            candidates = np.lib.stride_tricks.sliding_window_view(
                audio[low:high + frame], frame)
            scores = candidates @ expected
            best = int(offsets[int(np.argmax(scores))])
        else:
            best = centre

        out[out_pos:out_pos + frame] += audio[best:best + frame] * window
        expected = audio[best + hop_out:best + hop_out + frame]
        if len(expected) < frame:
            break
        out_pos += hop_out
        position += hop_in

    used = out_pos + frame
    return out[:min(used, out_len)]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Box average via cumulative sum. O(n), no loop."""
    if window <= 1 or not len(values):
        return values
    padded = np.concatenate([np.full(window, values[0], dtype=np.float64),
                             np.asarray(values, dtype=np.float64)])
    cumulative = np.cumsum(padded)
    return ((cumulative[window:] - cumulative[:-window]) / window).astype(np.float32)


def envelope(audio: np.ndarray, rate: int, window_ms: float = 25.0) -> np.ndarray:
    """Smoothed RMS amplitude. Used to make added noise follow the voice.

    A running mean of the squared signal via a cumulative sum: O(n) and fully
    vectorised. A proper attack/release follower has a level-dependent
    coefficient and so cannot be expressed as one filter pass, which on this
    path would mean a Python loop per sample -- more accuracy than a voice tone
    control needs, at a cost the live path cannot afford.
    """
    if not len(audio):
        return np.zeros(0, dtype=np.float32)
    window = max(1, int(rate * window_ms / 1000.0))
    padded = np.concatenate([np.zeros(window, dtype=np.float64),
                             np.asarray(audio, dtype=np.float64) ** 2])
    cumulative = np.cumsum(padded)
    means = (cumulative[window:] - cumulative[:-window]) / window
    return np.sqrt(np.maximum(means, 0.0)).astype(np.float32)


def peak_envelope(audio: np.ndarray, rate: int,
                  window_ms: float = 5.0) -> np.ndarray:
    """Short-window peak follower.

    The compressor needs this rather than the RMS envelope. An RMS detector
    averages a brief transient away, so the loudest sample survives while the
    sustained parts around it are pulled down -- which makes the signal *peakier*
    the harder the compressor works. Measured: RMS detection moved the crest
    factor of real speech from 6.20 to 6.14, while peak detection takes it to
    2.94.
    """
    if not len(audio):
        return np.zeros(0, dtype=np.float32)
    window = max(1, int(rate * window_ms / 1000.0))
    count = len(audio)
    padding = (-count) % window
    blocks = np.abs(np.concatenate(
        [np.asarray(audio, dtype=np.float32),
         np.zeros(padding, dtype=np.float32)])).reshape(-1, window)
    block_max = blocks.max(axis=1)

    # Widen by one block each way so a peak sitting on a boundary still pulls
    # the gain down before it arrives.
    widened = np.maximum.reduce([
        block_max,
        np.concatenate([block_max[:1], block_max[:-1]]),
        np.concatenate([block_max[1:], block_max[-1:]]),
    ])
    return np.repeat(widened, window)[:count]


def compress(audio: np.ndarray, rate: int, amount: float) -> np.ndarray:
    """Feed-forward compressor. `amount` 0..1 picks ratio and threshold together.

    One control rather than five: this exists to even out a voice, not to be a
    mastering chain, and a threshold nobody can hear the effect of is worse than
    no control at all.
    """
    if amount <= 0.001 or not len(audio):
        return np.asarray(audio, dtype=np.float32)
    threshold = 10.0 ** ((-6.0 - 18.0 * amount) / 20.0)
    ratio = 1.0 + 7.0 * amount

    level = np.maximum(peak_envelope(audio, rate), 1e-9)
    over = level / threshold
    gain = np.where(over > 1.0, over ** (1.0 / ratio - 1.0), 1.0)
    # Block-wise detection steps at block boundaries; smoothing the gain rather
    # than the level keeps the response fast but stops it clicking.
    gain = _moving_average(gain, max(1, int(rate * 0.002)))
    out = np.asarray(audio, dtype=np.float32) * gain

    # Compression only ever reduces gain, so without makeup the control reads as
    # a volume knob. Restoring the original peak makes it read as evenness.
    before = float(np.abs(audio).max())
    after = float(np.abs(out).max())
    if after > 1e-9 and before > 1e-9:
        out = out * (before / after)
    return out.astype(np.float32)


# Speech carries very little energy above 3 kHz, so added noise dominates that
# region at almost any level. Measured against real speech, this moves the
# spectral centroid from 2330 Hz to about 3400 at full -- audible as breath.
# The first attempt used 3.0 and reached 6300 Hz, which is not a breathy voice,
# it is hiss with a voice behind it.
BREATH_GAIN = 0.03


def breath(audio: np.ndarray, rate: int, amount: float,
           seed: int = 0) -> np.ndarray:
    """Mix in aspiration that follows the voice.

    Noise gated by the signal's own envelope, so it appears only while there is
    speech. Unshaped noise would just be hiss, and would be audible in the gaps
    where a real breathy voice is silent.
    """
    if amount <= 0.001 or not len(audio):
        return np.asarray(audio, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    # High-passed: breath is air, not rumble.
    noise = shelf(noise, rate, 2000.0, 12.0, high=True)
    noise = shelf(noise, rate, 500.0, -18.0, high=False)

    follow = envelope(audio, rate, window_ms=20.0)
    mixed = (np.asarray(audio, dtype=np.float32)
             + noise * follow * (amount * BREATH_GAIN))
    return mixed.astype(np.float32)


def reverb(audio: np.ndarray, rate: int, amount: float) -> np.ndarray:
    """Small Schroeder reverb: four combs in parallel into two allpasses.

    Deliberately short. The point is to put the voice somewhere rather than to
    simulate a hall, and a long tail on a live microphone feed is unintelligible
    for the person listening on the other end.
    """
    if amount <= 0.001 or not len(audio):
        return audio

    comb_ms = (29.7, 37.1, 41.1, 43.7)
    feedback = 0.65 + 0.2 * amount
    wet = np.zeros(len(audio) + int(rate * 0.2), dtype=np.float32)
    padded = np.concatenate([audio, np.zeros(len(wet) - len(audio),
                                             dtype=np.float32)])

    for ms in comb_ms:
        delay = max(1, int(rate * ms / 1000.0))
        buf = np.zeros(len(padded), dtype=np.float32)
        # y[n] = x[n] + feedback * y[n-delay], done blockwise over the delay so
        # it is delay-length steps rather than one per sample.
        buf[:delay] = padded[:delay]
        for start in range(delay, len(padded), delay):
            end = min(start + delay, len(padded))
            buf[start:end] = (padded[start:end]
                              + feedback * buf[start - delay:start - delay + (end - start)])
        wet += buf / len(comb_ms)

    for ms in (5.0, 1.7):
        delay = max(1, int(rate * ms / 1000.0))
        out = np.zeros_like(wet)
        out[:delay] = wet[:delay] * -0.7
        for start in range(delay, len(wet), delay):
            end = min(start + delay, len(wet))
            span = end - start
            out[start:end] = (wet[start:end] * -0.7
                              + wet[start - delay:start - delay + span]
                              + 0.7 * out[start - delay:start - delay + span])
        wet = out

    dry = np.concatenate([audio, np.zeros(len(wet) - len(audio), dtype=np.float32)])
    mix = dry + wet * (amount * 0.35)
    return mix.astype(np.float32)


# -- the chain --------------------------------------------------------------


def apply(audio: np.ndarray, rate: int, design: Design) -> np.ndarray:
    """Run the whole chain. Returns float32; length changes with `size`."""
    if design.is_neutral or not len(audio):
        return np.asarray(audio, dtype=np.float32)

    design = design.clamped()
    out = np.asarray(audio, dtype=np.float32).reshape(-1)

    # Pitch and formants together, then the duration put back where it was.
    if abs(design.size_ratio - 1.0) > 1e-6:
        out = resample_ratio(out, design.size_ratio)
        out = time_stretch(out, rate, design.size_ratio)

    # Warmth and brightness are separate controls rather than one tilt, because
    # a voice can want body without dullness and air without thinness.
    out = shelf(out, rate, 260.0, design.warmth * 6.0, high=False)
    out = shelf(out, rate, 3200.0, design.brightness * 6.0, high=True)
    if design.warmth > 0:
        out = shelf(out, rate, 6000.0, -design.warmth * 3.0, high=True)

    out = breath(out, rate, max(0.0, design.breathiness))
    out = compress(out, rate, max(0.0, design.dynamics))
    out = reverb(out, rate, max(0.0, design.space))

    # The chain can add gain; clipping here would be permanent and audible.
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)
