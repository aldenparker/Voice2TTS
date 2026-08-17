# Spike findings

Machine: RTX 5080 (Blackwell, sm_120), driver 610.47, Windows 11.
Python 3.12.10, venv at `%USERPROFILE%\.venvs\voice2tts` (deliberately outside
OneDrive -- a CUDA venv is GBs of files and OneDrive sync churn causes file locks).

## 1. Whisper on Blackwell -- WORKS

`faster-whisper` 1.2.1 / CTranslate2 4.8.1, `small.en`, cuda + float16:

| stage | time |
|---|---|
| model load | 0.63 s |
| first inference (cold) | **6.73 s** |
| steady-state transcribe | **235 ms** for 8.58 s audio (36.5x realtime) |

No fallback to whisper.cpp needed. Transcript was verbatim-accurate.

**The 6.73 s cold pass must be absorbed at startup.** It is cuDNN kernel autotune on
first use. Later runs on the same machine warm up in ~0.2 s because the driver
caches compiled kernels (`%APPDATA%\NVIDIA\ComputeCache`), but that cache is cleared
by driver updates, so the app always runs a warmup inference on a dummy buffer at
startup rather than letting it land on the user's first utterance.

### The CUDA DLL trap

CTranslate2 resolves `cublas64_12.dll` / `cudnn64_9.dll` by bare name at first GPU
use. The pip `nvidia-*-cu12` wheels install them to `site-packages/nvidia/*/bin`,
which Windows does not search. Verified failures:

- `os.add_dll_directory()` -- **not sufficient** (this is the commonly cited fix)
- prepending to `os.environ["PATH"]` -- **not sufficient**
- `ctypes.CDLL("cublas64_12.dll")` -- fails even with both of the above

What works: preload each DLL by **absolute path** with `ctypes.WinDLL`, in
dependency order, before importing `faster_whisper`. Once loaded, CTranslate2's
lookup matches the already-resident module by base name. See `_cuda.py`.

This must survive into the packaged build -- PyInstaller will need the same
treatment plus explicit binary collection.

### The second CUDA trap: two different directory depths

Once the GPU pack became a runtime download, there were two layouts in play:

    site-packages/nvidia/cublas/bin/          root is "nvidia"  -> */bin
    %LOCALAPPDATA%/Voice2TTS/cuda/nvidia/cublas/bin/   root is "cuda" -> */*/bin

`cuda.py` only globbed `*/bin`, so it never found the downloaded pack -- meaning
GPU acceleration silently did nothing in the installed build while appearing to
work in development, because the pip-installed wheels sit at the depth the glob
expected and masked it.

This was invisible to every test run from the virtualenv. It only surfaced when the
frozen build's own `--check` reported "CUDA libraries not installed" on a machine
where the pack was demonstrably present. The regression test builds the layout in a
temp directory so the pip wheels cannot mask it again.

**Lesson: verify packaged-only code paths against the packaged build.** A venv has
things a fresh install does not.

## 2. Piper TTS -- WORKS, and is basically free

`piper-tts` 1.7.0 (has a Windows wheel), `en_US-lessac-medium`, CPU only:

| metric | value |
|---|---|
| time to first chunk | **60 ms** |
| full utterance (8.58 s audio) | 169 ms |
| realtime factor | 50.7x |
| output format | 22050 Hz mono float32 |

`PiperVoice.synthesize()` yields one `AudioChunk` per sentence, so sentence-level
streaming is free -- chunk 0 was ready at 60 ms, chunk 2 at 169 ms. No GPU needed;
leave onnxruntime on CPU and keep the GPU for Whisper.

## 3. Silero VAD needs a 64-sample context window

Found while building, not during the spike, and worth recording because it fails
silently.

Silero v5 expects each inference to see the previous window's last 64 samples
prepended to the current 512, i.e. 576 samples. The ONNX input shape is declared
`[None, None]`, so feeding a bare 512 raises no error -- it just returns near-zero
probabilities for obviously loud speech:

| input | mean prob on clear speech | frames over 0.5 |
|---|---|---|
| 512 samples (no context) | 0.003 | 0% |
| 64 context + 512 samples | 0.931 | 93% |

The bug looked like "VAD never triggers" with no diagnostic anywhere. `SileroVad`
now carries the context across calls.

## 4. Audio output -- multi-target design settled

All real devices expose WASAPI at 48000 Hz stereo. Piper emits 22050 Hz mono, so
every target needs resampling; `soxr.ResampleStream` does it incrementally and
keeps filter state across chunks (a per-chunk `soxr.resample` would click at every
boundary).

Fan-out design in `04_multiout.py`: one thread + queue + `ResampleStream` + gain
per target device. Separate devices run on independent clocks and can differ in
rate and channel count, so they cannot share a stream. Resampling happens on the
producer side, per target, since each may want a different rate.

## 5. The speaker embedding is reachable (Voice Studio, Phase 0)

Run `spike/05_embedding_inspect.py` then `spike/06_embedding_surgery.py`.

**Verdict: viable, with no measurable cost.** Blending voices is on.

A multi-speaker Piper model takes a speaker index and looks the embedding up
inside the graph. `en_GB-vctk-medium` turned out to be the easy case:

    emb_g.weight   [109, 512]
    /emb_g/Gather  inputs=[emb_g.weight, sid]  outputs=[/emb_g/Gather_output_0]

One table, **exactly one consumer**, and `sid` feeds nothing else. Removing that
Gather and exposing its output tensor as a new `speaker_embedding` input of shape
`(1, 512)` is the whole surgery: one node out, one input in, nothing downstream
changed.

| | |
|---|---|
| Reproduces the indexed speaker | **byte-identical** for speakers 0, 1, 42, 108 |
| Blend of two speakers | runs, and differs from both parents |
| Inference speed | 11.6 ms → 11.8 ms, i.e. unchanged |
| Embedding width | 512 |

### VITS is stochastic, which nearly produced a false negative

The first comparison run "failed" because the patched model produced different
audio lengths from the original. It was not a surgery bug: `scales` is
`(noise_scale, length_scale, noise_w)`, and both noise terms feed the duration
predictor, so **the same model gives different audio on every call** — 7168
samples one run, 9984 the next, from identical inputs.

Equivalence can only be tested with `scales = (0, 1, 0)`. Verified both ways: with
normal scales two runs differ, with zero noise they are bit-for-bit identical.

This matters for the Voice Designer beyond the test. Auditioning two blends with
default noise compares two random draws as much as two voices, so the preview must
pin the noise while comparing and only restore it for the final listen.

### One packaging note

ONNX Runtime drops `emb_g.weight` when loading the patched model, since nothing
references it any more — the log says so explicitly. The tensor is still in the
file and readable with the `onnx` library, but the designer should extract the
table to a small side file rather than expecting to read it back out of a live
session.

## 6. Why we do not ship our own virtual audio device

Researched 2026-08-16, because "just make our own cable" is the obvious idea and it
will otherwise get proposed again.

- Only a **kernel-mode driver** can create an audio endpoint that Discord sees.
  Windows enumerates endpoints through MMDevice/WASAPI; user-mode code cannot add
  one. This is why OBS can ship a virtual *camera* (a DirectShow filter, user-mode)
  but nobody ships a virtual *microphone* the same way.
- Any INF-installed driver on Win10/11 x64 must be **signed by Microsoft**. That
  needs a Partner Center hardware account, which needs an **EV code signing
  certificate** (~$400-600/yr, issued only to a verified legal business entity).
- The direction of travel is against us: the **April 2026** Windows update removed
  default trust for cross-signed kernel drivers, and a new kernel trust policy is
  rolling out in evaluation mode.
- Bundling VB-CABLE instead is also not free -- VB-Audio route "integration or
  distribution deals" through a negotiated agreement.

So `cable.py` is a **bootstrapper**: it downloads VB-CABLE from VB-Audio's own
servers at the user's request and runs their installer elevated. The user obtains
the software from the vendor; we redistribute nothing.

Live test caught the reason scraping comes before the pinned URL: the pack was
`VBCABLE_Driver_Pack43.zip` when written and `Pack45` a few hours later.

## 7. Latency budget (projected)

    VAD endpoint detect   ~100 ms  (Silero, tunable trailing silence)
    Whisper transcribe    ~235 ms  (short utterance, warm)
    Piper first chunk      ~60 ms
    output buffering       ~50 ms
    ------------------------------
    ~450 ms from end of speech to first audio at the cable

Comfortably conversational. The cold-start warmup is the only thing that would
break this.

## Not yet verified

- **VB-CABLE is not installed** -- no virtual output device exists on this machine
  yet. Needs a manual admin install + reboot; cannot be bundled.
- Actual audible playback (`04_multiout.py`) has not been run -- needs a human to
  confirm it sounds right.
- Mic capture and Silero VAD.
