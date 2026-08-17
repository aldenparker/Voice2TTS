# Roadmap

Working plan and progress tracker. `CHANGELOG.md` records what shipped; this
records what is next and why.

Conventions: tick a box only when the work is tested and committed. When a plan
turns out to be wrong, edit the entry to say so rather than deleting it — the
reason something changed is usually worth more than the plan itself.

---

## Status

**Latest release: 0.5.2. In progress: 0.6.0**, the Voice Studio — trainer and
designer both. The code is complete and untested on real hardware; betas are
being cut from it as `v0.6.0-beta-N`.

Release automation is verified: tagging `vX.Y.Z` or `vX.Y.Z-beta-N` runs lint
and both test suites, then builds and publishes. Betas go out as GitHub
pre-releases, which is what keeps them out of `/releases/latest` and so away
from everyone on the stable build.

| | |
|---|---|
| Pipeline | mic → Silero VAD → Whisper → Piper → N outputs |
| Latency | ~300 ms utterance to first audio (RTX 5080, `small.en`) |
| Tests | 463 self-test + 147 GUI, ruff clean, CI and the release gate run both |
| Release gate | `release.yml` verifies (tag/version, lint, self-test) before it builds; asserted by a test |
| Licence | GPL-3.0-or-later (Piper links eSpeak NG) |
| Distribution | Inno Setup, per-user, unsigned; winget manifests staged |

---

## 0.6.0 — Voice Studio

**Goal.** Make a voice inside the app that the app can then use, without leaving
it or learning a separate toolchain.

**Not in scope.** Cloning arbitrary recordings of other people (see *Decisions*),
real-time voice conversion, layering multiple voices at once, non-English training.

### Why this is tractable

`piper.train` ships with Piper — `vits`, `export_onnx`, `export_generator`,
`infer_torch`. Fine-tuning is a supported upstream path, not something to build,
and `export_onnx` emits exactly the `.onnx` + `.onnx.json` the voice library
already loads. A trained voice appears in the library and works with profiles,
previews and everything else for free.

### Phase 0 — Spike: is the speaker embedding reachable? ✅ DONE

**Result: viable, with no measurable cost. Phase 3 is unblocked.**

- [x] Download `en_GB-vctk-medium` (109 speakers) and inspect the graph
- [x] Locate the embedding table and the `Gather` that indexes it
- [x] Replace that `Gather` with a direct graph input
- [x] Confirm the patched model reproduces the indexed speaker **byte-identically**
- [x] Measure inference speed — 11.6 ms → 11.8 ms, unchanged

`emb_g.weight` is `[109, 512]` with exactly one consumer, and `sid` feeds nothing
else, so the surgery is a single node. See `spike/05_embedding_inspect.py`,
`spike/06_embedding_surgery.py` and FINDINGS §5.

Two things learned that change Phase 3:

- **VITS is stochastic.** `scales = (noise_scale, length_scale, noise_w)` feeds the
  duration predictor, so the same model gives different audio and different lengths
  on every call. Auditioning two blends at default noise compares two random draws
  as much as two voices, so the designer must pin the noise while comparing and
  restore it only for the final listen. This nearly produced a false negative in
  the spike itself.
- **The embedding table must be extracted to a side file.** ONNX Runtime drops
  `emb_g.weight` from the patched model because nothing references it, so the
  designer cannot read speaker vectors back out of a live session.

### Phase 1 — Studio pack and hardware gate ✅ DONE (bar one live run)

- [x] `studiopack.py` — probe, gate, install, status, uninstall
- [x] Hardware probe: NVIDIA GPU, VRAM, free disk, CUDA pack present
- [x] Gate thresholds: **≥ 8 GB VRAM**, **≥ 25 GB free disk**, advisory
- [x] **IKWIAD override** — `studio.ignore_hardware_check`, reported in
      diagnostics so a training bug report shows the machine was under-spec
- [x] Interpreter bootstrap verified (embeddable Python → pip → install → import)
- [x] Settings → Studio: hardware report, verdict, override, install/remove
- [ ] **Run a full `install()` including torch and confirm
      `torch.cuda.is_available()`** — to be done from the GUI during testing

**The training environment is a separate interpreter, not an import.** Rather than
following the `gpupack.py` pattern of making DLLs importable in-process, the studio
unpacks the embeddable Python into `%LOCALAPPDATA%\Voice2TTS\studio` and training
runs as a subprocess against it. Three reasons:

- a frozen PyInstaller build cannot pip-install into itself;
- torch ships its own cuBLAS and cuDNN, and `cuda.py` already preloads those by
  absolute path — a second resident copy is exactly the ambiguity that took two
  attempts to get right the first time;
- `piper.train` is a command-line trainer, so a subprocess is its natural shape,
  and uninstalling becomes deleting one directory.

Pinned to `torch==2.9.1+cu128`; cu128 is the first CUDA line with Blackwell
(sm_120) kernels, which 50-series cards need. Floating the version would let a
silent major bump break runs that worked yesterday.

The gate is **advisory, not a refusal**. Under-spec hardware fails by running out
of memory, not by breaking anything, and a 6 GB card can plausibly train at a
reduced batch size. The real mitigation is checkpoint-and-resume (Phase 2), which
turns a crash into minutes lost rather than a night.

### Phase 2 — Voice Trainer ✅ (code complete; untested on real hardware)

Record your own voice, fine-tune from an existing checkpoint, export.

- [x] **Two ways in, one of them obviously easier.**
      - *Record here* — the encouraged path. Shows a prompt, records, reviews,
        re-records, and tracks how much usable audio is banked against the target.
        Feeding people the script is the whole point: "read for 30 minutes" is a
        chore, "read this next sentence" is not, and it yields better phonetic
        coverage than whatever someone would improvise.
      - *Import files* — accepted, with a one-line confirmation that the speaker
        consented. Same dataset preparation afterwards, so quality checks apply
        equally.
- [x] **Prompt corpus.** CMU ARCTIC's 1132 sentences, committed to the repo
      rather than downloaded: festvox.org refuses HTTPS entirely, and a
      75 KB text file is not worth an unauthenticated download inside an
      installer. Remaining time is estimated from words actually read, so a slow
      reader is not told to read as many sentences as a fast one.
- [x] **Dataset preparation** — resample to 22050 Hz at VHQ, write Piper's
      `wav|text` layout, flag clips that are too noisy, too quiet or too short.
      Silence trimming is left to the trainer, which already does it
      (`VitsDataModule(trim_silence=True)`); doing it twice would clip onsets.
- [x] **Training orchestration** — `piper.train fit` as a subprocess, progress
      parsed from Lightning's bar, stop button
- [x] **Checkpoint and resume** — `last.ckpt` every epoch, so a crash costs one
      epoch. Resuming uses `--ckpt_path`; starting from a base voice uses
      `--model.warmstart_ckpt`, which is *not* interchangeable — see
      `training.py`'s module docstring.
- [x] **Base checkpoints** — not originally planned, and load-bearing: the
      voices we ship are inference-only `.onnx` and cannot be trained from.
      Fetched from `rhasspy/piper-checkpoints`, ~850 MB, resumable.
- [x] **Export** via `piper.train.export_onnx` into the user voices directory,
      with the config the trainer wrote, as `.onnx` + `.onnx.json`
- [x] **Provenance sidecar** — base checkpoint, dataset duration, epochs, app
      version, written as `.onnx.provenance.json`
- [x] **Audition at checkpoints** via `piper.train.infer_torch`, available
      mid-run, so hours are not spent on a run that is going nowhere
- [x] Tests: 360 self-test + 113 GUI. Everything that decides whether GPU hours
      are wasted — argument construction, checkpoint selection, export guards —
      is covered offline. Training itself is not run in CI.

Expect 2–6 hours on a 5080 for a usable fine-tune.

**Not yet proven on hardware.** Everything above is verified against
piper-tts 1.7.0's real signatures and runs headless, but no voice has been
trained end to end. Specifically unverified: the studio pack's torch install,
`torch.cuda.is_available()` inside it, whether Lightning's progress bar parses
as expected in practice, and whether an exported voice loads in the library.

### Phase 3 — Voice Designer ✅ (code complete; untested on real hardware)

Build a voice without training, by moving through the speaker space of a
multi-speaker model and shaping the result.

25 of 174 catalogue voices are multi-speaker; `en_US-libritts-high` has 904
speakers and `en_GB-vctk-medium` has 109.

**A designed voice is baked, not patched** (`spike/07_bake_blend.py`). Phase 0
turned `sid` into a graph input, which would have left the designer needing its
own synthesis path — and so its own sentence splitting, streaming, phonemizer
and previews. Freezing the blended vector as an initializer instead produces an
ordinary single-speaker voice: `PiperVoice` loads it, the library lists it, and
profiles work on it, with no engine changes. Baking speaker 0 reproduces speaker
0 to 1.3e-6, and the file gets *smaller*, since the speaker table is dropped.

**Voices cannot be compared as waveforms.** The duration predictor is
conditioned on the speaker embedding, so two speakers say the same sentence in
different numbers of samples even with both noise terms pinned to zero. A
sample-domain difference measures the misalignment, not the timbre — which is
how the first run of the spike managed to call a correct blend "beyond both
parents". Comparison uses a long-term average spectrum.

- [x] **Similarity map** — PCA to 2D, scatter plot, click to blend the nearest
      speakers by inverse distance. PCA rather than UMAP: UMAP means numba and
      llvmlite in the installer to lay out a few hundred points, while PCA is
      one SVD (7 ms for 109 speakers), needs only numpy, and is deterministic,
      so a position means the same thing between sessions and machines.
      Computed on load rather than shipped — it is faster than reading the file
      would be.
- [x] **Recipe mixer** — the click produces named weights, shown as percentages
      and saved as speaker labels rather than indices
- [x] **Macro controls** — Size, Warmth, Brightness, Breathiness, Dynamics,
      Space over a fixed chain. Every one is verified by measurement against
      real speech, which is how `size` was caught shipping inverted (it made
      "larger" brighter and shorter) and how breathiness was caught at 100×
      its usable level.
- [x] `.v2tvoice` save/load, and appearance in the normal voice picker

**Previewing is cached against the blend.** Baking and loading a 77 MB model
costs about 1.2 s, which is a long time in a loop whose entire purpose is
click-listen-adjust. The macros are post-processing, so only a change of
speakers pays that; moving a slider reshapes cached dry audio in a few
milliseconds.

**Speaker IDs are opaque** (`p3922`, `p239`, `TXHC`) with no gender, age or accent
metadata in the catalogue. That is what rules out a browsable list and makes the
map the primary interface: with 904 unlabelled speakers, navigating by ear is the
only option. Semantic axes ("deeper ↔ brighter") would need labelling by listening
or sourcing metadata from the original datasets — a later addition, not a
prerequisite.

**Not a node graph.** The effects order is known and fixed, so a canvas that lets
you put reverb before pitch-shift offers freedom nobody wants. Revisit nodes if
layering or parallel routing is ever wanted — at that point they become correct
immediately.

### Formats

Two artifacts, deliberately not sharing a format.

**Designed voice — `.v2tvoice`, declarative TOML.** Schema version, base voice key
and hash, speaker recipe (or embedding vector), macro and chain values, metadata.
A few hundred bytes.

It references a base rather than containing one. That makes it shareable in a chat
message and diffable in git, and it sidesteps licensing entirely: you distribute a
recipe and a pointer, not a derivative of anyone's weights. Promote to a zip
container (`voice.toml` + `embedding.npy` + `preview.wav`) only if binary is
genuinely needed.

**Trained voice — Piper's own `.onnx` + `.onnx.json`.** Do not invent a wrapper.
The library already loads it; a wrapper would break that. Add the provenance
sidecar alongside.

### Decisions needed

- [x] **Consent policy — decided: accept both, steer towards recording in-app.**
      The trainer takes existing audio files *and* records inside the app. The
      in-app path is made the easier one by supplying the script: prompts sized to
      the duration still needed, so nobody has to find something to read for half
      an hour. Importing carries a one-line confirmation that the speaker consented
      — cheap, and proportionate for a feature that clones a voice into a live
      call. Say the word if even that is unwanted.
- [x] **Ship both tiers in 0.6.0 — decided: both.** Trainer and designer go out
      together. Noting for the record that the recommendation was trainer only,
      on the grounds that the multi-GB optional pack deserves its own shakeout;
      that risk has not gone away, it has been accepted.
- [x] **Base-voice licensing — decided: fetch, show, and require agreement; do
      not refuse.** The `MODEL_CARD` beside each checkpoint is fetched and its
      licence shown in the confirmation dialog before an 850 MB download starts,
      so the terms arrive before the commitment rather than after it. A card with
      no licence line reads as "not stated on the model card" rather than being
      assumed permissive.

      Refusing bases that forbid derivatives was rejected: the cards link to
      external licence pages (Blizzard, CSTR, LibriVox and others) rather than
      naming an SPDX identifier, so any automatic verdict would be a guess
      dressed as authority. Showing the actual link and asking is honest;
      pretending to have parsed it is not.

### Risks

| Risk | Mitigation |
|---|---|
| ~~Embedding surgery is not viable~~ | **Retired** — Phase 0 proved it works at no cost |
| Surgery fails on a model laid out differently | Only vctk was tested; check libritts before relying on it |
| Training crashes hours in | Checkpoint-and-resume is a Phase 2 requirement, not a nicety |
| 2.5 GB pack scares people off | On demand only; the app is fully usable without it |
| Trained voices sound poor | Audition at checkpoints so quality is visible early |
| Torch pulls CUDA that conflicts with the GPU pack | Verify during Phase 1; `cuda.py` already preloads by absolute path |

---

## Backlog

Not scheduled. Roughly by value.

- **Code signing.** Unsigned installs trigger SmartScreen every time, and the
  VB-CABLE bootstrapper — an unsigned exe that downloads and elevates another exe —
  is a plausible antivirus false positive. Needs a certificate; EV gives instant
  reputation and is also the prerequisite if a custom audio driver is ever wanted.
- **Multilingual recognition.** Everything is pinned to English models today. The
  voice library offers 174 voices in many languages and the app warns about the
  mismatch, but does not solve it. Needs a multilingual Whisper model and a real
  language setting.
- **Semantic voice axes.** Depends on Phase 3 landing and on sourcing speaker
  metadata (VCTK ships age/gender/accent; L2-ARCTIC encodes native language).
- **Voice layering / node graph.** The trigger condition for nodes being correct.
- **Sentence-level pipelining.** Start synthesis on sentence one while recognition
  finishes, for long utterances.
- **Localisation of the UI.** Worth little before multilingual recognition.
- **Soundboard.** Deliberately skipped: plenty of apps do it and it pulls the
  project away from doing one thing well.

---

## Needs testing on real hardware

Things verified in code but not yet proven with sound.

- [ ] **VB-Audio Matrix routing.** `VBMatrix In N` → `VBMatrix Out N` is a naming
      convention, not a guaranteed path — Matrix is a router, and it is installed
      but not running on the development machine, so no audio has ever traversed
      it. Start Matrix, route `In 1` → `Out 1`, then Settings → Audio → **Test the
      Discord path** confirms it, or **Find the right device** reports which input
      actually receives audio.
- [ ] **Update install cycle.** Version comparison, throttling, validation and
      checksums are all tested; downloading a release and relaunching over the top
      has only run for real once.
- [ ] **Hotkeys inside a fullscreen game.** Some anti-cheat blocks low-level
      keyboard hooks and there is no workaround from this side.
- [ ] **Studio pack install.** Downloading the embeddable Python, un-isolating
      it, bootstrapping pip and installing torch has never been run to
      completion. Confirm `torch.cuda.is_available()` reports true afterwards.
- [ ] **A designed voice, by ear.** The macros are tuned by measurement —
      spectral centroid, crest factor, autocorrelation pitch — because that is
      what can be automated. Nobody has listened to one yet. Breathiness in
      particular is a judgement call: it sits at 1/100th of the first attempt,
      chosen to keep the centroid rise modest, and may still be wrong in either
      direction. Check also that a designed voice loads in the frozen build,
      where `onnx` is a new dependency.
- [ ] **A voice, end to end.** Record → train → audition → export → select it on
      the Voice tab and hear it in Discord. Until this is done, Phase 2 is code
      that should work rather than a feature that does. Watch in particular for:
      Lightning's progress bar parsing (falls back to no percentage, harmless),
      batch size versus actual VRAM, and whether the exported `.onnx.json` is
      accepted by the voice loader unchanged.
