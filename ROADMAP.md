# Roadmap

Working plan and progress tracker. `CHANGELOG.md` records what shipped; this
records what is next and why.

Conventions: tick a box only when the work is tested and committed. When a plan
turns out to be wrong, edit the entry to say so rather than deleting it — the
reason something changed is usually worth more than the plan itself.

---

## Status

**Current release: 0.5.0** — published, installer and checksum attached.
Release automation is verified: tagging `v*` builds and publishes on its own.

| | |
|---|---|
| Pipeline | mic → Silero VAD → Whisper → Piper → N outputs |
| Latency | ~300 ms utterance to first audio (RTX 5080, `small.en`) |
| Tests | 210 self-test + 70 GUI, ruff clean, CI on every push |
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

### Phase 0 — Spike: is the speaker embedding reachable? (half a day)

**Gates Phase 3. Do this before committing to anything there.**

Multi-speaker Piper models take a speaker *index*; the embedding lookup happens
inside the ONNX graph. Blending needs to supply a *vector* instead.

- [ ] Download `en_GB-vctk-medium` (109 speakers) and inspect the graph
- [ ] Locate the embedding table and the `Gather` that indexes it
- [ ] Try replacing that `Gather` with a direct graph input
- [ ] Confirm a re-exported model still synthesizes, and that feeding the original
      speaker's vector reproduces the original speaker
- [ ] Measure whether inference speed is unchanged

**If it works** → Phase 3 as described.
**If it fails** → fall back to `piper.train.infer_torch` with arbitrary embeddings
(works, but slow to iterate and needs torch loaded just to audition), or cut
blending and ship the effects chain alone. Record which, and why, here.

### Phase 1 — Studio pack and hardware gate (~3 days)

The trainer needs torch, lightning, torchaudio and librosa — roughly 2.5 GB that
most users will never want. Reuse the `gpupack.py` pattern exactly: download on
demand into `%LOCALAPPDATA%`, extract, make importable, never bundle.

- [ ] `studiopack.py` — download, verify, install, uninstall, report size
- [ ] Extend `cuda.py` search paths if the pack ships CUDA-linked wheels
- [ ] Hardware probe: NVIDIA GPU, VRAM, free disk, CUDA pack present
- [ ] Gate thresholds: **≥ 8 GB VRAM**, **≥ 25 GB free disk**
- [ ] **IKWIAD override** — a config flag plus an explicit confirmation, recorded
      in diagnostics so a bug report shows the gate was bypassed

The gate is **advisory, not a refusal**. Under-spec hardware fails by running out
of memory, not by breaking anything, and a 6 GB card can plausibly train at a
reduced batch size. The real mitigation is checkpoint-and-resume (Phase 2), which
turns a crash into minutes lost rather than a night.

### Phase 2 — Voice Trainer (~3–4 weeks)

Record your own voice, fine-tune from an existing checkpoint, export.

- [ ] **Recording UI** — prompt sentences, record, review, re-record, level meter
      and clipping warning. Target 20–40 minutes of clean speech.
- [ ] Prompt set chosen for phonetic coverage rather than arbitrary text
- [ ] **Dataset preparation** — trim silence, normalise, resample to 22050 Hz,
      write Piper's expected layout, flag clips that are too noisy or too short
- [ ] **Training orchestration** — run `piper.train` as a subprocess, parse
      progress, show step count and loss, allow pause/stop
- [ ] **Checkpoint and resume** — survive a crash, a reboot, or a closed laptop
- [ ] **Export** via `piper.train.export_onnx` into `%APPDATA%\Voice2TTS\voices`
- [ ] **Provenance sidecar** — base checkpoint, dataset duration, step count,
      licence, so a voice can answer "where did you come from?" later
- [ ] Audition at checkpoints, so quality is visible before hours are spent
- [ ] Tests: dataset prep and export are pure functions; training itself gets a
      short smoke run behind a flag, not in CI

Expect 2–6 hours on a 5080 for a usable fine-tune.

### Phase 3 — Voice Designer (~1–2 weeks, gated on Phase 0)

Build a voice without training, by moving through the speaker space of a
multi-speaker model and shaping the result.

25 of 174 catalogue voices are multi-speaker; `en_US-libritts-high` has 904
speakers and `en_GB-vctk-medium` has 109.

- [ ] **Similarity map** — project speaker embeddings to 2D (UMAP or PCA),
      scatter plot, click to blend nearest neighbours by inverse distance,
      audition on hover. Projection precomputed and shipped as a small array.
- [ ] **Recipe mixer** — name 2–5 speakers with weights, for control and
      reproducibility once the map has found a region
- [ ] **Macro controls** — Warmth, Brightness, Breathiness, Size — each driving
      several parameters, with an Advanced disclosure over a fixed chain
      (pitch → formant → EQ → dynamics → space)
- [ ] `.v2tvoice` save/load, and appearance in the normal voice picker

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

- [ ] **Consent policy.** Should the trainer accept arbitrary audio files, or only
      recordings made inside it? This app pipes into Discord, so impersonation is
      the obvious misuse rather than a hypothetical. Recording in-app with a stated
      attestation is the conservative option. *Your call — but make it deliberately
      rather than by default.*
- [ ] **Ship both tiers in 0.6.0, or trainer only?** Recommendation: trainer only.
      It is a release on its own and the 2.5 GB optional pack deserves its own
      shakeout.
- [ ] **Base-voice licensing.** A fine-tune inherits its base checkpoint's terms.
      `voices.json` carries **no licence field** — it is in each voice's
      `MODEL_CARD` in the HuggingFace repo. Decide whether to fetch and display it,
      and whether to refuse bases that forbid derivatives.

### Risks

| Risk | Mitigation |
|---|---|
| Embedding surgery is not viable | Phase 0 runs first and is cheap; two fallbacks recorded |
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
