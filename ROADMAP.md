# Roadmap

Working plan and progress tracker. `CHANGELOG.md` records what shipped; this
records what is next and why.

Conventions: tick a box only when the work is tested and committed. When a plan
turns out to be wrong, edit the entry to say so rather than deleting it — the
reason something changed is usually worth more than the plan itself.

---

## Status

**Current release: 0.5.2** — 0.5.1 reverted the interface to Windows' native
appearance and shipped with every dropdown broken; 0.5.2 fixes that. Release
automation is verified: tagging `v*` runs lint and the self-test, then builds and
publishes.

| | |
|---|---|
| Pipeline | mic → Silero VAD → Whisper → Piper → N outputs |
| Latency | ~300 ms utterance to first audio (RTX 5080, `small.en`) |
| Tests | 214 self-test + 70 GUI, ruff clean, CI on every push |
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

### Phase 2 — Voice Trainer (~3–4 weeks)

Record your own voice, fine-tune from an existing checkpoint, export.

- [ ] **Two ways in, one of them obviously easier.**
      - *Record here* — the encouraged path. Shows a prompt, records, reviews,
        re-records, and tracks how much usable audio is banked against the target.
        Feeding people the script is the whole point: "read for 30 minutes" is a
        chore, "read this next sentence" is not, and it yields better phonetic
        coverage than whatever someone would improvise.
      - *Import files* — accepted, with a one-line confirmation that the speaker
        consented. Same dataset preparation afterwards, so quality checks apply
        equally.
- [ ] **Prompt corpus.** Phonetically balanced, and freely licensed enough to ship
      inside a GPL application. CMU ARCTIC's prompt list is the leading candidate:
      it is drawn from Project Gutenberg texts and was built for exactly this job.
      Harvard/IEEE sentences are the fallback. Estimate remaining time from words
      read so far rather than a fixed sentence count, since people read at very
      different speeds.
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

- [x] **Consent policy — decided: accept both, steer towards recording in-app.**
      The trainer takes existing audio files *and* records inside the app. The
      in-app path is made the easier one by supplying the script: prompts sized to
      the duration still needed, so nobody has to find something to read for half
      an hour. Importing carries a one-line confirmation that the speaker consented
      — cheap, and proportionate for a feature that clones a voice into a live
      call. Say the word if even that is unwanted.
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
