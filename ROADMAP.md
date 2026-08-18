# Roadmap

Working plan and progress tracker. `CHANGELOG.md` records what shipped; this
records what is next and why.

Conventions: tick a box only when the work is tested and committed. When a plan
turns out to be wrong, edit the entry to say so rather than deleting it — the
reason something changed is usually worth more than the plan itself.

---

## Status

**Latest release: 0.6.0. In progress: 0.6.1** — the Voice Studio shipped, and 0.6.1 fixes
the delay fast speech used to cause. Work on 0.7.0 happens on the
`0.7.0-translation` branch so main stays free for fixes like this one.

Release automation is verified: tagging `vX.Y.Z` or `vX.Y.Z-beta-N` runs lint
and both test suites, then builds and publishes. Betas go out as GitHub
pre-releases, which is what keeps them out of `/releases/latest` and so away
from everyone on the stable build.

| | |
|---|---|
| Pipeline | mic → Silero VAD → Whisper → Piper → N outputs |
| Latency | ~300 ms utterance to first audio (RTX 5080, `small.en`) |
| Tests | 496 self-test + 158 GUI, ruff clean, CI and the release gate run both |
| Release gate | `release.yml` verifies (tag shape, version, lint, both suites) before it builds; its PowerShell is executed by a test |
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

## 0.6.1 — Fast speech no longer waits ✅

**The problem.** Speaking quickly never leaves a 600 ms gap, so the endpoint rule
never fired and nothing was spoken until the speaker stopped or the 30 s cap cut
them off mid-word.

Measured on 34 s of continuous speech:

| | Before | After |
|---|---|---|
| First audio | **30.0 s** (the hard cap) | **5.0 s** |
| Segments | 2 | 7 |
| Speech kept | 33.5 s | 33.7 s |

**Lowering `min_silence_ms` could not have fixed it.** On that audio the Silero
probability sits at 1.0, only 7% of windows fall below the threshold at all, and
*no* gap reaches 600 ms — at 300 ms there are still zero cut points. The pauses
in fast speech are 100–250 ms, below anything the old rule could use.

- [x] **Soft endpoint.** Past `soft_endpoint_s` the silence requirement eases
      from `min_silence_ms` down to `min_silence_floor_ms`, reaching the floor at
      `max_segment_s`. Eased rather than stepped, so there is no length at which
      behaviour changes abruptly.
- [x] **A ceiling that cuts at the quietest moment.** When no pause arrives at
      all, the segment is cut at the lowest probability in the last second
      rather than at whatever sample the counter reached — at worst that lands
      between two words instead of mid-vowel.
- [x] **The remainder carries forward.** A forced cut is not the end of an
      utterance: the speaker is still talking, so the tail becomes the start of
      the next segment. Verified that 33.7 s of 33.9 s survives.
- [x] Exposed in Settings → Detection tuning, with the ceiling held above the
      split point so the pair cannot describe an impossible window.
- [x] `soft_endpoint_s = 0` disables the whole feature — relaxation *and*
      ceiling. An earlier version left the ceiling armed, which gave a third
      behaviour that was nobody's intent.

**The guarantee is about short utterances, not all of them.** Anything shorter
than `soft_endpoint_s` is segmented identically to before, and that is tested.
Something longer, followed by a real pause, may now end slightly sooner — at a
phrase boundary, which is the point.

Defaults are 3 s and 6 s: the ceiling is what bounds the wait, so it is set for
conversation rather than for transcript quality. 6 s still leaves ~5 s of
context per chunk, well above where recognition starts to suffer. Raise both for
fewer, longer, better-punctuated chunks.

### Still open

- [ ] **Push-to-talk is unchanged.** A long key press has the same problem, but
      no VAD runs in that mode, so there are no probabilities to find a quiet
      moment with. Options: run the VAD alongside PTT purely to find cut points,
      or cut on signal energy. Less pressing, because releasing the key is a
      control the speaker already has.
- [ ] **No context is carried across a cut.** Whisper sees each segment cold, so
      a sentence split across two segments loses the thread and may come back
      with odd capitalisation or punctuation. `faster-whisper` accepts an
      `initial_prompt`; passing the previous segment's tail, plus ~200 ms of
      audio overlap, should recover most of it. Worth doing before 0.7.0's
      streaming work, which makes the same problem sharper.

---

## 0.7.0 — Live translation

**Goal.** Speak English, have the other side hear German. A translation stage
between recognition and speech, running entirely on this machine.

Also in this release: **streaming recognition as a third mode**, which removes
the floor 0.6.1 left behind. See below.

**Not in scope.** Translating what *they* say back to you — that needs their
audio, which means loopback capture and a second pipeline. Worth doing later,
but it is a different feature wearing the same word.

### Where it goes

The pipeline is `capture → VAD → Whisper → substitutions → Piper → outputs`.
Translation slots between recognition and synthesis, but the substitution stage
has to split around it:

- **Before** — fixes for what the recogniser mishears (names, jargon). These
  correct the *source* text and must happen before it is translated, or the
  translator faithfully carries the error across.
- **After** — fixes for what the voice says badly. These are properties of the
  *target* language and voice, and applying the English list to German output
  would be nonsense.

That split is the first real work, and it is worth doing even if translation
slipped: the current single stage is already doing two jobs.

### Engine — OPUS-MT on CTranslate2 ✅ spiked, and it is comfortable

`spike/08_translate.py` measured it end to end on en→de. **Viable with room to
spare.**

| | |
|---|---|
| Model load | 180 ms, once |
| 5 words | 38 ms |
| 13 words | 57 ms |
| 25 words (a long utterance) | 98 ms |
| Budget | 300 ms, and the pipeline is ~300 ms today |

So a long sentence adds about a third to end-to-end latency, and a typical one
adds a tenth. Beam size 4 costs 96 ms against beam 1's 78 ms for the same
output on these sentences; 4 is affordable and stays the default.

Quality is genuinely usable — *"The build finished, but two of the tests are
still failing on Windows"* came back as *"Der Build ist abgeschlossen, aber zwei
der Tests scheitern immer noch unter Windows."*

The reason this is cheap is dependencies: **CTranslate2 is already here**,
because faster-whisper runs on it.

### Models — we convert and host them ourselves ✅ done

The original plan was to take Argos Translate's pre-converted `.argosmodel`
packages, because converting Helsinki-NLP's weights needs `transformers` and
`torch` purely to rewrite files we would never train.

**The licence question decided it.** Argos state terms for their *software* and
say nothing about the model weights — not in the index, not in the package, not
in the README. "Probably inherits CC-BY-4.0 from upstream" is not a licence, and
a GPL application should not put a download button on a guess.

So `scripts/convert_mt.py` converts Helsinki-NLP's originals at build time.
torch and transformers are build-time only, already in the PyInstaller
`excludes`, and never reach a user's machine. The result is better on every
axis that was supposed to be the cost:

| | Argos package | Ours |
|---|---|---|
| Packaged size | 151 MB | **63 MB** (int8) |
| "can you hear me?" | 37 ms | **19 ms** |
| A 25-word sentence | 98 ms | **68 ms** |
| Licence | unstated | **CC-BY-4.0, shipped in the package** |

Each pair ships `model/`, both tokenizers, a `LICENSE` carrying the attribution
CC-BY requires and the Tiedemann & Thottingal citation, and a `metadata.json`.
A `manifest.json` lists every pair with its size and sha256, so the app can show
what is available without downloading 60 MB to find out, and the list does not
go stale inside a shipped build.

Two things that produce *fluent nonsense* rather than an error, both found by
measurement and now guarded by tests:

- **Marian keeps two tokenizers**, one per direction. Decoding output with the
  source tokenizer does not fail — it leaves raw `▁` pieces in the text.
- **The input needs an explicit `</s>`.** Without it the model rambles:
  *"Hallo Hallo Hallo, können Sie hören Sie mich hören????"*

Other measured facts:

- **`sentencepiece` is a new runtime dependency.** 1.2 MB wheel.
- Argos packages carry a `stanza/` sentence splitter — a torch model, dropped on
  install. This pipeline already works one utterance at a time.
- **`argos-net.com` answers Python's default User-Agent with 403**, which no
  longer matters now that we host the models.
- `ct2-opus-mt-converter` needs only numpy and yaml and would make the CI job
  much lighter than transformers + torch, but the native OPUS-MT host 404'd on
  the filenames tried. Worth revisiting if the conversion job gets slow.

Alternatives considered:

| Option | Why not |
|---|---|
| Whisper's own `task="translate"` | ✅ **Done** — see below. To English only, but free. |
| NLLB-200 (distilled 600M) | 200 languages in one model, but **CC-BY-NC**. A non-commercial model inside a GPL application is a licence story we should not want. |
| argos-translate | Wraps the same OPUS-MT/CTranslate2 stack, and adds a dependency to do it. |
| A cloud API | Contradicts the entire premise. Nothing in this app leaves the machine. |

Pairs OPUS-MT does not publish directly pivot through English, at the cost of a
second hop and its compounding errors.

### The second method: Whisper translates it itself ✅ done

Whisper can emit English instead of the spoken language for one extra
parameter and no download. Measured against German synthesized by
`de_DE-thorsten-medium` and fed back through our own `WhisperEngine`:

| Model | "Hallo, kannst du mich hören?" | Per utterance, CPU |
|---|---|---|
| `base` | *"Hello, can you be nice to me?"* | 0.36 s |
| `small` | *"Hello, can you hear me?"* | 1.1 s |
| `medium` | *"Hello, can you hear me?"* | 3.1 s |

So it works, but **`base` is not good enough and `medium` is not worth 3×**.
`small` is the floor, and the Translate tab says so when a smaller one is
selected. All three mistranslated "Build" as a German word, which a dedicated
model working on correct English text does not.

The two methods are mutually exclusive in the interface — running both would
translate the text twice, the second time from a language it is no longer in.

### Prerequisites

- [x] **Multilingual recognition.** The plain (non-`.en`) Whisper models are now
      offered, along with a spoken-language picker and `auto` detection. The
      `.en` models are still the default and still better at English; the
      config refuses the combinations that cannot work (translating *from* a
      language an English-only model cannot hear, or detecting the language of
      a model that only knows one).
- [ ] **The voice must match the target language.** Piper voices are
      language-specific, and speaking German text with an English voice produces
      confident gibberish. The existing language guard already knows how to
      detect this mismatch; here it has to *drive* the voice choice rather than
      warn about it.

### Work

- [x] **Spike first** — `spike/08_translate.py`. Done; see the table above.
- [x] **Decide the model licence question.** Settled: we convert Helsinki-NLP's
      CC-BY-4.0 originals ourselves and host the result, so the terms are known
      and the attribution travels with the model. See above.
- [x] `scripts/convert_mt.py` — conversion, packaging, checksums, manifest
- [ ] The download side: read the manifest, fetch, verify sha256, install
- [ ] A CI job that runs `convert_mt.py --all` and attaches the assets to a
      `models-*` release tag
- [x] Split the substitution stage into source-side and target-side lists,
      with the Words tab editing both
- [x] Wire the chain into the pipeline, between the two rule sets
- [x] The Translate tab: language pair, model downloads, and a route line that
      says why a combination will not work
- [ ] Streaming recognition as a third mode (below)
- [ ] `translate.py` — model download, cache, and a `translate(text, src, dst)`
      that is a pure function over a loaded model
- [ ] Language pair picker, with the download surfaced like the voice library's
- [ ] Pivot through English where a direct pair does not exist, and say so
- [ ] Wire `task="translate"` as the zero-download path to English
- [ ] Review-before-speaking becomes far more valuable here and should probably
      default to on when translating: an ASR error feeds a translator that will
      produce something fluent and wrong
- [ ] Show both texts in the transcript, source above target

### Streaming recognition — a third mode

0.6.1 bounded the wait by cutting speech into segments; it did not remove it.
The floor is still `max_segment_s`, because recognition cannot start until a
segment is closed. Streaming removes that floor by recognising *while* someone
is still speaking.

**How it works.** Re-transcribe a growing buffer every second or so and compare
consecutive passes. Whatever two passes agree on is stable and can be spoken;
the rest is still in flux and is held. This is the LocalAgreement rule that
whisper-streaming uses, and it works because Whisper's output for the early part
of a buffer stops changing once enough context follows it.

**Why it is a mode and not the default.** Three costs, all real:

- **Compute.** The same audio is decoded many times. A 10 s utterance
  re-transcribed every second is roughly 10 passes over a growing buffer rather
  than one pass at the end. Fine on a 5080, not fine on the CPU path that ships
  by default.
- **It fits speech synthesis badly.** You cannot un-speak. Once Piper has said a
  word it is out, so only *stable* text can be spoken — and to avoid sounding
  robotic it has to be buffered to a phrase or sentence boundary anyway, which
  gives back part of the latency the technique bought.
- **Prosody.** Sentences spoken in pieces lose their intonation contour. The
  Designer's own effects chain has the same character: fine in isolation,
  noticeable when it happens every few words.

So it belongs behind a switch, alongside push-to-talk and VAD, for people who
want the lowest possible lag and have the hardware to pay for it.

- [ ] **Spike it before building.** Two numbers decide the design: how long
      until a word stabilises (which is the true latency, not the transcription
      time), and what the repeated decoding costs on both CPU and GPU. Reuse
      `spike/08_translate.py`'s shape — measure first, commit after.
- [ ] `streaming.py` — a growing buffer, a scheduled re-transcribe, and the
      agreement rule as a pure function over two token lists, so it can be
      tested without a model
- [ ] Speak only stable text, buffered to a phrase boundary
- [ ] A hard fallback to segmented mode when a pass takes longer than the
      interval — otherwise the buffer grows without bound and the lag it was
      meant to remove comes back worse
- [ ] Mode selector: push to talk / automatic / **streaming**, with the compute
      cost stated plainly next to it
- [ ] Interaction with translation: a translator sees a stable prefix rather
      than a whole sentence, and translation quality depends heavily on having
      the whole clause. Very likely translation must stay on sentence
      boundaries even when recognition does not — measure before assuming
      either way.

### Risks

| Risk | Mitigation |
|---|---|
| Latency makes it unusable in a live call | Measure in the spike, before building anything on top |
| An ASR error becomes a fluent mistranslation | Review-before-speaking on by default; show both texts |
| Pivoting compounds errors | Prefer direct pairs; mark pivoted ones in the UI |
| Model licences | **Open — see below.** OPUS-MT itself is CC-BY-4.0, but the Argos packages state no licence at all |

---

## 0.8.0 — Linux

**Goal.** Run properly on Linux. Not a port that technically starts, but the
same application: tray, hotkeys, virtual microphone, Studio.

**Only Linux.** macOS is not planned. It would need its own virtual audio
device (nothing like a null sink ships with the OS), its own signing and
notarisation, and Apple hardware to test on.

### The platform layer

OS-specific code currently sits wherever it was first needed, across fourteen
modules: `winreg` in three, `ctypes.WinDLL` in seven, `creationflags` in four,
WASAPI assumptions in four, `.exe` paths in six. `platform_win.py` already
exists for DPI, the single-instance guard and run-at-login — the right idea,
applied to a fraction of the surface.

So: a `voice2tts/platform/` package.

    platform/
      __init__.py     picks the implementation once, at import, and re-exports
      base.py         the interface every OS must satisfy
      windows.py      what platform_win.py is now, plus what is scattered today
      linux.py        the new one

`__init__.py` chooses by `sys.platform` and exposes plain functions, so callers
never branch on the OS themselves. The rule that keeps this honest: **no
`sys.platform` test outside `platform/`**, enforced by a test that greps for it.
Without that, the layer grows holes the first time something is urgent.

### What actually differs

| Concern | Windows | Linux |
|---|---|---|
| Virtual microphone | VB-CABLE, downloaded and installed | PipeWire/PulseAudio `module-null-sink`, created at runtime — **no install, no third-party driver**, which makes first run *simpler* than on Windows |
| Autostart | Registry `Run` key | XDG autostart `.desktop` |
| Host API | WASAPI, filtered to avoid duplicates | ALSA vs Pulse vs PipeWire through PortAudio; same duplicate problem, different names |
| Global hotkeys | `pynput` low-level hook | X11 fine; **Wayland is the hard problem** — see risks |
| Tray icon | `pystray` win32 | StatusNotifierItem; GNOME needs an extension |
| CUDA libraries | `ctypes.WinDLL` by absolute path | `ctypes.CDLL` and `LD_LIBRARY_PATH`; the nvidia wheels ship `.so` |
| Studio pack | Embeddable Python zip | `venv` from the system interpreter — embeddable Python is Windows-only |
| Subprocesses | `CREATE_NO_WINDOW`, `CREATE_NEW_PROCESS_GROUP` | `start_new_session=True`; no console to hide |
| Graceful stop | `CTRL_BREAK_EVENT` | `SIGINT`, which is what the trainer already expects |
| Paths | `%APPDATA%`, `%LOCALAPPDATA%` | `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` — `paths.py` already centralises this, so it is one module |
| Updates | Downloads and runs the installer | **Disabled.** Updates come from the package manager; running an installer over a distro package would be wrong |
| DPI / scaling | Per-monitor v2 via ctypes | Tk reads `Xft.dpi`; usually nothing to do |

### Packaging

- [ ] **Debian / Ubuntu** — `.deb`, depending on `python3`, `libportaudio2`, and
      `pipewire` or `pulseaudio`
- [ ] **Fedora** — `.rpm`, same shape
- [ ] **NixOS** — a flake exposing a package and a Home Manager module. The most
      work and the most reproducible; also the one that will find every
      undeclared dependency, because nothing is ambient
- [ ] CI builds all three plus the Windows installer from one tag
- [ ] Both suites run on Linux in CI. `bare_machine.py` matters more here, since
      a container has no audio at all

### Work

- [ ] Build `platform/`, move the Windows code into it **unchanged**, and add
      the test forbidding `sys.platform` elsewhere. No Linux code in this step:
      proving Windows still passes on a pure move is what makes the rest safe.
- [ ] `linux.py` — null sink, XDG autostart, host API choice, `.so` CUDA
      loading, venv-based studio pack
- [ ] Replace the VB-CABLE wizard step with null-sink creation on Linux
- [ ] Turn the in-app updater into a "your package manager handles this" notice
- [ ] Package for all three targets
- [ ] Test on X11 and Wayland, GNOME and KDE

### Risks

| Risk | Mitigation |
|---|---|
| **Wayland blocks global hotkeys** — the big one. Compositors do not let an ordinary process grab keys | Three options, in order: the `xdg-desktop-portal` GlobalShortcuts interface where the compositor implements it; reading `/dev/input` via evdev, which needs group membership and is intrusive; or accepting that push-to-talk is X11-only and VAD is the Wayland path. Decide from a spike, and say plainly in the docs which desktops get which. |
| Tray icon absent on GNOME | StatusNotifierItem needs an extension there. Ship a normal window as the fallback so the app is never invisible |
| PipeWire vs PulseAudio vs bare ALSA | Target PipeWire, now the default nearly everywhere, and fall back to Pulse. Bare ALSA gets no virtual microphone |
| The refactor breaks Windows | Move code without changing it, in its own commit, with both suites green before any Linux code exists |
| Three package formats is a lot of surface | The flake is the strictest; get it right first and the others follow |

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
