# Voice Arena — Speaker Diarization Pilot

Pilot experiment for the Voice Arena PM take-home: three sherpa-onnx embedding
models evaluated on 14 AMI clips, scored with pyannote.metrics + dscore, with
paired cluster bootstrap resampling.

## What's in this folder

Models (~100 MB total) and the pyannote-audio repo are not included — see Setup.

## Results

| Config | Model | DER (%) | Miss (%) | FA (%) | Conf (%) |
|---|---|---|---|---|---|
| A — auto | WeSpeaker ResNet-34 | **37.81** | 22.60 | 2.56 | 12.64 |
| B — auto | NeMo TitaNet-Small | 38.45 | 22.59 | 2.35 | 13.51 |
| C — auto | 3D-Speaker CAM++ | 45.68 | 22.52 | 3.38 | 19.77 |
| A — oracle | WeSpeaker ResNet-34 | 34.43 | 22.71 | 2.65 | 9.07 |

Bootstrap (B=10,000, paired, seed=7): C vs A is significant (group CI [1.05, 14.06]);
B vs A is not (group CI [−4.14, 4.50]).

## Setup

Requires Python 3.9+. Tested on macOS 14 (Apple Silicon).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install sherpa-onnx pyannote.metrics soundfile numpy intervaltree
```

### Download models

```bash
mkdir -p models/sherpa-onnx-pyannote-segmentation-3-0

# Segmentation — use the tar archive, NOT the HuggingFace direct link (returns a 29-byte stub)
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" \
  -o /tmp/seg.tar.bz2
tar -xjf /tmp/seg.tar.bz2 -C models/ --strip-components=1 \
  sherpa-onnx-pyannote-segmentation-3-0/model.onnx
mv models/model.onnx models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx

# Embeddings — note: release tag is misspelled "recongition" — that's correct
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx" \
  -o models/wespeaker_en_voxceleb_resnet34.onnx
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx" \
  -o models/nemo_en_titanet_small.onnx
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" \
  -o models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx
```

### Get audio reference data

```bash
git clone --depth 1 https://github.com/pyannote/pyannote-audio.git
```

`pilot/ds/ref.rttm` and `pilot/ds/all.uem` are already included. If you need
to regenerate them:

```bash
cat pyannote-audio/tests/data/debug.train.rttm \
    pyannote-audio/tests/data/debug.development.rttm \
    pyannote-audio/tests/data/debug.test.rttm > pilot/ds/ref.rttm

python3 -c "
uris = ['dev00','dev01','tst00','tst01'] + [f'trn0{i}' for i in range(10)]
open('pilot/ds/all.uem','w').write('\n'.join(f'{u} 1 0.00 30.00' for u in uris)+'\n')
"
```

## Run

```bash
python3 pilot/run_pilot.py pyannote-audio/tests/data models pilot/out
python3 pilot/score_pilot.py pilot/ds/dscore
```

Runtime: ~3 min for run_pilot, ~30 s for score_pilot (bootstrap B=10,000).

## Known gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `Protobuf parsing failed` | Segmentation model downloaded from HuggingFace (29-byte redirect stub) | Use the GitHub releases tar archive |
| 9-byte embedding .onnx files | Wrong release tag `speaker-recog-models` | Correct tag: `speaker-recongition-models` (typo is real) |
| `model.onnx` lands in `models/` not the subfolder | `--strip-components=1` drops the directory | `mv models/model.onnx models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` |
| `ModuleNotFoundError: intervaltree` | Not pulled in transitively | `pip install intervaltree` |
| `NoneType has no attribute 'group'` in score_pilot | Relative DSCORE path resolved against wrong cwd | Fixed in this repo (abspath patch) |
| "No speakers found" for trn01 | Clip is near-silence | Expected — dscore also warns "File trn01 missing", also expected |
| zsh rejects `#` in pasted multiline blocks | zsh interactive shell treats `#` as comment only in scripts | Use `python3 -c "..."` for one-liners instead of heredoc |
