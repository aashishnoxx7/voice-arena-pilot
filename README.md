# Voice Arena — Speaker Diarization Evaluation

**Can we replace an existing speaker diarization system without making meeting intelligence worse?**

Voice Arena is a speaker diarization evaluation project designed to help product managers compare existing, newer, cheaper, and self-hosted diarization systems under consistent evaluation conditions.

The project combines a reproducible pilot experiment with a proposed benchmark framework for evaluating how accurately systems identify **who spoke when** in recorded meetings.

> **Project status:** Pilot pipeline and benchmark design. The proposed 72-meeting benchmark is a design target, not a completed evaluation dataset. All system names and results in the leaderboard mockup are illustrative placeholders, not benchmark results.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Benchmark Design](#benchmark-design)
- [Evaluation Tracks](#evaluation-tracks)
- [Evaluation Metrics](#evaluation-metrics)
- [Pilot Experiment](#pilot-experiment)
- [Pilot Results](#pilot-results)
- [Fairness and Reproducibility](#fairness-and-reproducibility)
- [Evaluation Workflow](#evaluation-workflow)
- [Repository Structure](#repository-structure)
- [Reproducing the Pilot](#reproducing-the-pilot)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Supporting Documents](#supporting-documents)

---

## Problem Statement

Speaker diarization is the task of determining which speaker is speaking at each point in an audio recording.

Comparing diarization systems using published scores is difficult because evaluations may differ in:

- Audio datasets and recording environments.
- Scoring collars and treatment of overlapping speech.
- Whether the true number of speakers is supplied.
- Model versions, parameters, and preprocessing.
- Training-data overlap with evaluation datasets.
- Deployment constraints, speaker limits, and licensing.

Consequently, a lower published Diarization Error Rate (DER) does not necessarily mean that one system is better for a particular production environment.

Voice Arena addresses this problem by proposing a shared evaluation methodology that supports a practical decision:

**Should a product retain its current diarization system or replace it with an alternative?**

The objective is not simply to identify the system with the lowest error rate. It is to establish whether a candidate is meaningfully better, worse, acceptably non-inferior, or still inconclusive compared with the existing system.

## Project Objectives

1. Design a fair, reproducible evaluation framework for meeting speaker diarization.
2. Compare systems using the same recordings, scoring configuration, and evaluation rules.
3. Measure missed speech, false alarms, and speaker confusion separately.
4. Evaluate difficult conditions, including overlapping speech, hybrid meetings, and meetings with six to eight speakers.
5. Quantify uncertainty rather than relying on a single leaderboard rank.
6. Make model versions, deployment constraints, speaker limits, and licensing visible.
7. Validate the scoring pipeline through a small pilot experiment.

---

## Benchmark Design

### Proposed Version 1

The proposed benchmark focuses on batch diarization of recorded meetings using a single mixed, mono audio track.

| Component | Proposed design |
|---|---|
| Independent meetings | 72 |
| Total scored audio | 18 hours |
| Scored window | 15 minutes per meeting |
| Meeting environments | Remote, boardroom, hybrid |
| Meetings per environment | 24 |
| Speaker-count range | 2–8 speakers |
| Primary metric | Pooled DER |
| Scoring collar | 0 seconds |
| Overlapping speech | Scored |
| Primary evaluation | Speaker count not supplied |
| Reference annotations | Two independent annotators, followed by adjudication |

The 72-meeting test set is a proposed design. The pilot experiment described in this repository uses a much smaller collection of audio excerpts.

### Meeting Environments

**Fully remote meetings**

Participants connect through separate devices, introducing variation in microphones, codecs, and audio quality.

**Boardroom meetings**

Multiple participants share a room, creating challenges from reverberation, distance, cross-talk, and far-field microphones.

**Hybrid meetings**

Some participants attend in person while others join remotely. Echo and the recapture of remote voices can complicate speaker attribution.

### Dataset and Annotation Strategy

The proposed benchmark uses private, consented test recordings, with public development data intended for integration testing.

Each test meeting is planned to have:

- A reference annotation in RTTM format.
- A UEM file defining the scored region.
- Recording metadata and checksums.
- Speaker-count and meeting-environment labels.
- Annotation quality checks and adjudication records.

The test recordings and reference annotations are intended to remain private to reduce direct tuning against the evaluation set.

---

## Evaluation Tracks

Voice Arena separates evaluation conditions that provide systems with different amounts of information.

| Track | Description | Purpose |
|---|---|---|
| **BMB — Batch, Mixed Mono, Blind** | The true speaker count is not supplied. | Primary ranking that reflects the requirement to estimate speaker count. |
| **BMK — Batch, Mixed Mono, Known Count** | The true speaker count is supplied. | Separate ranking to measure performance when speaker count is known. |
| **Matched-Capacity View** | Systems are compared on meetings within their supported speaker capacity. | Diagnostic comparison that accounts for system limitations. |

Results from BMB and BMK must not be combined into a single ranking because the systems are operating with different information.

The primary benchmark also proposes three critical evaluation slices:

- Overlapping speech.
- Hybrid meetings.
- Meetings containing six to eight speakers.

These slices help reveal whether an improvement in overall quality comes at the expense of performance in difficult conditions.

---

## Evaluation Metrics

### Diarization Error Rate (DER)

DER is the primary evaluation metric.

\[
DER = \frac{MISS + FA + CONF}{TOTAL} \times 100\%
\]

Where:

- **MISS — Missed Speech:** Reference speaker-time that the system fails to detect.
- **FA — False Alarm:** Predicted speaker-time that does not correspond to reference speech, including excess simultaneous speakers.
- **CONF — Speaker Confusion:** Speech attributed to the wrong speaker.
- **TOTAL — Reference Speaker-Time:** Total reference speaking time in the scored region, counting each active speaker during overlapping speech.

The proposed headline configuration uses a zero-second scoring collar and includes overlapping speech.

### Supporting Metrics

**Jaccard Error Rate (JER)**

Provides an additional view of diarization quality that considers each reference speaker separately.

**Speaker-Count Error**

Measures whether the system predicts the correct number of speakers.

**Error Components**

MISS, FA, and CONF are reported independently to help identify whether a system primarily struggles with speech detection or speaker assignment.

**Confidence Intervals**

Paired comparisons use the same meetings for both systems. Uncertainty is estimated by resampling entire meetings rather than individual audio frames.

### Decision Framework

The proposed leaderboard distinguishes four outcomes:

| Outcome | Interpretation |
|---|---|
| Quality Winner | Evidence of a meaningful overall improvement without an unacceptable regression on critical slices. |
| Trade-off | Better overall quality, but a meaningful regression on a critical slice. |
| Unresolved | Available evidence cannot reliably determine which system is better. |
| Insufficient Evidence | Evaluation failures, data-quality issues, or inadequate coverage prevent a reliable decision. |

An unresolved result may still support a switch if the confidence interval rules out a degradation larger than the proposed non-inferiority margin.

The design uses illustrative policy thresholds of 1 percentage point for meaningful improvement, 1 point for the non-inferiority margin, and 2 points for critical-slice regression. These thresholds require validation before a public benchmark launch.

---

## Pilot Experiment

The pilot was conducted on **18 October 2026** to test the evaluation workflow end to end.

The experiment used:

- 14 single-channel audio excerpts.
- 30-second excerpts with reference annotations.
- Approximately seven inferred source meetings.
- The `sherpa-onnx` 1.13.8 diarization pipeline.
- Three speaker-embedding configurations.
- One additional run with the true speaker count supplied.
- `dscore` for primary scoring.
- `pyannote.metrics` 4.2.0 as an independent cross-check.

The excerpts were derived from AMI-related test fixtures in the `pyannote-audio` repository. Their original source meetings are not fully identified by the fixture metadata.

### Pilot Results

| Configuration | Pooled DER | MISS | FA | CONF |
|---|---:|---:|---:|---:|
| WeSpeaker ResNet34 — blind count | 37.81% | 22.60% | 2.56% | 12.64% |
| NeMo TitaNet-small — blind count | 38.45% | 22.59% | 2.35% | 13.51% |
| 3D-Speaker CAM++ — blind count | 45.68% | 22.52% | 3.38% | 19.77% |
| WeSpeaker ResNet34 — known count | 34.42% | 22.71% | 2.65% | 9.06% |

*Component percentages may not sum exactly to DER because of rounding.*

The two scoring implementations agreed to two decimal places on the pooled DER values for all four configurations.

### Key Observations

**1. The scoring pipeline is reproducible.**

The primary scorer and independent cross-check produced matching pooled DER values for the configurations tested.

**2. A small average difference does not establish a winner.**

WeSpeaker and TitaNet differed by approximately 0.65 DER percentage points. However, the paired uncertainty interval was wide, and their relative performance varied across individual clips.

**3. Speaker-count information changes the evaluation task.**

Providing the true speaker count reduced speaker confusion for the WeSpeaker configuration. This supports evaluating known-count systems separately from blind-count systems.

**4. Embedding changes did not eliminate shared segmentation errors.**

All tested configurations shared a segmentation model, so missed speech remained similar. Changing only the embedding model did not address the limitations shared by those configurations.

**5. Scoring choices affect comparisons.**

Changes to collar and overlap settings materially changed the difference between systems, reinforcing the need for a fixed, documented scoring configuration.

### Interpretation

The experiment validates parts of the scoring workflow and provides hypotheses for further investigation. It does **not** establish a general ranking of diarization systems or demonstrate production performance.

The sample is too small to support broad claims about meeting types, languages, accents, or deployment conditions.

---

## Fairness and Reproducibility

The proposed evaluation methodology includes the following safeguards:

- **Same recordings:** Every eligible system receives the same meeting audio within an evaluation track.
- **Consistent audio:** Canonical mono, 16 kHz PCM audio is prepared once, with transformations and checksums recorded.
- **Documented configurations:** Model versions, parameters, package versions, hardware, and timestamps are captured.
- **Separate information conditions:** Blind-count and known-count results are reported independently.
- **No test-set tuning:** Configuration choices are made using development data rather than held-out test results.
- **No silent exclusions:** System failures and unsupported inputs are documented.
- **Consistent scoring:** A pinned scorer and fixed configuration are used for all comparable outputs.
- **Independent validation:** Scoring results are cross-checked, and discrepancies are investigated.
- **Visible constraints:** Licensing, deployment type, speaker limits, and input restrictions appear alongside results.
- **Statistical uncertainty:** Paired confidence intervals and rank tiers are used instead of presenting every rank as definitive.

These measures aim to distinguish genuine quality differences from changes caused by datasets, configuration, preprocessing, or evaluation methodology.

---

## Evaluation Workflow

The proposed full benchmark pipeline follows these stages:

1. Freeze the benchmark version, reference annotations, scorer, and decision policy.
2. Prepare canonical audio and record transformations and checksums.
3. Verify system eligibility, licensing, speaker limits, and input requirements.
4. Run eligible systems under documented configurations.
5. Preserve raw system outputs.
6. Convert predictions into a common RTTM representation where necessary.
7. Validate outputs and score DER and its component errors.
8. Calculate critical-slice metrics and paired confidence intervals.
9. Apply the predefined decision framework.
10. Publish the leaderboard, metadata, scorer configuration, run records, and limitations.

The current pilot implements only a small experimental version of this workflow; the full benchmark pipeline remains a proposed design.

---

## Repository Structure

The main pilot entry points and supporting directories are:

```text
voice-arena-pilot/
├── models/
├── pilot/
│   ├── run_pilot.py
│   └── score_pilot.py
├── pyannote-audio/
├── .gitignore
└── README.md
```

The exact contents may vary with the current checkout. Supporting assessment documents can be stored separately in a `docs/` directory.

---

## Reproducing the Pilot

The reported pilot used the following configuration:

| Component | Configuration |
|---|---|
| Diarization library | `sherpa-onnx` 1.13.8 |
| Segmentation model | `sherpa-onnx-pyannote-segmentation-3-0` ONNX model |
| Embedding models | WeSpeaker ResNet34, NeMo TitaNet-small, 3D-Speaker CAM++ |
| Clustering | Fast clustering with cosine threshold 0.5 |
| Minimum speech duration | 0.3 seconds |
| Minimum silence duration | 0.5 seconds |
| Primary scorer | `dscore`, commit `e02f949` |
| Cross-check scorer | `pyannote.metrics` 4.2.0 |
| Bootstrap | 10,000 replicates, seed 7 |

Run the pilot scripts from the repository root in the project's configured Python environment:

```bash
python pilot/run_pilot.py
python pilot/score_pilot.py
```

These commands assume that the required dependencies, models, fixtures, and configuration are installed and available at the paths expected by the scripts.

For reproduction, retain the documented package versions, model configurations, scoring collar, overlap policy, and input fixtures. Changes to these settings may change the reported results.

The pilot does not reproduce or replace the proposed 72-meeting private benchmark.

---

## Limitations

The project has several important limitations:

1. **Small pilot dataset:** Fourteen short excerpts are not enough to establish reliable system rankings across diverse meetings.
2. **Source uncertainty:** The excerpts come from AMI-related fixtures, with incomplete source-meeting identification.
3. **Limited duration:** The pilot does not establish how systems behave across long meetings or extended speaker-identity tracking.
4. **Potential training overlap:** Public evaluation material may overlap with model training data.
5. **Proposed benchmark size:** The 72-meeting dataset is a design target, not a completed dataset.
6. **Uncalibrated decision thresholds:** The proposed improvement and non-inferiority margins require validation with development-set data and product managers.
7. **Limited language evidence:** Hindi, Hinglish, and Indian-accented English coverage depends on suitable recordings and reliable reference annotations.
8. **Deployment scope:** Streaming diarization, telephony, separate participant channels, transcription metrics, latency, and operational cost are outside the initial benchmark scope.

The leaderboard mockup is a product-design artifact. Its illustrative scores and placeholder system names must not be cited as measured benchmark results.

---

## Future Work

The next stages of the project are:

- Validate the proposed decision thresholds with product managers.
- Collect development-set data and estimate meeting-level variance.
- Test statistical coverage and reliability of the paired bootstrap.
- Validate annotation guidelines and adjudication procedures.
- Expand evaluation coverage for Indian English, Hindi, and Hinglish where suitable data is available.
- Validate converters for systems that return word-level or utterance-level speaker labels.
- Evaluate a broader inventory of candidate diarization systems.
- Build and usability-test the interactive leaderboard.
- Explore separate tracks for streaming, telephony, multichannel audio, and speaker-attributed transcription.

---

## Supporting Documents

The project is accompanied by three assessment documents:

| Document | Description |
|---|---|
| **Q1 Main Report** | Detailed benchmark design, system inventory, dataset proposal, annotation workflow, metrics, decision rules, pilot experiment, and references. |
| **Q1 One-Pager** | Concise summary of the problem, evaluation safeguards, benchmark proposal, pilot results, and limitations. |
| **Q1 Leaderboard Sketch** | Visual concept for the leaderboard, model-detail panel, critical slices, comparison outcomes, and empty states. |

Place these documents in a `docs/` directory if they are included in the repository. The leaderboard sketch is explicitly illustrative and is not evidence of a completed benchmark.

---

## Conclusion

Voice Arena is built around a decision rather than a ranking alone: **can a team switch diarization systems without materially reducing quality on the meetings that matter to its product?**

The pilot demonstrates a small, reproducible scoring workflow and illustrates why speaker-count assumptions, overlap handling, and uncertainty matter. The proposed benchmark extends that approach with controlled comparisons, critical-slice analysis, and explicit decision outcomes.

The project's central principle is that a leaderboard should communicate not only which system scores best, but also **how reliable that conclusion is, where a system struggles, and what the evidence does not yet establish.**
