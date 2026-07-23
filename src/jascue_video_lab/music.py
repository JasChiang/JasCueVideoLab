from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from array import array
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .media import sha256_file
from .models import FrozenStrictModel, StrictModel
from .storage import utc_now


MUSIC_ANALYSIS_VERSION = "local-music-analysis-v1"
MUSIC_MAP_CONTRACT_VERSION = "music-map-proposal-v1"
MUSIC_LOCK_CONTRACT_VERSION = "music-map-lock-v1"


class MusicAnalysisError(RuntimeError):
    pass


class CuePriority(StrEnum):
    HARD = "hard"
    PREFERRED = "preferred"
    OPTIONAL = "optional"


class MusicAnalysisParameters(FrozenStrictModel):
    analysis_version: Literal["local-music-analysis-v1"] = MUSIC_ANALYSIS_VERSION
    master_sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    analysis_sample_rate: int = Field(default=12_000, ge=8_000, le=48_000)
    window_samples: int = Field(default=1024, ge=128, le=8192)
    hop_samples: int = Field(default=256, ge=64, le=4096)
    min_bpm: float = Field(default=60.0, ge=30.0, le=240.0)
    max_bpm: float = Field(default=180.0, ge=40.0, le=300.0)
    section_min_duration_ms: int = Field(default=4_000, ge=1_000, le=60_000)
    energy_interval_ms: int = Field(default=250, ge=50, le=2_000)

    @model_validator(mode="after")
    def validate_ranges(self) -> "MusicAnalysisParameters":
        if self.min_bpm >= self.max_bpm:
            raise ValueError("min_bpm must be lower than max_bpm")
        if self.hop_samples > self.window_samples:
            raise ValueError("hop_samples must not exceed window_samples")
        return self


class MusicEnergyPoint(FrozenStrictModel):
    sample_index: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    energy: float = Field(ge=0.0, le=1.0)
    onset_strength: float = Field(ge=0.0, le=1.0)


class MusicCueCandidate(FrozenStrictModel):
    cue_id: str = Field(pattern=r"^mc-[0-9]{5}$")
    kind: Literal[
        "beat_candidate",
        "accent_candidate",
        "ending_hit_candidate",
    ]
    sample_index: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["local_pcm_analysis"] = "local_pcm_analysis"


class MusicSectionCandidate(FrozenStrictModel):
    section_id: str = Field(pattern=r"^section-[0-9]{3}$")
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    label: str = Field(pattern=r"^section_[0-9]{3}$")
    boundary_source: Literal["energy_change", "whole_track"]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> "MusicSectionCandidate":
        if self.end_sample <= self.start_sample:
            raise ValueError("music section must be a non-empty half-open interval")
        return self


class MusicMapProposal(StrictModel):
    contract_version: Literal["music-map-proposal-v1"] = MUSIC_MAP_CONTRACT_VERSION
    music_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    master_sample_rate: int = Field(ge=8_000, le=192_000)
    duration_samples: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    analysis_parameters: MusicAnalysisParameters
    estimated_bpm: float | None = Field(default=None, ge=30.0, le=300.0)
    tempo_confidence: float = Field(ge=0.0, le=1.0)
    meter_suggestion: int | None = Field(default=4, ge=2, le=12)
    first_beat_sample: int | None = Field(default=None, ge=0)
    cues: list[MusicCueCandidate]
    sections: list[MusicSectionCandidate]
    energy_curve: list[MusicEnergyPoint]
    uncertainties: list[str]
    requires_human_review: Literal[True] = True
    generated_at: str

    @model_validator(mode="after")
    def validate_timeline(self) -> "MusicMapProposal":
        if self.analysis_parameters.master_sample_rate != self.master_sample_rate:
            raise ValueError("analysis parameters and proposal sample rates differ")
        if self.first_beat_sample is not None and self.first_beat_sample >= self.duration_samples:
            raise ValueError("first beat must remain inside the music timeline")
        cue_samples = [cue.sample_index for cue in self.cues]
        if cue_samples != sorted(cue_samples):
            raise ValueError("music cues must be chronological")
        if any(sample >= self.duration_samples for sample in cue_samples):
            raise ValueError("music cue lies outside the music timeline")
        if not self.sections:
            raise ValueError("music proposal requires at least one section")
        if self.sections[0].start_sample != 0:
            raise ValueError("music sections must begin at sample zero")
        if self.sections[-1].end_sample != self.duration_samples:
            raise ValueError("music sections must cover the full track")
        for left, right in zip(self.sections, self.sections[1:]):
            if left.end_sample != right.start_sample:
                raise ValueError("music sections must be contiguous")
        return self


class MusicMapReview(StrictModel):
    contract_version: Literal["music-map-review-v1"] = "music-map-review-v1"
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    reviewed_at: str
    decision: Literal["approved", "rejected"]
    notes: str = ""
    bpm: float | None = Field(default=None, ge=30.0, le=300.0)
    first_downbeat_sample: int | None = Field(default=None, ge=0)
    meter: int | None = Field(default=None, ge=2, le=12)

    @model_validator(mode="after")
    def validate_approval(self) -> "MusicMapReview":
        values = (self.bpm, self.first_downbeat_sample, self.meter)
        if self.decision == "approved" and any(value is None for value in values):
            raise ValueError(
                "approved music review requires BPM, first downbeat, and meter"
            )
        if self.decision == "rejected" and any(value is not None for value in values):
            raise ValueError("rejected music review cannot create an approved beat grid")
        return self


class LockedMusicCue(FrozenStrictModel):
    cue_id: str = Field(pattern=r"^locked-cue-[0-9]{5}$")
    kind: Literal[
        "section_boundary",
        "downbeat",
        "beat",
        "accent",
        "ending_hit",
    ]
    sample_index: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    strength: float = Field(ge=0.0, le=1.0)
    priority: CuePriority
    source_candidate_ids: tuple[str, ...] = ()


class MusicMapLock(StrictModel):
    contract_version: Literal["music-map-lock-v1"] = MUSIC_LOCK_CONTRACT_VERSION
    music_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proposal_path: str
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review: MusicMapReview
    master_sample_rate: int = Field(ge=8_000, le=192_000)
    duration_samples: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    bpm: float = Field(ge=30.0, le=300.0)
    meter: int = Field(ge=2, le=12)
    first_downbeat_sample: int = Field(ge=0)
    cues: list[LockedMusicCue]
    sections: list[MusicSectionCandidate]
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lock(self) -> "MusicMapLock":
        if self.review.decision != "approved":
            raise ValueError("music lock requires an approved human review")
        if self.review.proposal_sha256 != self.proposal_sha256:
            raise ValueError("music review is not bound to this proposal")
        if self.first_downbeat_sample >= self.duration_samples:
            raise ValueError("first downbeat must remain inside the music timeline")
        samples = [cue.sample_index for cue in self.cues]
        if samples != sorted(samples):
            raise ValueError("locked music cues must be chronological")
        return self


def _canonical_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude={"definition_sha256"})
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_to_ms(sample_index: int, sample_rate: int) -> int:
    return round(Fraction(sample_index * 1000, sample_rate))


def _analysis_to_master_sample(
    sample_index: int, analysis_rate: int, master_rate: int
) -> int:
    return round(Fraction(sample_index * master_rate, analysis_rate))


def _decode_pcm(source: Path, sample_rate: int) -> array:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MusicAnalysisError(f"FFmpeg could not decode an audio stream: {message}")
    samples = array("h")
    samples.frombytes(completed.stdout)
    if not samples:
        raise MusicAnalysisError("decoded audio stream is empty")
    return samples


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def _normalize(values: list[float]) -> list[float]:
    ceiling = _percentile(values, 0.95)
    if ceiling <= 1e-12:
        return [0.0 for _ in values]
    return [min(1.0, max(0.0, value / ceiling)) for value in values]


def _frame_features(
    samples: array, parameters: MusicAnalysisParameters
) -> tuple[list[int], list[float], list[float]]:
    positions: list[int] = []
    energies: list[float] = []
    high_frequency: list[float] = []
    window = parameters.window_samples
    hop = parameters.hop_samples
    for start in range(0, max(1, len(samples) - window + 1), hop):
        frame = samples[start : start + window]
        if not frame:
            continue
        energy = math.sqrt(sum(float(value) * value for value in frame) / len(frame))
        difference = sum(
            abs(int(frame[index]) - int(frame[index - 1]))
            for index in range(1, len(frame))
        ) / max(1, len(frame) - 1)
        positions.append(start + len(frame) // 2)
        energies.append(energy)
        high_frequency.append(difference)
    normalized_energy = _normalize(energies)
    normalized_hf = _normalize(high_frequency)
    raw_onsets: list[float] = []
    previous_energy = 0.0
    previous_hf = 0.0
    for energy, frequency in zip(normalized_energy, normalized_hf, strict=True):
        raw_onsets.append(
            max(0.0, energy - previous_energy)
            + 0.75 * max(0.0, frequency - previous_hf)
        )
        previous_energy = energy
        previous_hf = frequency
    return positions, normalized_energy, _normalize(raw_onsets)


def _pick_onset_peaks(
    positions: list[int],
    strengths: list[float],
    sample_rate: int,
) -> list[tuple[int, float]]:
    minimum_gap = max(1, round(sample_rate * 0.08))
    peaks: list[tuple[int, float]] = []
    for index in range(1, max(1, len(strengths) - 1)):
        strength = strengths[index]
        local_start = max(0, index - 12)
        baseline = statistics.median(strengths[local_start : index + 1])
        if (
            strength < 0.12
            or strength < baseline + 0.06
            or strength < strengths[index - 1]
            or strength < strengths[index + 1]
        ):
            continue
        sample = positions[index]
        if peaks and sample - peaks[-1][0] < minimum_gap:
            if strength > peaks[-1][1]:
                peaks[-1] = (sample, strength)
            continue
        peaks.append((sample, strength))
    return peaks


def _estimate_tempo(
    peaks: list[tuple[int, float]],
    sample_rate: int,
    minimum_bpm: float,
    maximum_bpm: float,
) -> tuple[float | None, float]:
    if len(peaks) < 3:
        return None, 0.0
    bins: dict[float, float] = {}
    for left_index, (left_sample, left_strength) in enumerate(peaks):
        for gap in range(1, 5):
            right_index = left_index + gap
            if right_index >= len(peaks):
                break
            right_sample, right_strength = peaks[right_index]
            interval = right_sample - left_sample
            if interval <= 0:
                continue
            bpm = 60.0 * sample_rate / interval
            while bpm < minimum_bpm:
                bpm *= 2.0
            while bpm > maximum_bpm:
                bpm /= 2.0
            if not minimum_bpm <= bpm <= maximum_bpm:
                continue
            bucket = round(bpm * 2.0) / 2.0
            bins[bucket] = bins.get(bucket, 0.0) + (
                left_strength * right_strength / gap
            )
    if not bins:
        return None, 0.0
    ranked = sorted(bins.items(), key=lambda item: (-item[1], item[0]))
    bpm, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(bins.values())
    confidence = min(
        1.0,
        0.45 * best / max(total, 1e-9)
        + 0.55 * (best - runner_up) / max(best, 1e-9),
    )
    return bpm, max(0.0, confidence)


def _choose_first_beat(
    peaks: list[tuple[int, float]], sample_rate: int, bpm: float
) -> int | None:
    if not peaks:
        return None
    period = sample_rate * 60.0 / bpm
    tolerance = max(1.0, period * 0.12)
    candidates = sorted(peaks, key=lambda item: (-item[1], item[0]))[:24]
    best: tuple[float, int] | None = None
    for candidate, _ in candidates:
        score = 0.0
        for sample, strength in peaks:
            phase = abs(((sample - candidate + period / 2) % period) - period / 2)
            score += strength * math.exp(-((phase / tolerance) ** 2))
        normalized_candidate = int(candidate % period)
        result = (score, -normalized_candidate)
        if best is None or result > best:
            best = result
    assert best is not None
    return -best[1]


def _nearest_peak_strength(
    sample: int, peaks: list[tuple[int, float]], tolerance: int
) -> tuple[float, tuple[int, float] | None]:
    nearby = [
        peak for peak in peaks if abs(peak[0] - sample) <= max(1, tolerance)
    ]
    if not nearby:
        return 0.0, None
    selected = max(nearby, key=lambda item: (item[1], -abs(item[0] - sample)))
    return selected[1], selected


def _build_energy_curve(
    positions: list[int],
    energy: list[float],
    onsets: list[float],
    parameters: MusicAnalysisParameters,
) -> list[MusicEnergyPoint]:
    bucket_samples = max(
        1, round(parameters.analysis_sample_rate * parameters.energy_interval_ms / 1000)
    )
    buckets: dict[int, tuple[list[float], list[float]]] = {}
    for sample, energy_value, onset in zip(positions, energy, onsets, strict=True):
        bucket = sample // bucket_samples
        energy_values, onset_values = buckets.setdefault(bucket, ([], []))
        energy_values.append(energy_value)
        onset_values.append(onset)
    result: list[MusicEnergyPoint] = []
    for bucket, (energy_values, onset_values) in sorted(buckets.items()):
        analysis_sample = bucket * bucket_samples
        master_sample = _analysis_to_master_sample(
            analysis_sample,
            parameters.analysis_sample_rate,
            parameters.master_sample_rate,
        )
        result.append(
            MusicEnergyPoint(
                sample_index=master_sample,
                time_ms=_sample_to_ms(master_sample, parameters.master_sample_rate),
                energy=round(statistics.fmean(energy_values), 6),
                onset_strength=round(max(onset_values), 6),
            )
        )
    return result


def _section_boundaries(
    curve: list[MusicEnergyPoint],
    duration_samples: int,
    parameters: MusicAnalysisParameters,
) -> list[int]:
    if len(curve) < 8:
        return [0, duration_samples]
    changes: list[tuple[float, int]] = []
    radius = max(2, round(2_000 / parameters.energy_interval_ms))
    for index in range(radius, len(curve) - radius):
        before = statistics.fmean(
            point.energy for point in curve[index - radius : index]
        )
        after = statistics.fmean(
            point.energy for point in curve[index : index + radius]
        )
        changes.append((abs(after - before), curve[index].sample_index))
    threshold = max(0.18, _percentile([item[0] for item in changes], 0.85))
    minimum_gap = round(
        parameters.master_sample_rate * parameters.section_min_duration_ms / 1000
    )
    selected = [0]
    for change, sample in sorted(changes, key=lambda item: (-item[0], item[1])):
        if change < threshold:
            continue
        if sample < minimum_gap or duration_samples - sample < minimum_gap:
            continue
        if all(abs(sample - boundary) >= minimum_gap for boundary in selected):
            selected.append(sample)
        if len(selected) >= 8:
            break
    return sorted([*selected, duration_samples])


def analyze_music(
    source: Path,
    *,
    parameters: MusicAnalysisParameters | None = None,
) -> MusicMapProposal:
    resolved = source.expanduser().resolve(strict=True)
    config = parameters or MusicAnalysisParameters()
    samples = _decode_pcm(resolved, config.analysis_sample_rate)
    duration_samples = _analysis_to_master_sample(
        len(samples), config.analysis_sample_rate, config.master_sample_rate
    )
    positions, energy, onsets = _frame_features(samples, config)
    peaks = _pick_onset_peaks(positions, onsets, config.analysis_sample_rate)
    bpm, tempo_confidence = _estimate_tempo(
        peaks, config.analysis_sample_rate, config.min_bpm, config.max_bpm
    )
    first_beat_analysis = (
        _choose_first_beat(peaks, config.analysis_sample_rate, bpm)
        if bpm is not None
        else None
    )
    first_beat_master = (
        _analysis_to_master_sample(
            first_beat_analysis, config.analysis_sample_rate, config.master_sample_rate
        )
        if first_beat_analysis is not None
        else None
    )

    cue_rows: list[tuple[str, int, float, float]] = []
    if bpm is not None and first_beat_analysis is not None:
        period = config.analysis_sample_rate * 60.0 / bpm
        index = 0
        while True:
            analysis_sample = round(first_beat_analysis + index * period)
            if analysis_sample >= len(samples):
                break
            strength, _ = _nearest_peak_strength(
                analysis_sample, peaks, round(period * 0.14)
            )
            cue_rows.append(
                (
                    "beat_candidate",
                    _analysis_to_master_sample(
                        analysis_sample,
                        config.analysis_sample_rate,
                        config.master_sample_rate,
                    ),
                    strength,
                    min(1.0, tempo_confidence * 0.7 + strength * 0.3),
                )
            )
            index += 1
    accent_peaks = [
        peak
        for peak in peaks
        if peak[1] >= max(0.45, _percentile([item[1] for item in peaks], 0.7))
    ]
    for analysis_sample, strength in accent_peaks:
        cue_rows.append(
            (
                "accent_candidate",
                _analysis_to_master_sample(
                    analysis_sample,
                    config.analysis_sample_rate,
                    config.master_sample_rate,
                ),
                strength,
                min(1.0, 0.45 + strength * 0.5),
            )
        )
    ending_candidates = [
        peak
        for peak in peaks
        if peak[0] >= len(samples) - config.analysis_sample_rate * 6
        and peak[1] >= 0.35
    ]
    if ending_candidates:
        analysis_sample, strength = ending_candidates[-1]
        cue_rows.append(
            (
                "ending_hit_candidate",
                _analysis_to_master_sample(
                    analysis_sample,
                    config.analysis_sample_rate,
                    config.master_sample_rate,
                ),
                strength,
                min(1.0, 0.35 + strength * 0.55),
            )
        )
    cue_rows.sort(key=lambda item: (item[1], item[0]))
    cues = [
        MusicCueCandidate(
            cue_id=f"mc-{index:05d}",
            kind=kind,
            sample_index=min(sample, duration_samples - 1),
            time_ms=_sample_to_ms(
                min(sample, duration_samples - 1), config.master_sample_rate
            ),
            strength=round(strength, 6),
            confidence=round(confidence, 6),
        )
        for index, (kind, sample, strength, confidence) in enumerate(cue_rows, start=1)
    ]
    energy_curve = _build_energy_curve(positions, energy, onsets, config)
    boundaries = _section_boundaries(energy_curve, duration_samples, config)
    sections = [
        MusicSectionCandidate(
            section_id=f"section-{index:03d}",
            start_sample=start,
            end_sample=end,
            label=f"section_{index:03d}",
            boundary_source=(
                "whole_track" if len(boundaries) == 2 else "energy_change"
            ),
            confidence=(0.5 if len(boundaries) == 2 else 0.6),
        )
        for index, (start, end) in enumerate(
            zip(boundaries, boundaries[1:]), start=1
        )
    ]
    uncertainties = [
        "Beat, downbeat, meter, and section labels are local signal-analysis proposals; human review is required before cue scheduling.",
        "The analyzer does not infer musical semantics, lyrics, or editorial intent.",
    ]
    if bpm is None:
        uncertainties.append(
            "No stable tempo grid could be inferred; review must supply a BPM and first downbeat."
        )
    elif tempo_confidence < 0.45:
        uncertainties.append(
            "Tempo confidence is low; half-time, double-time, or free-tempo interpretation may be present."
        )
    digest = sha256_file(resolved)
    return MusicMapProposal(
        music_id=f"sha256:{digest}",
        source_sha256=digest,
        master_sample_rate=config.master_sample_rate,
        duration_samples=duration_samples,
        duration_ms=_sample_to_ms(duration_samples, config.master_sample_rate),
        analysis_parameters=config,
        estimated_bpm=bpm,
        tempo_confidence=round(tempo_confidence, 6),
        meter_suggestion=4,
        first_beat_sample=first_beat_master,
        cues=cues,
        sections=sections,
        energy_curve=energy_curve,
        uncertainties=uncertainties,
        generated_at=utc_now(),
    )


def _locked_cues(
    proposal: MusicMapProposal,
    *,
    bpm: float,
    first_downbeat_sample: int,
    meter: int,
) -> list[LockedMusicCue]:
    beat_period = Fraction(
        proposal.master_sample_rate * 60_000,
        round(bpm * 1000),
    )
    candidates = proposal.cues
    raw: list[
        tuple[int, str, float, CuePriority, tuple[str, ...]]
    ] = []
    beat_index = 0
    while True:
        sample = first_downbeat_sample + round(beat_period * beat_index)
        if sample >= proposal.duration_samples:
            break
        tolerance = max(1, round(float(beat_period) * 0.14))
        nearby = [
            cue
            for cue in candidates
            if abs(cue.sample_index - sample) <= tolerance
        ]
        strength = max((cue.strength for cue in nearby), default=0.0)
        raw.append(
            (
                sample,
                "downbeat" if beat_index % meter == 0 else "beat",
                strength,
                (
                    CuePriority.PREFERRED
                    if beat_index % meter == 0
                    else CuePriority.OPTIONAL
                ),
                tuple(cue.cue_id for cue in nearby),
            )
        )
        beat_index += 1
    for cue in candidates:
        if cue.kind == "accent_candidate":
            raw.append(
                (
                    cue.sample_index,
                    "accent",
                    cue.strength,
                    CuePriority.PREFERRED,
                    (cue.cue_id,),
                )
            )
        elif cue.kind == "ending_hit_candidate":
            raw.append(
                (
                    cue.sample_index,
                    "ending_hit",
                    cue.strength,
                    CuePriority.HARD,
                    (cue.cue_id,),
                )
            )
    for section in proposal.sections[1:]:
        raw.append(
            (
                section.start_sample,
                "section_boundary",
                section.confidence,
                CuePriority.HARD,
                (),
            )
        )
    priority_rank = {
        CuePriority.OPTIONAL: 0,
        CuePriority.PREFERRED: 1,
        CuePriority.HARD: 2,
    }
    kind_rank = {
        "beat": 0,
        "accent": 1,
        "downbeat": 2,
        "section_boundary": 3,
        "ending_hit": 4,
    }
    # Windowed onset analysis places a transient near the centre of its
    # analysis window, while the reviewed beat grid may place the same event a
    # few milliseconds earlier. Treat cues within 60 ms as one musical event so
    # the scheduler cannot spend both an "accent" and a "beat" on the same hit.
    merge_tolerance = round(proposal.master_sample_rate * 0.06)
    grouped: list[
        list[tuple[int, str, float, CuePriority, tuple[str, ...]]]
    ] = []
    for row in sorted(raw, key=lambda item: (item[0], item[1])):
        if grouped and row[0] - grouped[-1][-1][0] <= merge_tolerance:
            grouped[-1].append(row)
        else:
            grouped.append([row])
    deduplicated: list[
        tuple[int, str, float, CuePriority, tuple[str, ...]]
    ] = []
    for rows in grouped:
        selected = max(
            rows,
            key=lambda row: (
                priority_rank[row[3]],
                kind_rank[row[1]],
                row[2],
            ),
        )
        deduplicated.append(
            (
                selected[0],
                selected[1],
                max(row[2] for row in rows),
                max((row[3] for row in rows), key=priority_rank.__getitem__),
                tuple(
                    dict.fromkeys(
                        source_id
                        for row in rows
                        for source_id in row[4]
                    )
                ),
            )
        )
    deduplicated.sort(key=lambda item: (item[0], item[1]))
    return [
        LockedMusicCue(
            cue_id=f"locked-cue-{index:05d}",
            kind=kind,  # type: ignore[arg-type]
            sample_index=sample,
            time_ms=_sample_to_ms(sample, proposal.master_sample_rate),
            strength=round(strength, 6),
            priority=priority,
            source_candidate_ids=source_ids,
        )
        for index, (sample, kind, strength, priority, source_ids) in enumerate(
            deduplicated, start=1
        )
    ]


def review_music_map(
    proposal: MusicMapProposal,
    *,
    proposal_path: Path,
    reviewer: str,
    decision: Literal["approved", "rejected"],
    notes: str = "",
    bpm: float | None = None,
    first_downbeat_sample: int | None = None,
    meter: int | None = None,
) -> tuple[MusicMapReview, MusicMapLock | None]:
    resolved_proposal = proposal_path.expanduser().resolve(strict=True)
    proposal_sha256 = sha256_file(resolved_proposal)
    saved_proposal = MusicMapProposal.model_validate_json(
        resolved_proposal.read_text(encoding="utf-8")
    )
    if saved_proposal != proposal:
        raise ValueError("in-memory MusicMap proposal differs from the saved artifact")
    if decision == "approved":
        bpm = bpm if bpm is not None else proposal.estimated_bpm
        first_downbeat_sample = (
            first_downbeat_sample
            if first_downbeat_sample is not None
            else proposal.first_beat_sample
        )
        meter = meter if meter is not None else proposal.meter_suggestion
        if bpm is None or first_downbeat_sample is None or meter is None:
            raise ValueError(
                "approval needs explicit BPM, first downbeat, and meter when analysis could not infer them"
            )
        if first_downbeat_sample >= proposal.duration_samples:
            raise ValueError("reviewed first downbeat lies outside the music timeline")
    else:
        bpm = None
        first_downbeat_sample = None
        meter = None
    review = MusicMapReview(
        proposal_sha256=proposal_sha256,
        reviewer=reviewer,
        reviewed_at=utc_now(),
        decision=decision,
        notes=notes,
        bpm=bpm,
        first_downbeat_sample=first_downbeat_sample,
        meter=meter,
    )
    if decision == "rejected":
        return review, None
    assert bpm is not None and first_downbeat_sample is not None and meter is not None
    cues = _locked_cues(
        proposal,
        bpm=bpm,
        first_downbeat_sample=first_downbeat_sample,
        meter=meter,
    )
    definition = {
        "music_id": proposal.music_id,
        "proposal_sha256": proposal_sha256,
        "review": review.model_dump(mode="json"),
        "master_sample_rate": proposal.master_sample_rate,
        "duration_samples": proposal.duration_samples,
        "bpm": bpm,
        "meter": meter,
        "first_downbeat_sample": first_downbeat_sample,
        "cues": [cue.model_dump(mode="json") for cue in cues],
        "sections": [section.model_dump(mode="json") for section in proposal.sections],
    }
    lock = MusicMapLock(
        music_id=proposal.music_id,
        proposal_path=str(resolved_proposal),
        proposal_sha256=proposal_sha256,
        review=review,
        master_sample_rate=proposal.master_sample_rate,
        duration_samples=proposal.duration_samples,
        duration_ms=proposal.duration_ms,
        bpm=bpm,
        meter=meter,
        first_downbeat_sample=first_downbeat_sample,
        cues=cues,
        sections=proposal.sections,
        definition_sha256=_canonical_hash(definition),
    )
    return review, lock
