#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


SCENARIOS = {
    # Angles come from each Goal Force example CSV's target_indirect_force_angle.
    # Goal Force uses Cartesian angles; image y increases downward, so y is negated
    # when constructing the desired image-space direction vector.
    #
    # mode:
    #   collision        -> two free objects with an impact/transfer event
    #   strike           -> actor/tool strikes a target in a discrete event
    #   direct_actuation -> actor starts in/near contact and continuously pushes/nudges
    #   single_target    -> only target motion is semantically available
    "ball_dominos": {"angle_deg": 5.0, "mode": "collision"},
    "pendulum": {"angle_deg": 0.0, "mode": "strike"},
    "pool": {"angle_deg": 180.0, "mode": "collision"},
    "bulb": {"angle_deg": 0.0, "mode": "direct_actuation"},
    "cantaloupes": {"angle_deg": 0.0, "mode": "collision"},
    "tennis": {"angle_deg": 175.0, "mode": "collision"},
    "golf": {"angle_deg": 185.0, "mode": "strike"},
    "toycar": {"angle_deg": 225.0, "mode": "single_target"},
    "paw_tool1": {"angle_deg": 335.0, "mode": "direct_actuation"},
    "paw_tool2": {"angle_deg": 135.0, "mode": "direct_actuation"},
    "soccer": {"angle_deg": 180.0, "mode": "strike"},
    "paw_tool3": {"angle_deg": 180.0, "mode": "direct_actuation"},
}

OUTPUT_FIELDS = [
    "scenario",
    "seed",
    "video",
    "interaction_mode",
    "object_count",
    "contact_metric_valid",
    "expected_angle_deg",
    "goal_completion",
    "goal_progress_diameters",
    "goal_progress_px",
    "direction_alignment",
    "contact_causality",
    "motion_stability",
    "flow_consistency",
    "track_visibility",
    "projectile_onset_frame",
    "target_onset_frame",
    "contact_frame",
    "minimum_gap_px",
    "target_diameter_px",
    "forward_motion_fraction",
    "reverse_motion_fraction",
    "precontact_motion_fraction",
    "postcontact_response",
    "collision_approach_score",
    "collision_transfer_alignment",
    "target_noise_threshold_px",
    "postcontact_displacement_diameters",
    "mask_area_change_robust",
    "trajectory_jerk_norm",
    "track_mask_disagreement_norm",
]


def parse_name(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_seed(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected feature filename: {path.name}")

    scenario = match.group(1)
    if scenario not in SCENARIOS:
        raise ValueError(f"No scenario configuration for: {scenario}")

    return scenario, int(match.group(2))


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    indices = np.arange(len(result))

    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        if not valid.any():
            raise ValueError("A trajectory contains no valid coordinates.")

        result[:, column] = np.interp(
            indices,
            indices[valid],
            result[valid, column],
        )

    return result


def mask_centroids(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count, object_count = masks.shape[:2]
    centroids = np.full((frame_count, object_count, 2), np.nan)
    areas = masks.sum(axis=(2, 3)).astype(np.float64)

    for frame_index in range(frame_count):
        for object_index in range(object_count):
            ys, xs = np.nonzero(masks[frame_index, object_index])
            if len(xs):
                centroids[frame_index, object_index] = [
                    xs.mean(),
                    ys.mean(),
                ]

    for object_index in range(object_count):
        centroids[:, object_index] = interpolate_nans(
            centroids[:, object_index]
        )

    return centroids, areas


def track_paths(
    tracks: np.ndarray,
    visibility: np.ndarray,
    object_ids: np.ndarray,
    initial_centroids: np.ndarray,
) -> np.ndarray:
    frame_count = tracks.shape[0]
    object_count = initial_centroids.shape[0]
    paths = np.zeros((frame_count, object_count, 2), dtype=np.float64)

    for object_index in range(object_count):
        point_indices = np.flatnonzero(object_ids == object_index)
        if not len(point_indices):
            raise ValueError(f"No CoTracker points for object {object_index}.")

        base_tracks = tracks[0, point_indices].astype(np.float64)
        base_visible = visibility[0, point_indices]

        for frame_index in range(frame_count):
            current = tracks[frame_index, point_indices].astype(np.float64)
            valid = (
                base_visible
                & visibility[frame_index, point_indices]
                & np.isfinite(current).all(axis=1)
                & np.isfinite(base_tracks).all(axis=1)
            )

            if valid.any():
                displacement = np.median(
                    current[valid] - base_tracks[valid],
                    axis=0,
                )
                paths[frame_index, object_index] = (
                    initial_centroids[object_index] + displacement
                )
            elif frame_index:
                paths[frame_index, object_index] = paths[
                    frame_index - 1,
                    object_index,
                ]
            else:
                paths[frame_index, object_index] = initial_centroids[
                    object_index
                ]

    return paths


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def flow_summaries(
    flow: np.ndarray,
    masks: np.ndarray,
    flow_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    pair_count, _, flow_height, flow_width = flow.shape
    object_count = masks.shape[1]

    background_flow = np.zeros((pair_count, 2), dtype=np.float64)
    object_flow = np.full(
        (pair_count, object_count, 2),
        np.nan,
        dtype=np.float64,
    )

    for frame_index in range(pair_count):
        object_masks = [
            resize_mask(
                masks[frame_index, object_index],
                flow_width,
                flow_height,
            )
            for object_index in range(object_count)
        ]

        union = np.zeros((flow_height, flow_width), dtype=bool)
        for object_mask in object_masks:
            union |= object_mask

        union = cv2.dilate(
            union.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)

        background = ~union
        current_flow = flow[frame_index].astype(np.float64)

        if background.sum() >= 16:
            background_vector = np.median(
                current_flow[:, background],
                axis=1,
            )
        else:
            background_vector = np.median(
                current_flow.reshape(2, -1),
                axis=1,
            )

        # Stored vectors are in downsampled-pixel units.
        background_vector /= flow_scale
        background_flow[frame_index] = background_vector

        for object_index, object_mask in enumerate(object_masks):
            if object_mask.sum() >= 4:
                vector = np.median(
                    current_flow[:, object_mask],
                    axis=1,
                )
                object_flow[frame_index, object_index] = (
                    vector / flow_scale - background_vector
                )

    return background_flow, object_flow


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(np.float64, copy=True)

    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window

    return np.column_stack([
        np.convolve(padded[:, column], kernel, mode="valid")
        for column in range(values.shape[1])
    ])


def smooth_1d(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(np.float64, copy=True)

    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def motion_onset(path: np.ndarray, object_diameter: float) -> int:
    speed = np.linalg.norm(np.diff(path, axis=0), axis=1)
    speed = smooth_1d(speed, window=3)

    threshold = max(
        0.20,
        0.005 * object_diameter,
        0.15 * float(np.percentile(speed, 90)),
    )

    moving = speed > threshold

    for index in range(max(0, len(moving) - 2)):
        if moving[index:index + 3].all():
            return index + 1

    return len(path) - 1


def mask_gap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if not mask_a.any() or not mask_b.any():
        return float("nan")

    distance = cv2.distanceTransform(
        (~mask_a).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    return float(distance[mask_b].min())


def contact_metrics(
    masks: np.ndarray,
    target_diameter: float,
) -> tuple[np.ndarray, int, float, float]:
    gaps = np.array([
        mask_gap(masks[index, 0], masks[index, 1])
        for index in range(len(masks))
    ])

    finite = np.isfinite(gaps)
    minimum_gap = float(np.min(gaps[finite]))
    contact_threshold = max(2.0, 0.05 * target_diameter)

    contacts = np.flatnonzero(finite & (gaps <= contact_threshold))
    contact_frame = int(contacts[0]) if len(contacts) else -1

    proximity = math.exp(
        -minimum_gap / max(1.0, 0.20 * target_diameter)
    )

    return gaps, contact_frame, minimum_gap, proximity


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (a_norm * b_norm), -1.0, 1.0))


def collision_causality(
    compensated_paths: np.ndarray,
    gaps: np.ndarray,
    projectile_onset: int,
    target_diameter: float,
) -> tuple[float, int, int, float, float, float, float, float, float]:
    projectile_path = compensated_paths[:, 0]
    target_path = compensated_paths[:, 1]

    projectile_velocity = np.diff(projectile_path, axis=0)
    target_velocity = np.diff(target_path, axis=0)
    target_speed = np.linalg.norm(target_velocity, axis=1)

    center_distance = smooth_1d(
        np.linalg.norm(target_path - projectile_path, axis=1),
        window=3,
    )

    # Find the first closest approach after the projectile begins moving.
    search_start = max(0, projectile_onset - 2)
    search_end = min(len(center_distance), projectile_onset + 18)
    if search_end <= search_start:
        search_start = 0
        search_end = len(center_distance)

    contact_frame = search_start + int(
        np.argmin(center_distance[search_start:search_end])
    )

    # velocity[i] is frame i -> i+1. Exclude the transition entering contact
    # and one extra transition so collision motion cannot leak into baseline.
    baseline_end = max(1, contact_frame - 2)
    baseline_speed = target_speed[:baseline_end]

    # Estimate jitter from the quieter half of early speeds. This is robust to
    # one or two contaminated samples when contact occurs near the beginning.
    quiet_cutoff = float(np.percentile(baseline_speed, 50))
    quiet_speed = baseline_speed[baseline_speed <= quiet_cutoff]
    if not len(quiet_speed):
        quiet_speed = baseline_speed

    noise_median = float(np.median(quiet_speed))
    noise_mad = float(
        np.median(np.abs(quiet_speed - noise_median))
    )
    raw_noise_threshold = noise_median + 3.0 * 1.4826 * noise_mad

    noise_floor = max(0.12, 0.0025 * target_diameter)
    noise_cap = max(0.75, 0.025 * target_diameter)
    target_noise_threshold = float(np.clip(
        raw_noise_threshold,
        noise_floor,
        noise_cap,
    ))

    # Position baseline from frames safely before contact.
    anchor_end = max(1, contact_frame - 1)
    anchor_start = max(0, anchor_end - 4)
    target_anchor = np.median(
        target_path[anchor_start:anchor_end],
        axis=0,
    )

    post_end_frame = min(len(target_path), contact_frame + 13)
    post_positions = target_path[contact_frame:post_end_frame]
    post_displacements = np.linalg.norm(
        post_positions - target_anchor,
        axis=1,
    )

    if len(post_displacements):
        best_relative_index = int(np.argmax(post_displacements))
        best_frame = contact_frame + best_relative_index
        maximum_post_displacement = float(
            post_displacements[best_relative_index]
        )
    else:
        best_frame = contact_frame
        maximum_post_displacement = 0.0

    postcontact_displacement_diameters = (
        maximum_post_displacement / max(target_diameter, 1.0)
    )

    # Detect sustained target motion after contact using both speed and net
    # displacement. This avoids declaring early mask jitter to be motion.
    displacement_threshold = max(
        3.0 * target_noise_threshold,
        0.035 * target_diameter,
    )
    target_onset = len(target_path) - 1

    for frame_index in range(contact_frame, max(contact_frame, post_end_frame - 1)):
        relative_index = frame_index - contact_frame
        displacement_window = post_displacements[
            relative_index:min(len(post_displacements), relative_index + 2)
        ]

        speed_start = max(0, frame_index - 1)
        speed_window = target_speed[
            speed_start:min(len(target_speed), speed_start + 2)
        ]

        sustained_displacement = (
            len(displacement_window) == 2
            and np.all(displacement_window > displacement_threshold)
        )
        sustained_speed = (
            len(speed_window) == 2
            and np.all(speed_window > target_noise_threshold)
        )

        if sustained_displacement or sustained_speed:
            target_onset = frame_index
            break

    # Projectile approach before collision.
    pre_start = max(0, contact_frame - 6)
    pre_velocity_end = max(pre_start, min(contact_frame - 1, len(projectile_velocity)))
    pre_projectile = projectile_velocity[pre_start:pre_velocity_end]

    incoming_vector = (
        np.median(pre_projectile, axis=0)
        if len(pre_projectile)
        else np.zeros(2)
    )

    initial_relative = target_path[pre_start] - projectile_path[pre_start]
    approach_alignment = max(
        0.0,
        vector_cosine(incoming_vector, initial_relative),
    )

    closure = max(
        0.0,
        float(center_distance[pre_start] - center_distance[contact_frame]),
    )
    closure_score = 1.0 - math.exp(
        -closure / max(1.0, 0.20 * target_diameter)
    )

    if contact_frame > pre_start:
        distance_steps = np.diff(
            center_distance[pre_start:contact_frame + 1]
        )
        closing_fraction = float(np.mean(distance_steps < 0.0))
    else:
        closing_fraction = 0.0

    collision_approach_score = (
        0.45 * closure_score
        + 0.35 * approach_alignment
        + 0.20 * closing_fraction
    )

    outgoing_vector = (
        target_path[best_frame] - target_anchor
        if best_frame < len(target_path)
        else np.zeros(2)
    )

    # For two colliding balls, the target is accelerated approximately along
    # the collision normal: the line from projectile center to target center
    # at impact. A glancing collision therefore need not send the target in
    # the projectile's incoming direction.
    normal_start = max(0, contact_frame - 1)
    normal_end = min(len(target_path), contact_frame + 2)
    relative_at_impact = (
        target_path[normal_start:normal_end]
        - projectile_path[normal_start:normal_end]
    )
    collision_normal = (
        np.median(relative_at_impact, axis=0)
        if len(relative_at_impact)
        else np.zeros(2)
    )

    # Measure whether the target leaves approximately along the collision
    # axis.  The sign of the normal depends on object ordering, while the
    # separate direction_alignment metric already checks whether motion is in
    # the intended Goal Force direction.  Using |cos| here therefore avoids
    # turning a correct transfer into zero solely because the stored normal is
    # reversed.
    collision_transfer_alignment = abs(
        vector_cosine(collision_normal, outgoing_vector)
    )

    displacement_activation = 1.0 - math.exp(
        -maximum_post_displacement
        / max(1.0, 0.15 * target_diameter)
    )

    # Post-contact speeds begin with the transition entering contact. The
    # transition index contact_frame - 1 is therefore post-contact evidence.
    post_speed_start = max(0, contact_frame - 1)
    post_speed_end = min(len(target_speed), contact_frame + 10)
    post_speed = target_speed[post_speed_start:post_speed_end]
    post_excess_speed = np.clip(
        post_speed - target_noise_threshold,
        0.0,
        None,
    )
    speed_activation = 1.0 - math.exp(
        -float(post_excess_speed.sum())
        / max(1.0, 0.08 * target_diameter)
    )

    # Alignment is supportive rather than mandatory: the target may visibly
    # move after collision even when the estimated direction is noisy.
    postcontact_response = (
        displacement_activation
        * (0.35 + 0.65 * speed_activation)
        * (0.70 + 0.30 * collision_transfer_alignment)
    )

    # Penalize only meaningful target displacement before contact. Using net
    # displacement instead of accumulated speed prevents jitter from adding up.
    pre_position_end = max(1, contact_frame)
    pre_positions = target_path[:pre_position_end]
    pre_displacements = np.linalg.norm(
        pre_positions - target_anchor,
        axis=1,
    )
    maximum_pre_displacement = (
        float(np.max(pre_displacements))
        if len(pre_displacements)
        else 0.0
    )

    position_noise = max(
        3.0 * target_noise_threshold,
        0.025 * target_diameter,
    )
    meaningful_pre_motion = max(
        0.0,
        maximum_pre_displacement - position_noise,
    )
    meaningful_post_motion = max(
        0.0,
        maximum_post_displacement - position_noise,
    )
    precontact_motion_fraction = (
        meaningful_pre_motion
        / (meaningful_pre_motion + meaningful_post_motion + 1e-8)
    )

    finite = np.isfinite(gaps)
    minimum_gap = (
        float(np.min(gaps[finite]))
        if finite.any()
        else float("inf")
    )
    proximity = math.exp(
        -minimum_gap / max(1.0, 0.20 * target_diameter)
    )

    contact_causality = (
        proximity
        * collision_approach_score
        * postcontact_response
        * math.exp(-3.0 * precontact_motion_fraction)
    )

    return (
        contact_causality,
        contact_frame,
        target_onset,
        precontact_motion_fraction,
        postcontact_response,
        collision_approach_score,
        collision_transfer_alignment,
        target_noise_threshold,
        postcontact_displacement_diameters,
    )



def _meaningful_motion_fraction(
    path: np.ndarray,
    event_frame: int,
    target_diameter: float,
) -> tuple[float, float, float]:
    event_frame = int(np.clip(event_frame, 0, len(path) - 1))

    anchor_end = max(1, event_frame)
    anchor_start = max(0, anchor_end - 4)
    anchor = np.median(path[anchor_start:anchor_end], axis=0)

    pre = path[:max(1, event_frame)]
    post = path[event_frame:min(len(path), event_frame + 13)]

    max_pre = (
        float(np.max(np.linalg.norm(pre - anchor, axis=1)))
        if len(pre)
        else 0.0
    )
    max_post = (
        float(np.max(np.linalg.norm(post - anchor, axis=1)))
        if len(post)
        else 0.0
    )

    # A small position dead-band keeps segmentation/tracking jitter from being
    # interpreted as causal target motion.
    noise = max(0.5, 0.025 * target_diameter)
    meaningful_pre = max(0.0, max_pre - noise)
    meaningful_post = max(0.0, max_post - noise)

    fraction = meaningful_pre / (
        meaningful_pre + meaningful_post + 1e-8
    )
    return fraction, max_pre, max_post


def strike_causality(
    compensated_paths: np.ndarray,
    gaps: np.ndarray,
    actor_onset: int,
    target_onset: int,
    contact_frame: int,
    target_diameter: float,
) -> tuple[float, int, float, float, float]:
    target_path = compensated_paths[:, 1]

    finite = np.isfinite(gaps)
    if contact_frame < 0 and finite.any():
        finite_indices = np.flatnonzero(finite)
        contact_frame = int(
            finite_indices[np.argmin(gaps[finite_indices])]
        )

    if contact_frame < 0:
        contact_frame = int(np.clip(actor_onset, 0, len(target_path) - 1))

    minimum_gap = (
        float(np.min(gaps[finite])) if finite.any() else float("inf")
    )
    proximity = math.exp(
        -minimum_gap / max(1.0, 0.20 * target_diameter)
    )

    pre_fraction, _, max_post = _meaningful_motion_fraction(
        target_path,
        contact_frame,
        target_diameter,
    )
    post_displacement_diameters = (
        max_post / max(target_diameter, 1.0)
    )
    response = 1.0 - math.exp(
        -max_post / max(1.0, 0.15 * target_diameter)
    )

    # Target motion should not substantially precede the actor.  A few frames
    # of lag after impact are acceptable.
    early_target = max(0, actor_onset - target_onset)
    order_score = math.exp(-early_target / 2.0)

    lag = max(0, target_onset - contact_frame)
    lag_score = math.exp(-max(0, lag - 5) / 6.0)

    score = (
        proximity
        * response
        * order_score
        * lag_score
        * math.exp(-3.0 * pre_fraction)
    )

    return (
        score,
        contact_frame,
        pre_fraction,
        response,
        post_displacement_diameters,
    )


def direct_actuation_causality(
    compensated_paths: np.ndarray,
    gaps: np.ndarray,
    actor_onset: int,
    target_onset: int,
    target_diameter: float,
) -> tuple[float, float, float, float]:
    actor_path = compensated_paths[:, 0]
    target_path = compensated_paths[:, 1]

    event_frame = int(np.clip(actor_onset, 0, len(target_path) - 1))

    finite = np.isfinite(gaps)
    minimum_gap = (
        float(np.min(gaps[finite])) if finite.any() else float("inf")
    )

    # Persistent/direct contact is better represented by proximity around the
    # actuation window than by the first overlap frame.
    window_start = max(0, event_frame - 2)
    window_end = min(
        len(gaps),
        max(event_frame + 8, target_onset + 4),
    )
    local_gap = gaps[window_start:window_end]
    local_finite = np.isfinite(local_gap)
    if local_finite.any():
        typical_gap = float(np.median(local_gap[local_finite]))
    else:
        typical_gap = minimum_gap

    proximity = math.exp(
        -typical_gap / max(1.0, 0.25 * target_diameter)
    )

    pre_fraction, _, max_post = _meaningful_motion_fraction(
        target_path,
        event_frame,
        target_diameter,
    )
    post_displacement_diameters = (
        max_post / max(target_diameter, 1.0)
    )
    response = 1.0 - math.exp(
        -max_post / max(1.0, 0.15 * target_diameter)
    )

    # Actor should begin before or approximately with the target.  We allow a
    # short response lag because deformable actors can be harder to track.
    early_target = max(0, actor_onset - target_onset)
    order_score = math.exp(-early_target / 2.0)

    lag = max(0, target_onset - actor_onset)
    lag_score = math.exp(-max(0, lag - 6) / 8.0)

    # Require the actor itself to show some displacement after its onset.  This
    # prevents a static hand/paw mask that happens to overlap the target from
    # receiving a high causal score.
    actor_anchor = actor_path[event_frame]
    actor_post = actor_path[event_frame:min(len(actor_path), event_frame + 10)]
    actor_disp = (
        float(np.max(np.linalg.norm(actor_post - actor_anchor, axis=1)))
        if len(actor_post)
        else 0.0
    )
    actor_scale = max(1.0, 0.10 * target_diameter)
    actor_activation = 1.0 - math.exp(-actor_disp / actor_scale)

    score = (
        proximity
        * response
        * (0.35 + 0.65 * actor_activation)
        * order_score
        * lag_score
        * math.exp(-3.0 * pre_fraction)
    )

    return (
        score,
        pre_fraction,
        response,
        post_displacement_diameters,
    )

def robust_flow_consistency(
    compensated_paths: np.ndarray,
    object_flow: np.ndarray,
) -> float:
    path_velocity = np.diff(compensated_paths, axis=0)
    errors = []
    motion = []

    object_count = min(
        compensated_paths.shape[1],
        object_flow.shape[1],
    )

    for object_index in range(object_count):
        valid = np.isfinite(object_flow[:, object_index]).all(axis=1)
        if not valid.any():
            continue

        errors.extend(
            np.linalg.norm(
                object_flow[valid, object_index]
                - path_velocity[valid, object_index],
                axis=1,
            )
        )
        motion.extend(
            np.linalg.norm(path_velocity[valid, object_index], axis=1)
        )

    if not errors:
        return 0.0

    error = float(np.median(errors))
    scale = float(np.median(motion)) + 1.0
    return math.exp(-error / scale)


def rounded_or_nan(value: float, digits: int = 6) -> float:
    value = float(value)
    return round(value, digits) if math.isfinite(value) else float("nan")


def extract_metrics(path: Path) -> dict[str, object]:
    scenario, seed = parse_name(path)
    config = SCENARIOS[scenario]
    interaction_mode = str(config["mode"])

    with np.load(path) as data:
        required = {
            "masks",
            "tracks",
            "visibility",
            "track_object_ids",
            "flow",
            "flow_scale",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{path.name} is missing corrected features: "
                f"{sorted(missing)}"
            )

        masks = data["masks"]
        tracks = data["tracks"].astype(np.float64)
        visibility = data["visibility"].astype(bool)
        object_ids = data["track_object_ids"].astype(np.int64)
        flow = data["flow"]
        flow_scale = float(data["flow_scale"].item())

    if masks.ndim != 4:
        raise ValueError(
            f"{path.name}: expected masks shaped (T, objects, H, W), "
            f"got {masks.shape}"
        )

    object_count = int(masks.shape[1])

    # All two-object scenarios were annotated as:
    #   object 0 = actor/projectile
    #   object 1 = target
    #
    # The original pool features were known to contain those semantics in the
    # opposite stored order, so preserve the previously validated correction.
    if scenario == "pool":
        if object_count != 2:
            raise ValueError(
                f"{path.name}: pool requires exactly 2 tracked objects; "
                f"found {object_count}"
            )
        masks = masks[:, [1, 0]]
        object_ids = 1 - object_ids

    if interaction_mode == "single_target":
        # The Colab feature extractor used for this dataset was hard-coded to
        # allocate two object slots and to save object 1 as the target.  For
        # toycar the hand/actor is not visible in frame 0, so object 0 is not a
        # meaningful projectile.  Discard that slot completely and keep only
        # the second (target) slot.  This avoids fabricating contact physics
        # while letting us reuse the already-extracted target features.
        if object_count == 2:
            target_slot = 1
            target_points = object_ids == target_slot
            if not target_points.any():
                raise ValueError(
                    f"{path.name}: no CoTracker points found for toycar target "
                    f"slot {target_slot}."
                )

            masks = masks[:, target_slot:target_slot + 1]
            tracks = tracks[:, target_points]
            visibility = visibility[:, target_points]
            object_ids = np.zeros(int(target_points.sum()), dtype=np.int64)
            object_count = 1

        elif object_count != 1:
            raise ValueError(
                f"{path.name}: {scenario} expects either one semantic target "
                f"object or the two-slot Colab representation; found "
                f"{object_count}."
            )

        projectile_index = None
        target_index = 0
    else:
        if object_count != 2:
            raise ValueError(
                f"{path.name}: {scenario} requires exactly 2 tracked objects "
                f"(actor/projectile then target), found {object_count}"
            )
        projectile_index = 0
        target_index = 1

    centroids, areas = mask_centroids(masks)

    diameters = 2.0 * np.sqrt(
        np.maximum(areas.mean(axis=0), 1.0) / math.pi
    )
    target_diameter = float(diameters[target_index])

    cotracker_paths = track_paths(
        tracks,
        visibility,
        object_ids,
        centroids[0],
    )

    # Fuse SAM centroids with long-term CoTracker displacement.
    fused_paths = 0.5 * centroids + 0.5 * cotracker_paths

    background_flow, object_flow = flow_summaries(
        flow,
        masks,
        flow_scale,
    )

    camera_path = np.zeros((len(fused_paths), 2), dtype=np.float64)
    camera_path[1:] = np.cumsum(background_flow, axis=0)
    compensated = fused_paths - camera_path[:, None, :]

    # ------------------------------------------------------------------
    # Goal-directed target motion
    # ------------------------------------------------------------------
    angle_deg = float(config["angle_deg"])
    angle_rad = math.radians(angle_deg)
    desired_direction = np.array([
        math.cos(angle_rad),
        -math.sin(angle_rad),
    ])

    target_path = compensated[:, target_index]
    target_relative = target_path - target_path[0]
    projection = target_relative @ desired_direction

    goal_frame = int(np.argmax(projection))
    goal_progress_px = max(0.0, float(projection[goal_frame]))
    goal_progress_diameters = (
        goal_progress_px / max(target_diameter, 1.0)
    )
    goal_completion = 1.0 - math.exp(-goal_progress_diameters)

    # Frame-to-frame direction score. This avoids saturation from using only
    # final displacement and explicitly penalizes reverse motion.
    target_velocity = np.diff(target_path, axis=0)
    target_speed = np.linalg.norm(target_velocity, axis=1)
    directed_motion = target_velocity @ desired_direction

    active_threshold = max(
        0.15,
        0.003 * target_diameter,
        0.10 * float(np.percentile(target_speed, 90)),
    )
    active = target_speed > active_threshold

    if active.any():
        forward_motion = np.clip(directed_motion[active], 0.0, None)
        reverse_motion = np.clip(-directed_motion[active], 0.0, None)
        lateral_motion = np.sqrt(
            np.maximum(
                target_speed[active] ** 2
                - directed_motion[active] ** 2,
                0.0,
            )
        )

        forward_sum = float(forward_motion.sum())
        reverse_sum = float(reverse_motion.sum())
        lateral_sum = float(lateral_motion.sum())
        total_motion = forward_sum + reverse_sum + lateral_sum + 1e-8

        forward_motion_fraction = forward_sum / total_motion
        reverse_motion_fraction = reverse_sum / total_motion
        direction_alignment = (
            forward_motion_fraction
            * math.exp(-2.5 * reverse_motion_fraction)
        )
    else:
        forward_motion_fraction = 0.0
        reverse_motion_fraction = 0.0
        direction_alignment = 0.0

    target_onset = motion_onset(
        target_path,
        target_diameter,
    )

    # ------------------------------------------------------------------
    # Interaction/contact physics
    # ------------------------------------------------------------------
    collision_approach_score = float("nan")
    collision_transfer_alignment = float("nan")
    target_noise_threshold = float("nan")
    postcontact_displacement_diameters = float("nan")

    if projectile_index is None:
        # There is no visible first-frame actor for toycar. The Goal Force CSV
        # specifies only an indirect target force, so a projectile/contact score
        # would be fabricated. Keep the target-motion metrics, but mark contact
        # metrics unavailable so the scorer can ignore/renormalize them later.
        contact_metric_valid = 0
        projectile_onset = -1
        contact_frame = -1
        minimum_gap = float("nan")
        precontact_motion_fraction = float("nan")
        postcontact_response = float("nan")
        contact_causality = float("nan")

    else:
        contact_metric_valid = 1
        projectile_onset = motion_onset(
            compensated[:, projectile_index],
            float(diameters[projectile_index]),
        )

        gaps, contact_frame, minimum_gap, proximity = contact_metrics(
            masks[:, [projectile_index, target_index]],
            target_diameter,
        )

        if interaction_mode == "collision":
            (
                contact_causality,
                contact_frame,
                target_onset,
                precontact_motion_fraction,
                postcontact_response,
                collision_approach_score,
                collision_transfer_alignment,
                target_noise_threshold,
                postcontact_displacement_diameters,
            ) = collision_causality(
                compensated[:, [projectile_index, target_index]],
                gaps,
                projectile_onset,
                target_diameter,
            )

        elif interaction_mode == "strike":
            (
                contact_causality,
                contact_frame,
                precontact_motion_fraction,
                postcontact_response,
                postcontact_displacement_diameters,
            ) = strike_causality(
                compensated[:, [projectile_index, target_index]],
                gaps,
                projectile_onset,
                target_onset,
                contact_frame,
                target_diameter,
            )

        elif interaction_mode == "direct_actuation":
            (
                contact_causality,
                precontact_motion_fraction,
                postcontact_response,
                postcontact_displacement_diameters,
            ) = direct_actuation_causality(
                compensated[:, [projectile_index, target_index]],
                gaps,
                projectile_onset,
                target_onset,
                target_diameter,
            )

        else:
            raise ValueError(
                f"Unknown interaction mode for {scenario}: "
                f"{interaction_mode}"
            )

    # ------------------------------------------------------------------
    # Stability / feature-consistency diagnostics
    # ------------------------------------------------------------------
    # Stability is intentionally target-only.  Hands, paws, clubs, and other
    # actors can deform or articulate substantially, which made a shared
    # actor+target stability score incomparable across interaction classes.
    target_areas = areas[:, target_index]
    target_log_area = np.log(np.maximum(target_areas, 1.0))
    target_area_changes = np.abs(np.diff(target_log_area))
    mask_area_change_robust = float(
        np.median(target_area_changes)
        + 0.25 * np.percentile(target_area_changes, 90)
    )

    target_velocity_for_stability = np.diff(target_path, axis=0)
    target_acceleration = np.diff(target_velocity_for_stability, axis=0)
    target_jerk = np.diff(target_acceleration, axis=0)
    trajectory_jerk_norm = (
        float(
            np.median(np.linalg.norm(target_jerk, axis=1))
            / max(target_diameter, 1.0)
        )
        if len(target_jerk)
        else 0.0
    )

    target_disagreement = np.linalg.norm(
        (centroids[:, target_index] - centroids[0, target_index])
        - (
            cotracker_paths[:, target_index]
            - cotracker_paths[0, target_index]
        ),
        axis=1,
    )
    track_mask_disagreement_norm = float(
        np.median(target_disagreement) / max(target_diameter, 1.0)
    )

    motion_stability = math.exp(
        -8.0 * mask_area_change_robust
        -10.0 * trajectory_jerk_norm
        -3.0 * track_mask_disagreement_norm
    )

    flow_consistency = robust_flow_consistency(
        compensated[:, target_index:target_index + 1],
        object_flow[:, target_index:target_index + 1],
    )

    return {
        "scenario": scenario,
        "seed": seed,
        "video": path.stem,
        "interaction_mode": interaction_mode,
        "object_count": object_count,
        "contact_metric_valid": contact_metric_valid,
        "expected_angle_deg": round(angle_deg, 3),
        "goal_completion": round(goal_completion, 6),
        "goal_progress_diameters": round(
            goal_progress_diameters,
            6,
        ),
        "goal_progress_px": round(goal_progress_px, 3),
        "direction_alignment": round(direction_alignment, 6),
        "contact_causality": rounded_or_nan(
            contact_causality,
            6,
        ),
        "motion_stability": round(motion_stability, 6),
        "flow_consistency": round(flow_consistency, 6),
        "track_visibility": round(
            float(
                visibility[:, object_ids == target_index].mean()
                if np.any(object_ids == target_index)
                else visibility.mean()
            ),
            6,
        ),
        "projectile_onset_frame": projectile_onset,
        "target_onset_frame": target_onset,
        "contact_frame": contact_frame,
        "minimum_gap_px": rounded_or_nan(minimum_gap, 3),
        "target_diameter_px": round(target_diameter, 3),
        "forward_motion_fraction": round(
            forward_motion_fraction,
            6,
        ),
        "reverse_motion_fraction": round(
            reverse_motion_fraction,
            6,
        ),
        "precontact_motion_fraction": rounded_or_nan(
            precontact_motion_fraction,
            6,
        ),
        "postcontact_response": rounded_or_nan(
            postcontact_response,
            6,
        ),
        "collision_approach_score": rounded_or_nan(
            collision_approach_score,
            6,
        ),
        "collision_transfer_alignment": rounded_or_nan(
            collision_transfer_alignment,
            6,
        ),
        "target_noise_threshold_px": rounded_or_nan(
            target_noise_threshold,
            6,
        ),
        "postcontact_displacement_diameters": rounded_or_nan(
            postcontact_displacement_diameters,
            6,
        ),
        "mask_area_change_robust": round(
            mask_area_change_robust,
            6,
        ),
        "trajectory_jerk_norm": round(
            trajectory_jerk_norm,
            6,
        ),
        "track_mask_disagreement_norm": round(
            track_mask_disagreement_norm,
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("features"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics.csv"),
    )
    args = parser.parse_args()

    paths = sorted(args.features.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No .npz files found under {args.features}"
        )

    rows = [extract_metrics(path) for path in paths]
    rows.sort(key=lambda row: (str(row["scenario"]), int(row["seed"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output}")

    scenario_counts: dict[str, int] = {}
    for row in rows:
        name = str(row["scenario"])
        scenario_counts[name] = scenario_counts.get(name, 0) + 1

    print("Scenario counts:")
    for name in sorted(scenario_counts):
        print(f"  {name}: {scenario_counts[name]}")

    if len(rows) == 40 and all(
        scenario_counts.get(name) == 4
        for name in SCENARIOS
    ):
        print("Training feature set verified: 10 scenarios x 4 seeds = 40 rows.")


if __name__ == "__main__":
    main()