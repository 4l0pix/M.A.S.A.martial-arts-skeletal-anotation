//pure chart helpers: source-time evidence is never interpolated
export function trackTimeline(records, rows, trackId, feature) {
  const byFrame = new Map(rows
    .filter((row) => row.track_id === trackId)
    .map((row) => [row.frame_index, row]));
  return records.map((record) => {
    const value = byFrame.get(record.frame_index)?.[feature];
    return {
      frame_index: record.frame_index,
      timestamp_s: record.timestamp_s,
      value: Number.isFinite(value) ? value : null,
    };
  });
}

export function chartGeometry(samples) {
  const plot = { left: 72, right: 778, top: 22, bottom: 216 };
  const finiteTimes = samples.map((sample) => sample.timestamp_s).filter(Number.isFinite);
  const evidence = samples.filter((sample) =>
    Number.isFinite(sample.timestamp_s) && Number.isFinite(sample.value));
  const extent = (values) => values.reduce(
    ([low, high], value) => [Math.min(low, value), Math.max(high, value)],
    [Infinity, -Infinity],
  );
  const xDomain = finiteTimes.length ? extent(finiteTimes) : null;
  if (xDomain && xDomain[0] === xDomain[1]) {
    xDomain[0] -= 0.5;
    xDomain[1] += 0.5;
  }
  if (!evidence.length) return { segments: [], xDomain, yDomain: null, plot };
  const yDomain = extent(evidence.map((sample) => sample.value));
  const padding = (yDomain[1] - yDomain[0]) * 0.08 || Math.max(Math.abs(yDomain[0]) * 0.08, 1);
  yDomain[0] -= padding;
  yDomain[1] += padding;
  const segments = [];
  let segment = null;
  for (const sample of samples) {
    if (!Number.isFinite(sample.timestamp_s) || !Number.isFinite(sample.value)) {
      segment = null;
      continue;
    }
    if (!segment) {
      segment = [];
      segments.push(segment);
    }
    segment.push({
      ...sample,
      x: plot.left + (sample.timestamp_s - xDomain[0]) / (xDomain[1] - xDomain[0])
        * (plot.right - plot.left),
      y: plot.bottom - (sample.value - yDomain[0]) / (yDomain[1] - yDomain[0])
        * (plot.bottom - plot.top),
    });
  }
  return { segments, xDomain, yDomain, plot };
}

export const CONFIRMED_NOTE = 'User confirmed permission to process this footage in the local UI; '
  + 'no additional rights notes supplied.';

export function validateRun(source, permission, config, intervalLimit = 120) {
  if (!source) return { field: 'upload-file', message: 'Select and inspect a source before running analysis.' };
  if (!permission) return { field: 'permission', message: 'Confirm “I have permission to process this footage”, then run analysis again. Written notes are optional.' };
  const duration = source.info.duration_s;
  if (!Number.isFinite(config.start_s) || config.start_s < 0 || config.start_s >= duration) {
    return { field: 'start-s', message: 'Clip start must be a finite time within the source.' };
  }
  if (!Number.isFinite(config.end_s) || config.end_s <= config.start_s
      || config.end_s > duration || config.end_s - config.start_s > intervalLimit) {
    return { field: 'end-s', message: `Clip end must be after the start, within the source, and at most ${intervalLimit} seconds later.` };
  }
  if (!Number.isInteger(config.stride) || config.stride < 1 || config.stride > 120) {
    return { field: 'stride', message: 'Process every Nth frame must be an integer from 1 to 120.' };
  }
  for (const [name, field] of [['confidence', 'confidence'], ['keypoint_confidence', 'keypoint-confidence']]) {
    if (!Number.isFinite(config[name]) || config[name] <= 0 || config[name] > 1) {
      return { field, message: 'Confidence thresholds must be finite and greater than 0, up to 1.' };
    }
  }
  for (const [name, field, allowed] of [
    ['model', 'model', ['yolo11n-pose.pt', 'yolo11s-pose.pt']],
    ['tracker', 'tracker', ['botsort.yaml', 'bytetrack.yaml']],
    ['device', 'device', ['cpu', 'mps']],
    ['expected_people', 'expected-people', [1, 2]],
  ]) {
    if (!allowed.includes(config[name])) return { field, message: `Choose a supported ${field} setting.` };
  }
  const mode = config.selection_mode ?? 'automatic';
  if (!['automatic', 'manual'].includes(mode)) return { field: 'selection-mode', message: 'Choose manual fighter selection or the automatic person cap.' };
  if (mode === 'manual') {
    const boxes = config.fighter_boxes;
    if (!Array.isArray(boxes) || boxes.length !== config.expected_people
        || config.fighter_seed_frame !== Math.ceil(config.start_s * source.info.fps)
        || boxes.some((box) => !Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)
          || box[0] < 0 || box[1] < 0 || box[2] > source.info.width || box[3] > source.info.height
          || box[2] <= box[0] || box[3] <= box[1])
        || new Set(boxes.map((box) => JSON.stringify(box))).size !== config.expected_people) {
      return { field: 'mark-fighter-a', message: 'Draw the selected fighters on the first analyzed source frame. Use Mark Fighter A / B; do not reuse boxes from another frame.' };
    }
  }
  return null;
}

export function settingsDiffer(saved, current, duration) {
  const canonical = (config) => JSON.stringify([
    config.model, config.tracker, config.device, config.start_s, config.end_s ?? duration,
    config.stride, config.confidence, config.keypoint_confidence, config.expected_people,
    config.rights_note?.trim() || CONFIRMED_NOTE, config.source_note?.trim() || '',
    config.selection_mode ?? 'automatic', config.fighter_boxes ?? null, config.fighter_seed_frame ?? null,
    config.fixed_camera ?? false,
  ]);
  return canonical(saved) !== canonical(current);
}
