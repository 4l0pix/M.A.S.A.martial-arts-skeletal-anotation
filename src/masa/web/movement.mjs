//experimental evidence and explicit review exports; no invented judgments
import { chartGeometry } from './chart.mjs';

export function movementGeometry(records) {
  const samples = (value) => records.map((record) => ({ frame_index: record.frame_index,
    timestamp_s: record.timestamp_s, value: value(record.movement) ?? null }));
  const distance = chartGeometry(samples((m) => m?.distance_px));
  const roles = ['A', 'B'].map((role) => ({ role,
    samples: samples((m) => m?.approach_px_s?.[role]) }));
  const shared = chartGeometry(roles.flatMap((item) => item.samples));
  const segments = roles.flatMap(({ role, samples: values }) => chartGeometry(values).segments
    .map((points) => ({ role, points: points.map((point) => ({ ...point,
      y: shared.plot.bottom - (point.value - shared.yDomain[0])
        / (shared.yDomain[1] - shared.yDomain[0]) * (shared.plot.bottom - shared.plot.top),
    })) })));
  return { distance, approach: { ...shared, segments } };
}

export function reviewDocument(run, reviews) {
  const phases = run.manifest.movement.phases;
  const unique = new Map();
  for (const entry of reviews) {
    const phase = phases.find((item) => item.id === entry.phase_id);
    if (!phase || !['correct', 'incorrect', 'unclear'].includes(entry.verdict)) {
      throw new Error('Choose an existing interval and a review verdict.');
    }
    unique.set(phase.id, { ...phase, phase_id: phase.id, verdict: entry.verdict,
      recorded_at: entry.recorded_at ?? null, note: String(entry.note ?? '').slice(0, 2000) });
  }
  return { schema_version: '1.0', job_id: run.job.id, source_sha256: run.manifest.source_sha256,
    scope: 'manual review of descriptive movement labels, not strikes or tactical success',
    human_reviewed_intervals: unique.size, reviews: [...unique.values()] };
}

export function createMovementView({ document: doc, onSeek, createObjectURL, revokeObjectURL }) {
  const el = (id) => doc.getElementById(id);
  let run = null;
  let reviews = [];
  let download = null;
  const node = (parent, tag, attrs, text) => {
    const item = doc.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, String(value));
    if (text !== undefined) item.textContent = text;
    parent.append(item);
    return item;
  };
  function chart(id, geometry, unit) {
    const group = el(id);
    group.replaceChildren();
    const { plot, xDomain, yDomain } = geometry;
    if (!xDomain || !yDomain) {
      node(group, 'text', { x: 72, y: 80 }, `No reliable ${unit} evidence; missing data is not zero.`);
      return;
    }
    for (const fraction of [0, 0.5, 1]) {
      const y = plot.bottom - fraction * (plot.bottom - plot.top);
      node(group, 'line', { x1: plot.left, x2: plot.right, y1: y, y2: y, class: 'chart-grid' });
      node(group, 'text', { x: 64, y: y + 4, 'text-anchor': 'end' },
        (yDomain[0] + fraction * (yDomain[1] - yDomain[0])).toFixed(1));
      node(group, 'text', { x: plot.left + fraction * (plot.right - plot.left), y: 236,
        'text-anchor': 'middle' }, (xDomain[0] + fraction * (xDomain[1] - xDomain[0])).toFixed(2));
    }
    node(group, 'text', { x: 425, y: 254, 'text-anchor': 'middle' }, `Source time (s)   ${unit}`);
    for (const segment of geometry.segments) {
      const points = segment.points ?? segment;
      node(group, 'polyline', { points: points.map((p) => `${p.x},${p.y}`).join(' '),
        class: `motion-line motion-${segment.role ?? 'distance'}` });
      if (points.length === 1) node(group, 'circle', { cx: points[0].x, cy: points[0].y,
        r: 3, class: `motion-point motion-${segment.role ?? 'distance'}` });
    }
  }
  async function jump(phase) {
    el('phase-review-select').value = String(phase.id);
    const previous = reviews.find((entry) => entry.phase_id === phase.id);
    el('phase-verdict').value = previous?.verdict ?? '';
    el('phase-review-note').value = previous?.note ?? '';
    return onSeek(phase.first_ordinal);
  }
  function render(value) {
    if (download) revokeObjectURL(download);
    download = null;
    run = value;
    reviews = [];
    el('download-phase-reviews').removeAttribute('href');
    const report = run?.manifest.movement;
    el('movement-panel').hidden = !report?.enabled;
    el('phase-ribbon-panel').hidden = !report?.enabled;
    if (!report?.enabled) return;
    const counts = (values) => ['reliable', 'uncertain', 'excluded']
      .map((key) => `${key}: ${values?.[key] ?? 0}`).join('   ');
    el('movement-summary').textContent = `Frames — ${counts(report.frame_counts)}. Intervals — ${counts(report.interval_counts)}. Camera: ${report.fixed_camera_confirmed ? 'user confirmed fixed' : 'not confirmed fixed; no approach labels'}.`;
    el('phase-review-summary').textContent = '0 intervals reviewed. No human validation has been recorded.';
    const phases = report.phases;
    const ribbon = el('phase-ribbon');
    ribbon.replaceChildren();
    el('phase-review-select').replaceChildren();
    const start = phases[0]?.start_s ?? 0;
    const duration = (phases.at(-1)?.end_s ?? start) - start;
    for (const phase of phases) {
      const label = `${phase.label.replaceAll('_', ' ')}   ${phase.status}   source frames ${phase.start_frame}–${phase.end_frame_exclusive - 1}`;
      const rect = node(ribbon, 'rect', { x: duration ? (phase.start_s - start) / duration * 800 : 0,
        width: duration ? (phase.end_s - phase.start_s) / duration * 800 : 800,
        y: 2, height: 40, class: `phase phase-${phase.label}`, role: 'button', tabindex: 0,
        'aria-label': label });
      node(rect, 'title', {}, label);
      rect.addEventListener('click', () => jump(phase));
      rect.addEventListener('keydown', (event) => {
        if (['Enter', ' '].includes(event.key)) { event.preventDefault(); jump(phase); }
      });
      const option = doc.createElement('option');
      option.value = String(phase.id);
      option.textContent = label;
      el('phase-review-select').append(option);
    }
    el('phase-review-select').value = phases.length ? String(phases[0].id) : '';
    el('phase-verdict').value = '';
    el('phase-review-note').value = '';
    const geometry = movementGeometry(run.records);
    chart('movement-distance', geometry.distance, 'distance (px)');
    chart('movement-approach', geometry.approach, 'approach (px/s): A violet / B light');
  }
  el('phase-jump').addEventListener('click', () => {
    const phase = run?.manifest.movement?.phases.find((p) => String(p.id) === el('phase-review-select').value);
    if (phase) return jump(phase);
  });
  el('phase-review-select').addEventListener('change', () => {
    const previous = reviews.find((entry) => String(entry.phase_id) === el('phase-review-select').value);
    el('phase-verdict').value = previous?.verdict ?? '';
    el('phase-review-note').value = previous?.note ?? '';
  });
  el('record-phase-review').addEventListener('click', () => {
    if (!run) return;
    try {
      const entry = { phase_id: Number(el('phase-review-select').value),
        verdict: el('phase-verdict').value, note: el('phase-review-note').value,
        recorded_at: new Date().toISOString() };
      const result = reviewDocument(run, [...reviews, entry]);
      reviews = result.reviews;
      if (download) revokeObjectURL(download);
      download = createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
      el('download-phase-reviews').href = download;
      el('phase-review-summary').textContent = `${result.human_reviewed_intervals} intervals reviewed. Download the JSON to keep these judgments; they are not stored on the server or included in the run ZIP.`;
    } catch (error) { el('phase-review-summary').textContent = error.message; }
  });
  return { render };
}
