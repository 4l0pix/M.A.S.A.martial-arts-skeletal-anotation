import { chartGeometry, settingsDiffer, trackTimeline, validateRun } from './chart.mjs';
import { createFighterPicker } from './targets.mjs';
import { createMovementView } from './movement.mjs';

//all requests, including frame images, stay on this app's origin
export function createAPI({ origin = globalThis.location?.origin, getToken, fetcher = globalThis.fetch }) {
  return async function request(path, options = {}) {
    const url = new URL(path, origin);
    if (url.origin !== origin || !url.pathname.startsWith('/api/')) {
      throw new Error('Only same-origin API requests are allowed.');
    }
    const { json, responseType = 'json', ...requestOptions } = options;
    const method = (requestOptions.method || 'GET').toUpperCase();
    const headers = new Headers(requestOptions.headers);
    if (!['GET', 'HEAD'].includes(method)) {
      const token = getToken?.();
      if (!token) throw new Error('The browser session is not ready. Reload this page before retrying.');
      headers.set('X-CSRF-Token', token);
    }
    if (json !== undefined) headers.set('Content-Type', 'application/json');
    const response = await fetcher(url.href, {
      ...requestOptions,
      method,
      headers,
      ...(json === undefined ? {} : { body: JSON.stringify(json) }),
      credentials: 'same-origin',
      mode: 'same-origin',
      cache: 'no-store',
      redirect: 'error',
    });
    if (!response.ok) {
      let payload;
      try { payload = await response.json(); } catch { payload = {}; }
      const message = typeof payload?.error === 'string' ? payload.error
        : typeof payload?.detail === 'string' ? payload.detail
          : `Engine request failed (${response.status}). Please retry.`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    if (responseType === 'blob') return response.blob();
    if (response.status === 204) return null;
    try { return await response.json(); } catch {
      throw new Error('The engine returned an unreadable response. Reload this page or retry.');
    }
  };
}

const CONFIG_FIELDS = {
  model: 'model', tracker: 'tracker', device: 'device', start_s: 'start-s', end_s: 'end-s',
  stride: 'stride', confidence: 'confidence', keypoint_confidence: 'keypoint-confidence',
  expected_people: 'expected-people', rights_note: 'rights-note', source_note: 'source-note',
};
const NUMERIC_FIELDS = new Set([
  'start_s', 'end_s', 'stride', 'confidence', 'keypoint_confidence', 'expected_people',
]);

export function createController({
  document: doc = globalThis.document,
  origin = globalThis.location?.origin,
  request,
  setTimer = globalThis.setTimeout,
  clearTimer = globalThis.clearTimeout,
  createObjectURL = (blob) => URL.createObjectURL(blob),
  revokeObjectURL = (url) => URL.revokeObjectURL(url),
} = {}) {
  const el = (id) => doc.getElementById(id);
  const state = {
    token: null, source: null, sourceStale: false, job: null, results: null, busy: false, uncertain: false,
    limits: { upload_bytes: 200 * 1024 * 1024, interval_s: 120 }, defaults: {},
  };
  const api = request || createAPI({ origin, getToken: () => state.token });
  const imageURLs = new Map();
  const imageRequests = new Map();
  let pollTimer = null;
  let polling = null;
  let resultRequest = 0;
  let reviewOrdinal = 0;
  const sourcePath = () => `/api/sources/${encodeURIComponent(state.source.id)}`;
  const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable';
  const errorMessage = (error) => error instanceof Error ? error.message : 'The request failed. Please retry.';
  const picker = createFighterPicker({ document: doc,
    getSource: () => state.sourceStale ? null : state.source,
    getCount: () => Number(el('expected-people').value),
    getStartFrame: () => state.source && el('start-s').value.trim()
      ? Math.ceil(Number(el('start-s').value) * state.source.info.fps) : NaN,
    showFrame: sourceFrame, onChange: savedWarning,
  });
  const movementView = createMovementView({ document: doc, createObjectURL, revokeObjectURL,
    onSeek: async (ordinal) => {
      const run = state.results;
      const record = run?.records[ordinal];
      if (!record) return;
      await Promise.all([reviewFrame(ordinal), sourceFrame(record.frame_index)]);
      if (state.results !== run) return;
      el('source-video').currentTime = record.timestamp_s;
      el('overlay-video').currentTime = ordinal * (run.manifest.config?.stride ?? 1)
        / (run.manifest.video?.fps ?? state.source.info.fps);
    },
  });

  function error(error, field) {
    el('app-error').textContent = errorMessage(error);
    el('app-error').hidden = false;
    (field ? el(field) : el('app-error')).focus();
  }

  function controls() {
    const disabled = !state.token || state.busy || state.uncertain || ['queued', 'running'].includes(state.job?.status);
    el('source-fields').disabled = disabled;
    el('config-fields').disabled = disabled || state.sourceStale;
    el('run-analysis').disabled = disabled || state.sourceStale;
    el('run-analysis').textContent = ['queued', 'running'].includes(state.job?.status)
      ? 'Analysis running…' : state.uncertain ? 'Check job status' : state.busy ? 'Working…' : 'Run analysis';
    picker.refresh();
    el('movement-review-fields').disabled = disabled || state.sourceStale;
  }

  function readConfig() {
    const values = Object.fromEntries(Object.entries(CONFIG_FIELDS).map(([key, id]) => {
      const value = el(id).value;
      return [key, NUMERIC_FIELDS.has(key) ? (value.trim() ? Number(value) : NaN) : value];
    }));
    return { ...values, ...picker.getConfig(), fixed_camera: el('fixed-camera').checked };
  }

  async function acceptJob(job) {
    state.job = job;
    if (pollTimer !== null) clearTimer(pollTimer);
    pollTimer = null;
    el('resume-job').hidden = true;
    el('job-progress').hidden = false;
    if (Number.isFinite(job.progress)) el('job-progress').value = Math.max(0, Math.min(1, job.progress));
    else el('job-progress').removeAttribute('value');
    el('job-status').textContent = job.status === 'failed'
      ? `Analysis failed: ${job.error || job.message || 'No error detail was supplied.'}`
      : `${job.status} · ${job.message || 'Waiting for engine status.'}`;
    controls();
    if (['queued', 'running'].includes(job.status)) {
      pollTimer = setTimer(poll, 500);
    } else if (job.status === 'completed') {
      await loadResults();
    }
  }

  async function poll() {
    if (state.uncertain) return recoverSubmission();
    if (polling || !state.job) return;
    const owner = {};
    polling = owner;
    const id = state.job.id;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (polling === owner) polling = null;
      if (state.job?.id === id) await acceptJob(job);
    } catch (failure) {
      if (state.job?.id !== id) return;
      el('job-status').textContent = `Status connection interrupted: ${errorMessage(failure)} The job may still be running. Retry the status connection.`;
      el('resume-job').hidden = false;
    } finally {
      if (polling === owner) polling = null;
    }
  }

  async function recoverSubmission() {
    if (polling) return;
    const owner = {};
    polling = owner;
    try {
      const data = await api('/api/bootstrap');
      if (data.source?.id !== state.source?.id || data.source?.sha256 !== state.source?.sha256) {
        state.uncertain = false;
        state.job = null;
        invalidateSource(new Error('The browser session or source changed. Reload this page and inspect the source again.'));
        return;
      }
      state.token = data.csrf_token;
      state.uncertain = false;
      if (polling === owner) polling = null;
      el('app-error').hidden = true;
      el('resume-job').hidden = true;
      if (data.job) await acceptJob(data.job);
      else {
        state.job = null;
        el('job-status').textContent = 'No job was found for the inspected source. Review settings and run analysis again.';
        el('job-progress').hidden = true;
        controls();
      }
    } catch (failure) {
      el('job-status').textContent = `The analysis request could not be confirmed: ${errorMessage(failure)} Retry the status connection before submitting again.`;
      el('resume-job').hidden = false;
    } finally {
      if (polling === owner) polling = null;
    }
  }

  async function submit() {
    if (!state.token || state.busy || state.uncertain || ['queued', 'running'].includes(state.job?.status)) return;
    const config = readConfig();
    const invalid = validateRun(state.sourceStale ? null : state.source, el('permission').checked, config, state.limits.interval_s);
    if (invalid) return error(new Error(invalid.message), invalid.field);
    state.busy = true;
    el('app-error').hidden = true;
    el('job-status').textContent = 'Submitting analysis to the engine…';
    el('job-progress').hidden = false;
    el('job-progress').removeAttribute('value');
    controls();
    try {
      const job = await api('/api/jobs', {
        method: 'POST', json: { source_id: state.source.id, permission_confirmed: el('permission').checked, config },
      });
      await acceptJob(job);
    } catch (failure) {
      if (failure.status === 409 && /source.*chang|reload.*source|source.*reload/i.test(errorMessage(failure))) {
        invalidateSource(failure);
      }
      state.uncertain = !Number.isInteger(failure.status) || failure.status >= 500;
      el('job-status').textContent = state.uncertain
        ? 'The analysis request could not be confirmed. It may already be running. Retry the status connection before submitting again.'
        : 'Analysis was not started. Review the error and retry.';
      el('resume-job').hidden = !state.uncertain;
      el('job-progress').hidden = true;
      error(failure);
    } finally {
      state.busy = false;
      controls();
    }
  }

  function applyConfig(config) {
    for (const [key, id] of Object.entries(CONFIG_FIELDS)) {
      let value = config[key];
      if (key === 'end_s' && value == null && state.source) {
        value = Math.min(state.source.info.duration_s, (config.start_s || 0) + state.limits.interval_s);
      }
      el(id).value = value == null ? '' : String(value);
    }
    el('confidence-value').textContent = el('confidence').value;
    el('keypoint-confidence-value').textContent = el('keypoint-confidence').value;
    picker.configure(config);
    el('fixed-camera').checked = config.fixed_camera === true;
  }

  function savedWarning() {
    const saved = state.results?.job;
    el('saved-warning').hidden = !saved || (saved.source_id === state.source?.id
      && !settingsDiffer(saved.config, readConfig(), state.source.info.duration_s));
  }

  function releaseImage(id) {
    imageRequests.get(id)?.abort();
    imageRequests.delete(id);
    const url = imageURLs.get(id);
    if (url) revokeObjectURL(url);
    imageURLs.delete(id);
    el(id).hidden = true;
    el(id).removeAttribute('src');
    el(id).setAttribute('aria-busy', 'false');
  }

  function clearVideo(id) {
    el(id).pause();
    el(id).removeAttribute('src');
    el(id).load();
  }

  function clearResults() {
    resultRequest += 1;
    state.results = null;
    movementView.render(null);
    el('results').hidden = true;
    el('results-empty').hidden = false;
    el('results-empty').textContent = 'No completed run for this source.';
    el('reload-results').hidden = true;
    el('saved-warning').hidden = true;
    for (const id of ['provenance', 'raw-record', 'review-flags', 'result-id', 'result-count', 'selected-feature']) {
      el(id).textContent = '';
    }
    el('chart-content').replaceChildren();
    for (const id of ['original-frame', 'overlay-frame', 'contact-sheet']) releaseImage(id);
    for (const id of ['original-frame-error', 'overlay-frame-error', 'contact-sheet-error']) el(id).hidden = true;
    clearVideo('overlay-video');
    for (const link of doc.querySelectorAll('[data-artifact]')) link.removeAttribute('href');
  }

  function invalidateSource(failure) {
    state.sourceStale = true;
    el('fixed-camera').checked = false;
    el('permission').checked = false;
    el('rights-note').value = '';
    el('source-note').value = '';
    clearResults();
    picker.clear();
    releaseImage('source-frame-image');
    clearVideo('source-video');
    el('export-source-frame').removeAttribute('href');
    el('source-panel').hidden = true;
    el('source-empty').hidden = false;
    el('source-empty').textContent = 'The source changed. Inspect it again and confirm permission before another run.';
    el('source-status').textContent = 'Source changed; reinspection required.';
    error(failure);
    controls();
  }

  function svg(tag, attributes, text) {
    const node = doc.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    el('chart-content').append(node);
    return node;
  }

  function drawChart() {
    el('chart-content').replaceChildren();
    const result = state.results;
    if (!result) return;
    const track = result.features.tracks.find((id) => String(id) === el('track-select').value);
    const feature = result.features.features.find(({ name }) => name === el('feature-select').value);
    const samples = feature && track !== undefined
      ? trackTimeline(result.records, result.features.rows, track, feature.name) : [];
    const geometry = chartGeometry(samples);
    const { plot, xDomain, yDomain } = geometry;
    el('chart-title').textContent = feature ? `Track ${track ?? 'unavailable'} · ${feature.name}` : 'No feature evidence';
    el('chart-description').textContent = feature
      ? `${feature.name} in ${feature.unit} over original source time in seconds. Missing evidence is shown as gaps.`
      : 'No tracks or features are available for this completed run.';
    el('feature-caption').textContent = feature
      ? `X: source time (s) · Y: ${feature.name} (${feature.unit}). Missing evidence stays missing.`
      : 'No track / feature evidence is available. Missing evidence is not zero.';
    el('chart-empty').hidden = geometry.segments.length > 0;
    if (xDomain) {
      for (let tick = 0; tick <= 4; tick += 1) {
        const fraction = tick / 4;
        const x = plot.left + fraction * (plot.right - plot.left);
        svg('line', { x1: x, x2: x, y1: plot.top, y2: plot.bottom, class: 'chart-grid' });
        svg('text', { x, y: 236, 'text-anchor': 'middle' }, format(xDomain[0] + fraction * (xDomain[1] - xDomain[0]), 2));
      }
    }
    if (yDomain) {
      for (let tick = 0; tick <= 4; tick += 1) {
        const fraction = tick / 4;
        const y = plot.bottom - fraction * (plot.bottom - plot.top);
        svg('line', { x1: plot.left, x2: plot.right, y1: y, y2: y, class: 'chart-grid' });
        svg('text', { x: plot.left - 8, y: y + 4, 'text-anchor': 'end' }, format(yDomain[0] + fraction * (yDomain[1] - yDomain[0]), 2));
      }
    }
    svg('text', { x: 425, y: 254, 'text-anchor': 'middle' }, 'Source time (s)');
    for (const segment of geometry.segments) {
      svg('polyline', { points: segment.map(({ x, y }) => `${x},${y}`).join(' '), class: 'chart-line' });
      if (segment.length === 1) svg('circle', { cx: segment[0].x, cy: segment[0].y, r: 3, class: 'chart-point' });
    }
    const selected = samples[reviewOrdinal];
    if (selected && xDomain && Number.isFinite(selected.timestamp_s)) {
      const x = plot.left + (selected.timestamp_s - xDomain[0]) / (xDomain[1] - xDomain[0]) * (plot.right - plot.left);
      svg('line', { x1: x, x2: x, y1: plot.top, y2: plot.bottom, class: 'chart-cursor' });
    }
    el('selected-feature').textContent = selected && feature
      ? `Source frame ${selected.frame_index} · ${format(selected.timestamp_s)} s · ${Number.isFinite(selected.value) ? `${format(selected.value)} ${feature.unit}` : 'Missing evidence'}`
      : 'No selected track / feature evidence.';
  }

  async function reviewFrame(value = 0) {
    if (!state.results?.records.length) return;
    const { job, records } = state.results;
    reviewOrdinal = Math.max(0, Math.min(records.length - 1, Math.trunc(Number(value)) || 0));
    const record = records[reviewOrdinal];
    el('review-frame').value = String(reviewOrdinal);
    el('review-frame-number').value = String(reviewOrdinal);
    el('review-prev').disabled = reviewOrdinal === 0;
    el('review-next').disabled = reviewOrdinal === records.length - 1;
    const label = `source frame ${record.frame_index} · ${format(record.timestamp_s)} s`;
    el('review-frame-value').textContent = `${reviewOrdinal + 1} / ${records.length} · ${label}`;
    el('original-caption').textContent = `Original · ${label}`;
    el('overlay-caption').textContent = `Overlay · ${label}`;
    el('original-frame').alt = `Original ${label}`;
    el('overlay-frame').alt = `Pose overlay ${label}`;
    el('raw-record').textContent = JSON.stringify(record, null, 2);
    el('review-flags').textContent = record.flags?.length
      ? `Review flags: ${record.flags.join(' · ')}` : 'No heuristic review flags for this record.';
    drawChart();
    const path = `/api/jobs/${encodeURIComponent(job.id)}/frames/${reviewOrdinal}`;
    await Promise.all([
      loadImage('original-frame', `${path}?view=original`, 'original-frame-error'),
      loadImage('overlay-frame', `${path}?view=overlay`, 'overlay-frame-error'),
    ]);
  }

  function populateSelect(id, options) {
    const select = el(id);
    select.replaceChildren();
    for (const [value, label] of options) {
      const option = doc.createElement('option');
      option.value = String(value);
      option.textContent = label;
      select.append(option);
    }
    select.value = options.length ? String(options[0][0]) : '';
    select.disabled = options.length === 0;
  }

  async function loadResults() {
    if (state.sourceStale || state.job?.status !== 'completed') return;
    const job = state.job;
    const generation = ++resultRequest;
    const path = `/api/jobs/${encodeURIComponent(job.id)}`;
    el('reload-results').hidden = true;
    el('results-empty').textContent = 'Loading completed run…';
    try {
      const [manifest, records, features] = await Promise.all([
        api(`${path}/manifest`), api(`${path}/records`), api(`${path}/features`),
      ]);
      if (generation !== resultRequest || state.job?.id !== job.id || state.source?.id !== job.source_id) return;
      state.results = { job, manifest, records, features };
      movementView.render(state.results);
      el('app-error').hidden = true;
      el('results').hidden = false;
      el('results-empty').hidden = true;
      el('result-count').textContent = `${records.length} processed frames`;
      el('result-id').textContent = `Saved job ${job.id}`;
      el('provenance').textContent = JSON.stringify(manifest, null, 2);
      el('overlay-video').src = `${path}/artifacts/overlay.mp4`;
      el('overlay-video').load();
      for (const link of doc.querySelectorAll('[data-artifact]')) {
        link.href = `${path}/artifacts/${encodeURIComponent(link.dataset.artifact)}?download=true`;
      }
      const fighters = new Map(Object.entries(manifest.selection?.fighter_tracks || {})
        .map(([label, id]) => [id, label]));
      populateSelect('track-select', features.tracks.map((id) => [id,
        fighters.has(id) ? `Fighter ${fighters.get(id)} · Track ${id}` : `Track ${id}`]));
      populateSelect('feature-select', features.features.map(({ name, unit }) => [name, `${name} (${unit})`]));
      for (const id of ['review-frame', 'review-frame-number']) el(id).max = String(Math.max(0, records.length - 1));
      el('review-controls').hidden = records.length === 0;
      el('review-empty').hidden = records.length > 0;
      reviewOrdinal = 0;
      savedWarning();
      drawChart();
      await Promise.all([
        loadImage('contact-sheet', `${path}/artifacts/contact_sheet.jpg`, 'contact-sheet-error'),
        reviewFrame(0),
      ]);
    } catch (failure) {
      if (generation !== resultRequest || state.job?.id !== job.id) return;
      el('results-empty').textContent = 'The run completed, but its artifacts could not be loaded. Retry loading the completed run.';
      el('reload-results').hidden = false;
      error(failure);
    }
  }

  async function loadImage(id, path, errorID) {
    const image = el(id);
    imageRequests.get(id)?.abort();
    const task = new AbortController();
    imageRequests.set(id, task);
    image.hidden = true;
    image.setAttribute('aria-busy', 'true');
    el(errorID).hidden = true;
    try {
      const blob = await api(path, { responseType: 'blob', signal: task.signal });
      if (imageRequests.get(id) !== task) return;
      const previous = imageURLs.get(id);
      const url = createObjectURL(blob);
      imageURLs.set(id, url);
      if (previous) revokeObjectURL(previous);
      image.src = url;
      image.hidden = false;
    } catch (failure) {
      if (imageRequests.get(id) !== task || task.signal.aborted) return;
      if (failure.status === 409 && (path.includes('/sources/') || path.includes('view=original'))) {
        invalidateSource(failure);
        return;
      }
      el(errorID).textContent = errorMessage(failure);
      el(errorID).hidden = false;
    } finally {
      if (imageRequests.get(id) === task) image.setAttribute('aria-busy', 'false');
    }
  }

  async function sourceFrame(value = 0) {
    if (!state.source || state.sourceStale) return;
    const sourceID = state.source.id;
    const maximum = Math.max(0, state.source.info.frame_count - 1);
    const index = Math.max(0, Math.min(maximum, Math.trunc(Number(value)) || 0));
    el('source-frame').value = String(index);
    el('source-frame-number').value = String(index);
    el('source-frame-value').textContent = String(index);
    el('source-prev').disabled = index === 0;
    el('source-next').disabled = index === maximum;
    el('source-frame-caption').textContent = `Original source frame ${index} · ${format(index / state.source.info.fps)} s`;
    el('source-frame-image').alt = `Original source frame ${index}`;
    const path = `${sourcePath()}/frames/${index}`;
    el('export-source-frame').href = `${path}?download=true`;
    el('export-source-frame').download = `source-frame-${String(index).padStart(6, '0')}.png`;
    await loadImage('source-frame-image', path, 'source-frame-error');
    if (state.source?.id === sourceID && Number(el('source-frame').value) === index) {
      picker.frameChanged(index);
    }
  }

  async function showSource(source) {
    state.source = source;
    el('source-empty').hidden = Boolean(source);
    el('source-panel').hidden = !source;
    if (!source) return;
    el('source-name').textContent = source.name;
    el('source-kind').textContent = source.kind === 'local' ? 'Host-local path' : 'Upload';
    el('source-status').textContent = `Inspected: ${source.name}`;
    el('source-video').src = `${sourcePath()}/video`;
    el('source-video').load();
    el('source-metadata').replaceChildren();
    for (const [label, value] of [
      ['Duration', `${format(source.info.duration_s)} s`],
      ['Frame rate', `${format(source.info.fps)} fps`],
      ['Dimensions', `${source.info.width} × ${source.info.height}`],
      ['Source frames', source.info.frame_count], ['SHA-256', source.sha256],
    ]) {
      const cell = doc.createElement('div');
      const term = doc.createElement('dt');
      const detail = doc.createElement('dd');
      term.textContent = label;
      detail.textContent = String(value);
      cell.append(term, detail);
      el('source-metadata').append(cell);
    }
    for (const id of ['source-frame', 'source-frame-number']) {
      el(id).max = String(Math.max(0, source.info.frame_count - 1));
    }
    for (const id of ['start-s', 'end-s']) el(id).max = String(source.info.duration_s);
    await sourceFrame(0);
  }

  async function start() {
    try {
      const data = await api('/api/bootstrap');
      state.token = data.csrf_token;
      state.limits = data.limits;
      state.defaults = data.defaults;
      state.source = data.source;
      state.job = data.job;
      applyConfig({ ...data.defaults, ...data.job?.config });
      el('upload-limit').textContent = `MP4 / MOV / AVI / MKV / M4V / WebM · maximum ${format(data.limits.upload_bytes / (1024 * 1024), 0)} MiB`;
      el('interval-help').textContent = `Use a nonempty interval within the source, at most ${data.limits.interval_s} seconds.`;
      el('connection-status').textContent = 'Engine connected';
      controls();
      await showSource(data.source);
      if (data.job && state.job?.id === data.job.id && state.source?.id === data.job.source_id) {
        await acceptJob(data.job);
      }
    } catch (failure) {
      el('connection-status').textContent = 'Engine unavailable';
      error(failure);
    }
  }

  async function inspect(kind) {
    if (!state.token || state.busy || state.uncertain || ['queued', 'running'].includes(state.job?.status)) return;
    let path;
    let options;
    if (kind === 'upload') {
      const file = el('upload-file').files?.[0];
      if (!file) return error(new Error('Choose a video file to inspect.'), 'upload-file');
      if (!/\.(mp4|mov|avi|mkv|m4v|webm)$/i.test(file.name)) {
        return error(new Error('Choose an MP4, MOV, AVI, MKV, M4V or WebM video.'), 'upload-file');
      }
      if (!file.size || file.size > state.limits.upload_bytes) {
        return error(new Error('The upload must be nonempty and within the displayed upload limit. Use a local path for a larger host-local video.'), 'upload-file');
      }
      path = `/api/sources/upload?filename=${encodeURIComponent(file.name)}`;
      options = { method: 'POST', body: file, headers: { 'Content-Type': 'application/octet-stream' } };
    } else {
      const value = el('local-path').value;
      if (!value.trim()) return error(new Error('Enter a video path on the app host.'), 'local-path');
      path = '/api/sources/path';
      options = { method: 'POST', json: { path: value } };
    }
    state.busy = true;
    el('app-error').hidden = true;
    el('source-status').textContent = kind === 'upload'
      ? 'Uploading and inspecting on the app host…' : 'Inspecting host-local video…';
    controls();
    try {
      const source = await api(path, options);
      const changed = state.sourceStale || state.source?.id !== source.id || state.source?.sha256 !== source.sha256;
      state.sourceStale = false;
      if (changed) {
        el('fixed-camera').checked = false;
        state.job = null;
        clearResults();
        picker.clear();
        el('permission').checked = false;
        el('rights-note').value = '';
        el('source-note').value = '';
        el('start-s').value = '0';
        el('end-s').value = String(Math.min(source.info.duration_s, state.limits.interval_s));

        el('job-status').textContent = 'Source inspected. Confirm permission, then run analysis.';
        el('job-progress').hidden = true;
      }
      await showSource(source);
    } catch (failure) {
      el('source-status').textContent = 'Inspection failed. The previous inspected selection is retained.';
      error(failure);
    } finally {
      state.busy = false;
      controls();
    }
  }

  el('config-form').addEventListener('submit', (event) => {
    event.preventDefault();
    return submit();
  });
  el('resume-job').addEventListener('click', poll);
  el('reload-results').addEventListener('click', loadResults);
  for (const id of ['source-frame', 'source-frame-number']) {
    el(id).addEventListener('input', () => sourceFrame(el(id).value));
  }
  el('source-prev').addEventListener('click', () => sourceFrame(Number(el('source-frame').value) - 1));
  el('source-next').addEventListener('click', () => sourceFrame(Number(el('source-frame').value) + 1));
  for (const trigger of doc.querySelectorAll('[data-dialog]')) {
    const dialog = el(trigger.dataset.dialog);
    trigger.addEventListener('click', () => dialog.showModal());
    dialog.addEventListener('close', () => trigger.focus());
  }
  for (const button of doc.querySelectorAll('[data-close]')) {
    button.addEventListener('click', () => el(button.dataset.close).close());
  }
  el('fixed-camera').addEventListener('change', savedWarning);
  for (const id of Object.values(CONFIG_FIELDS)) {
    el(id).addEventListener('input', () => {
      el('confidence-value').textContent = el('confidence').value;
      el('keypoint-confidence-value').textContent = el('keypoint-confidence').value;
      savedWarning();
    });
    el(id).addEventListener('change', savedWarning);
  }
  for (const id of ['review-frame', 'review-frame-number']) {
    el(id).addEventListener('input', () => reviewFrame(el(id).value));
  }
  el('review-prev').addEventListener('click', () => reviewFrame(reviewOrdinal - 1));
  el('review-next').addEventListener('click', () => reviewFrame(reviewOrdinal + 1));
  for (const id of ['track-select', 'feature-select']) el(id).addEventListener('change', drawChart);
  el('upload-form').addEventListener('submit', (event) => {
    event.preventDefault();
    return inspect('upload');
  });
  el('local-form').addEventListener('submit', (event) => {
    event.preventDefault();
    return inspect('local');
  });
  for (const mode of ['upload', 'local']) {
    el(`mode-${mode}`).addEventListener('change', () => {
      el('upload-mode').hidden = mode !== 'upload';
      el('local-mode').hidden = mode !== 'local';
    });
  }

  return { start };
}

if (globalThis.document) {
  createController().start();
}
