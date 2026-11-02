//manual source-pixel seeds, never a claim of athlete recognition
export function imagePoint(rect, info, x, y, clamp = false) {
  const scale = Math.min(rect.width / info.width, rect.height / info.height);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const left = rect.left + (rect.width - info.width * scale) / 2;
  const top = rect.top + (rect.height - info.height * scale) / 2;
  const point = [(x - left) / scale, (y - top) / scale];
  if (!clamp && (point[0] < 0 || point[1] < 0 || point[0] > info.width || point[1] > info.height)) return null;
  return [Math.max(0, Math.min(info.width, point[0])), Math.max(0, Math.min(info.height, point[1]))];
}

export function orderedBox(first, second) {
  const box = [Math.min(first[0], second[0]), Math.min(first[1], second[1]),
    Math.max(first[0], second[0]), Math.max(first[1], second[1])];
  return box[2] > box[0] && box[3] > box[1] ? box : null;
}

const reference = (source) => source ? `${source.id}:${source.sha256}` : null;

export class FighterBoxes {
  constructor() { this.clear(); }
  clear() { this.source = null; this.frame = null; this.boxes = [null, null]; }
  set(role, box, source, frame) {
    if (![0, 1].includes(role) || !source || !Number.isInteger(frame) || frame < 0
        || !Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)
        || box[0] < 0 || box[1] < 0 || box[2] > source.info.width || box[3] > source.info.height
        || box[2] <= box[0] || box[3] <= box[1]) {
      throw new Error('Use finite, ordered fighter-box coordinates within the source bounds.');
    }
    if (this.source !== reference(source) || this.frame !== frame) this.clear();
    this.source = reference(source);
    this.frame = frame;
    this.boxes[role] = [...box];
  }
  config(mode, source, count) {
    const valid = mode === 'manual' && this.source === reference(source);
    const boxes = valid ? this.boxes.slice(0, count) : [];
    return { selection_mode: mode,
      fighter_boxes: valid && boxes.length === count && boxes.every(Boolean) ? boxes.map((box) => [...box]) : null,
      fighter_seed_frame: valid ? this.frame : null };
  }
}

export function createFighterPicker({ document: doc, getSource, getCount, getStartFrame,
  showFrame, onChange = () => {} }) {
  const el = (id) => doc.getElementById(id);
  const state = new FighterBoxes();
  const layer = el('fighter-box-layer');
  const fields = ['box-x1', 'box-y1', 'box-x2', 'box-y2'];
  let active = 0;
  let editing = false;
  let draft = null;
  let generation = 0;
  const mode = () => el('selection-mode').value || 'automatic';
  const currentFrame = () => Number(el('source-frame').value);
  const usable = () => getSource() && !el('config-fields').disabled && mode() === 'manual';

  function node(tag, attributes, text) {
    const element = doc.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    if (text !== undefined) element.textContent = text;
    layer.append(element);
  }

  function render(updateFields = true) {
    const source = getSource();
    const manual = mode() === 'manual';
    const sameSource = source && state.source === reference(source);
    const show = source && !el('source-frame-image').hidden && manual
      && (editing || (sameSource && state.frame === currentFrame() && state.boxes.some(Boolean)));
    if (show) layer.removeAttribute('hidden');
    else layer.setAttribute('hidden', '');
    layer.setAttribute('data-editing', String(Boolean(editing && usable())));
    layer.replaceChildren();
    el('mark-fighter-b').hidden = getCount() === 1;
    el('fighter-coordinate-role').textContent = `Editing Fighter ${'AB'[active]} in original source pixels.`;
    for (const [index, id] of ['mark-fighter-a', 'mark-fighter-b'].entries()) {
      el(id).setAttribute('aria-pressed', String(editing && active === index));
    }
    const boxes = sameSource ? state.boxes : [null, null];
    if (updateFields) fields.forEach((id, index) => { el(id).value = boxes[active] ? String(boxes[active][index]) : ''; });
    const status = manual
      ? `${getCount() === 1 ? 'Fighter A' : 'Fighters A / B'}   ${boxes.slice(0, getCount()).map((box, index) => `${'AB'[index]}: ${box ? 'selected' : 'not selected'}`).join('   ')}${sameSource ? `   source frame ${state.frame}` : ''}`
      : 'Automatic cap: keeps the highest-confidence people, which may include spectators. No A/B identity assignment.';
    el('fighter-selection-status').textContent = status;
    el('fighter-drawing-status').textContent = editing
      ? `Drag a box around Fighter ${'AB'[active]} on source frame ${currentFrame()}. Escape cancels drawing. Use coordinate editing if needed.`
      : manual ? 'Mark A / B selects the first frame of the analysis interval. Boxes are not a static crop.' : '';
    if (!show) return;
    layer.setAttribute('viewBox', `0 0 ${source.info.width} ${source.info.height}`);
    for (let index = 0; index < getCount(); index += 1) {
      const box = draft && active === index ? orderedBox(draft.start, draft.end) : boxes[index];
      if (!box) continue;
      node('rect', { x: box[0], y: box[1], width: box[2] - box[0], height: box[3] - box[1],
        class: 'fighter-box', 'data-fighter': 'AB'[index] });
      node('text', { x: box[0] + 5, y: Math.max(20, box[1] + 20), class: 'fighter-box-label' }, `Fighter ${'AB'[index]}`);
    }
  }

  function clear() {
    generation += 1;
    state.clear();
    editing = false;
    draft = null;
    active = 0;
    render();
    onChange();
  }

  async function prepare(role) {
    if (!usable() || role >= getCount()) return false;
    const source = getSource();
    const frame = getStartFrame();
    if (!Number.isInteger(frame) || frame < 0 || frame >= source.info.frame_count) {
      el('fighter-selection-status').textContent = 'Choose a valid clip start before marking fighters.';
      return false;
    }
    const version = ++generation;
    editing = false;
    await showFrame(frame);
    try { await el('source-frame-image').decode?.(); } catch { return false; }
    if (version !== generation || !usable() || reference(getSource()) !== reference(source)
        || frame !== getStartFrame() || currentFrame() !== frame || el('source-frame-image').hidden) return false;
    if (state.source !== reference(source) || state.frame !== frame) state.clear();
    active = role;
    draft = null;
    editing = true;
    render();
    layer.focus();
    return true;
  }

  function store(box) {
    state.set(active, box, getSource(), getStartFrame());
    if (active === 0 && getCount() === 2 && !state.boxes[1]) active = 1;
    else editing = false;
    draft = null;
    render();
    onChange();
  }

  for (const [role, id] of ['mark-fighter-a', 'mark-fighter-b'].entries()) {
    el(id).addEventListener('click', () => prepare(role));
  }
  el('clear-fighters').addEventListener('click', clear);
  el('selection-mode').addEventListener('change', clear);
  el('expected-people').addEventListener('change', clear);
  el('start-s').addEventListener('input', clear);
  el('start-s').addEventListener('change', clear);
  el('apply-fighter-box').addEventListener('click', async () => {
    const values = fields.map((id) => el(id).value.trim());
    if (values.some((value) => !value)) {
      el('fighter-selection-status').textContent = 'Enter all four source-pixel coordinates.';
      return;
    }
    const box = values.map(Number);
    if (!await prepare(active)) return;
    try { store(box); } catch (error) { el('fighter-selection-status').textContent = error.message; }
  });
  layer.addEventListener('pointerdown', (event) => {
    if (!editing || !usable() || event.button !== 0 || currentFrame() !== getStartFrame()) return;
    const point = imagePoint(el('source-frame-image').getBoundingClientRect(), getSource().info,
      event.clientX, event.clientY);
    if (!point) return;
    event.preventDefault();
    draft = { start: point, end: point, pointer: event.pointerId };
    layer.setPointerCapture?.(event.pointerId);
  });
  layer.addEventListener('pointermove', (event) => {
    if (!draft || draft.pointer !== event.pointerId || !usable()) return;
    draft.end = imagePoint(el('source-frame-image').getBoundingClientRect(), getSource().info,
      event.clientX, event.clientY, true);
    render(false);
  });
  layer.addEventListener('pointerup', (event) => {
    if (!draft || draft.pointer !== event.pointerId || !usable()) return;
    const end = imagePoint(el('source-frame-image').getBoundingClientRect(), getSource().info,
      event.clientX, event.clientY, true);
    const box = orderedBox(draft.start, end);
    draft = null;
    if (box) store(box);
    else render();
  });
  layer.addEventListener('pointercancel', () => { draft = null; render(); });
  layer.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { editing = false; draft = null; render(); }
  });

  return {
    clear,
    refresh: render,
    configure(config) {
      state.clear();
      generation += 1;
      editing = false;
      draft = null;
      el('selection-mode').value = config.selection_mode ?? 'automatic';
      if (config.selection_mode === 'manual' && Array.isArray(config.fighter_boxes) && getSource()) {
        config.fighter_boxes.forEach((box, index) => state.set(index, box, getSource(), config.fighter_seed_frame));
      }
      render();
    },
    getConfig: () => state.config(mode(), getSource(), getCount()),
    frameChanged(index) {
      if (index !== getStartFrame()) { generation += 1; editing = false; draft = null; }
      render();
    },
  };
}
