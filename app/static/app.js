/* T-Tracer - frontend.
 *
 * Interaction model, from the reviewer: click a candidate to mark it shippable, star
 * one as the favourite, and the favourite is what gets written to the chosen
 * folder. Marking several is deliberate - across three calibration batches the
 * average image had ~5 shippable candidates, and forcing a single choice threw
 * that away. */

const $ = (s) => document.querySelector(s);

/* ---------------- session token (item 8) ----------------
 * main.py puts the launch token in the page URL because there is nowhere
 * else for a freshly loaded static page to learn it from - it is generated
 * fresh per launch and never persisted. Read once here and attached to every
 * API call from then on. */
const TOKEN = new URLSearchParams(location.search).get('token') || '';
function apiFetch(url, opts = {}) {
  const headers = new Headers(opts.headers || {});
  if (TOKEN) headers.set('X-T-Tracer-Token', TOKEN);
  return fetch(url, { ...opts, headers });
}
// <img src> can't carry a header, so the two read-only image endpoints
// (preview and source) take the token as a query param instead. Applied once,
// where the URLs first arrive from the server (loadResults), so every
// consumer of img.source / candidate.preview - cards, the viewer - gets a
// working URL for free without needing to know about tokens at all.
function withToken(url) {
  if (!url || !TOKEN) return url;
  return url + (url.includes('?') ? '&' : '?') + 'tt_token=' + encodeURIComponent(TOKEN);
}
// `images` ACCUMULATES across drops. Each entry carries the job it came from,
// because preview URLs and the save endpoint are both job-scoped. The first
// version replaced the list on every upload, so adding a second logo silently
// erased the first one along with its picks.
const state = { jobId: null, images: [], sel: {}, fav: {}, failed: {},
                touched: {}, notes: {} };

/* ---------------- destination folder ----------------
 * Kept SERVER-side rather than in localStorage, and that is the entire fix for
 * "the folder is forgotten every time I open it".
 *
 * The old code wrote localStorage correctly and it still never survived a
 * restart: main.py binds a FREE PORT on every launch, localStorage is keyed by
 * ORIGIN, and http://127.0.0.1:41337 is a different origin from
 * http://127.0.0.1:39112. Every start was a fresh, empty store. A settings
 * file next to the work directory does not care which port the app got. */
let settings = { dest: '', share_stats: false, role: 'user', live_view: false,
                 live_keep: false, dev_mode: false };

async function loadSettings() {
  try {
    settings = await (await apiFetch('/api/settings')).json();
    $('#dest').value = settings.dest || '';
  } catch (_) { /* defaults are fine */ }
  refreshAction();
}

async function saveSettings(patch) {
  Object.assign(settings, patch);
  try {
    await apiFetch('/api/settings', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  } catch (_) { /* a lost preference is not worth an error */ }
  // Dev mode owns a control in the action bar, so a flip of that switch has to
  // repaint it - otherwise Save picks only appears after the next unrelated
  // click. (Cards keep the Exclude/Delete buttons they were BUILT with; those
  // need a re-trace or a reopen, which the Settings copy says.)
  refreshAction();
}

$('#dest').addEventListener('change', () => {
  saveSettings({ dest: $('#dest').value.trim() });
  refreshAction();
});

$('#browse').addEventListener('click', async () => {
  // Two routes to a native dialog. pywebview's bridge when that is the window
  // backend; otherwise the server shells out to kdialog/zenity, so Browse
  // behaves the same inside a Chrome app window.
  try {
    if (window.pywebview?.api?.pick_folder) {
      const p = await window.pywebview.api.pick_folder();
      if (p) setDest(p);
      return;
    }
    const r = await apiFetch('/api/pick-folder');
    const j = await r.json();
    if (j.path) setDest(j.path);
    else if (j.unavailable) {
      toast('No folder dialog on this system — type or paste a path instead.');
      $('#dest').focus();
    }
  } catch (_) {
    toast('Could not open the folder picker — type or paste a path instead.');
    $('#dest').focus();
  }
});

function setDest(p) {
  $('#dest').value = p;
  $('#dest').dispatchEvent(new Event('change'));
}

/* ---------------- adding images ---------------- */
const drop = $('#drop');
$('#browseImages').addEventListener('click', (e) => { e.stopPropagation(); $('#file').click(); });
drop.addEventListener('click', () => $('#file').click());
drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file').click(); }
});
$('#file').addEventListener('change', (e) => { if (e.target.files.length) upload(e.target.files); });

['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('hot'); }));
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('hot'); }));
drop.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files);
});

async function upload(fileList) {
  const fd = new FormData();
  [...fileList].forEach((f) => fd.append('files', f));
  $('#progress').hidden = false;
  $('#barfill').style.width = '2%';
  $('#progresstext').textContent = 'Uploading…';
  try {
    const r = await apiFetch('/api/jobs', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const j = await r.json();
    state.jobId = j.job_id;
    if (settings.live_view) liveStart(j.job_id);
    if (j.rejected?.length) toast(`Skipped ${j.rejected.length} non-image file(s).`, true);
    if (j.oversized?.length) toast(`Skipped ${j.oversized.length} file(s) over the upload size limit.`, true);
    // Distinct from "oversized" - a write failure (e.g. the disk is full) is
    // not a limit the user can fix by picking a smaller file.
    if (j.failed?.length) toast(`Could not save ${j.failed.length} file(s) - check disk space.`, true);
    poll(j.job_id);
  } catch (err) {
    $('#progress').hidden = true;
    toast(String(err.message || err), true);
  }
}

/* ---------------- progress ---------------- */
async function poll(jobId) {
  const r = await apiFetch(`/api/jobs/${jobId}`);
  const j = await r.json();
  const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
  $('#barfill').style.width = `${Math.max(pct, 3)}%`;
  $('#progresstext').textContent =
    j.status === 'ready'  ? `Traced ${j.total} image${j.total === 1 ? '' : 's'}`
  : j.status === 'queued' ? `Waiting for the current batch to finish…`
                          : `Tracing ${j.done} of ${j.total}…`;
  // 'queued' must keep polling too. Only one batch traces at a time, so a
  // second submission sits in 'queued' until the first finishes - and a poll
  // that stopped there would leave it looking hung forever.
  if (j.status === 'tracing' || j.status === 'queued')
    return setTimeout(() => poll(jobId), 700);
  if (j.errors?.length) toast(j.errors[0], true);
  setTimeout(() => { $('#progress').hidden = true; }, 900);
  loadResults(jobId);
}

/* ---------------- calibration intake ----------------
 * `_private/new to test/` is a queue of artwork waiting for a verdict. The
 * server derives what is left (judged stems are out, stems already in a job
 * are out), so this panel only has to show the number and ask for a batch.
 * Hidden entirely when there is no intake folder - a customer install has
 * none, and an empty control is worse than no control. */
async function loadIntake() {
  let j;
  try {
    j = await (await apiFetch('/api/intake')).json();
  } catch (_) { return; }
  const box = $('#intake');
  if (!j.exists || (!j.remaining && !j.in_flight)) { box.hidden = true; return; }
  box.hidden = false;
  $('#intake-count').textContent = j.remaining
    ? `${j.remaining} still to judge, ${j.judged} done of ${j.total}`
    : `Nothing left to judge - ${j.in_flight} waiting on a verdict`;
  $('#intake-go').disabled = !j.remaining;
}

$('#intake-go').addEventListener('click', async () => {
  const btn = $('#intake-go');
  btn.disabled = true;
  $('#progress').hidden = false;
  $('#barfill').style.width = '2%';
  $('#progresstext').textContent = 'Loading the next batch…';
  try {
    const r = await apiFetch('/api/intake/next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count: Number($('#intake-n').value) }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    state.jobId = j.job_id;
    if (settings.live_view) liveStart(j.job_id);
    if (j.failed?.length) toast(`Could not read ${j.failed.length} file(s).`, true);
    poll(j.job_id);
  } catch (err) {
    $('#progress').hidden = true;
    toast(String(err.message || err), true);
  }
  loadIntake();
});

/* ---------------- live view (debug, off by default) ----------------
 * Polls /api/jobs/{id}/live at ~3/s while the job traces. The server holds
 * ONE frame per (image, strategy, step) - the latest - and encodes on
 * request, so a slow browser just sees fewer frames and the tracer never
 * waits on it. Each tile is one strategy; its image is that strategy's most
 * recent step. Frames that did not change (same seq) are not re-decoded. */
const live = { jobId: null, seen: new Map(), frames: new Map(), since: 0, timer: null };

function liveStart(jobId) {
  live.jobId = jobId;
  live.seen.clear();
  live.frames.clear();
  live.since = 0;
  $('#live-grid').innerHTML = '';
  $('#live-big').hidden = true;
  $('#live-actions').hidden = true;
  $('#live').hidden = false;
  livePoll();
}

async function livePoll() {
  if (!live.jobId) return;
  let j;
  try {
    j = await (await apiFetch(`/api/jobs/${live.jobId}/live?since=${live.since}`)).json();
  } catch (_) { return; }
  if (!j.enabled) { $('#live').hidden = true; live.jobId = null; return; }
  // The server sends only frames newer than `since`; merge into what we hold.
  for (const f of j.frames) live.frames.set(`${f.stem}|${f.strategy}|${f.step}`, f);
  live.since = j.seq || live.since;
  j.frames = [...live.frames.values()];
  // One BIG tile: the strategy currently running, showing its newest frame
  // as it changes (about 3/s). When a strategy reaches 'scored' it drops
  // into the small grid and waits; the next one takes the big slot.
  const grid = $('#live-grid');
  const big = $('#live-big');
  const done = new Set(j.frames.filter((f) => f.step === 'scored').map((f) => `${f.stem}|${f.strategy}`));
  let current = null;
  for (const f of j.frames) {
    const key = `${f.stem}|${f.strategy}`;
    if (done.has(key)) {
      let tile = grid.querySelector(`[data-key="${CSS.escape(key)}"]`);
      if (!tile) {
        tile = document.createElement('div');
        tile.className = 'live-tile';
        tile.dataset.key = key;
        tile.innerHTML = `<img alt=""><span class="meta"><b></b> <span class="step"></span></span>`;
        tile.querySelector('b').textContent = f.strategy;
        grid.appendChild(tile);
      }
      if (f.step !== 'scored' || live.seen.get(key) === f.seq) continue;
      live.seen.set(key, f.seq);
      tile.querySelector('img').src = `data:image/png;base64,${f.png}`;
      tile.querySelector('.step').textContent = 'done';
    } else if (!current || f.seq > current.seq) {
      current = f;
    }
  }
  if (current) {
    const key = `big|${current.stem}|${current.strategy}`;
    big.hidden = false;
    if (live.seen.get(key) !== current.seq) {
      live.seen.set(key, current.seq);
      big.querySelector('img').src = `data:image/png;base64,${current.png}`;
      $('#live-big-name').textContent = current.strategy;
      $('#live-big-step').textContent = current.step;
    }
  } else {
    big.hidden = true;
  }
  const running = j.status === 'tracing' || j.status === 'queued';
  if (running) {
    live.timer = setTimeout(livePoll, 333);
    return;
  }
  // Done. Retained frames are the user's call: save or trash, never automatic.
  if (j.keep && j.retained > 0) {
    $('#live-actions').hidden = false;
  } else {
    await apiFetch(`/api/jobs/${live.jobId}/live/discard`, { method: 'POST' }).catch(() => {});
    setTimeout(() => { $('#live').hidden = true; }, 1500);
    live.jobId = null;
  }
}

$('#live-save').addEventListener('click', async () => {
  if (!live.jobId) return;
  try {
    const r = await apiFetch(`/api/jobs/${live.jobId}/live/save`, { method: 'POST' });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    toast(`Saved ${j.saved} snapshot${j.saved === 1 ? '' : 's'} to ${j.dir}`);
  } catch (err) { toast(String(err.message || err), true); }
  $('#live').hidden = true;
  live.jobId = null;
});
$('#live-discard').addEventListener('click', async () => {
  if (!live.jobId) return;
  await apiFetch(`/api/jobs/${live.jobId}/live/discard`, { method: 'POST' }).catch(() => {});
  $('#live').hidden = true;
  live.jobId = null;
});

/* ---------------- results ---------------- */
async function loadResults(jobId) {
  const r = await apiFetch(`/api/jobs/${jobId}/results`);
  const fresh = (await r.json()).images;
  const box = $('#results');
  box.hidden = false;

  fresh.forEach((img) => {
    img.jobId = jobId;
    // Preview/source URLs are loaded via <img src>, which cannot carry the
    // auth header apiFetch uses - stamp the token on as a query param here,
    // once, so every consumer downstream (cards, the full-size viewer) just
    // uses img.source / candidate.preview and gets a working URL for free.
    img.source = withToken(img.source);
    img.candidates.forEach((c) => { c.preview = withToken(c.preview); });
    // A repeat of the same filename would collide on `stem` as the key, so it
    // gets a display key that stays unique across drops. The saved SVG keeps
    // the plain name; the server handles that collision separately.
    img.key = state.images.some((x) => x.stem === img.stem)
      ? `${img.stem}#${jobId.slice(0, 4)}` : img.stem;
    // Sensible default: the top-ranked candidate starts starred. Across 26
    // judged images the top-scored one was shippable 81% of the time, so this
    // is usually right and always one click from being changed.
    if (img.candidates.length) {
      state.fav[img.key] = img.candidates[0].name;
      state.sel[img.key] = new Set([img.candidates[0].name]);
    }
    state.images.push(img);
    box.appendChild(cardFor(img));
    // Paint immediately. The pre-starred default was being set in state but
    // never rendered, so a freshly traced image showed no star, no highlighted
    // tile and no "1 shippable" line until the user clicked something - the
    // default looked like nothing had been chosen at all.
    paint(img.key);
  });
  refreshAction();
}

function cardFor(img) {
  const card = document.createElement('section');
  card.className = 'card';

  const head = document.createElement('div');
  head.className = 'card-head';
  const h2 = document.createElement('h2');
  h2.textContent = img.stem;
  const st = document.createElement('span');
  st.className = 'state';
  st.dataset.for = img.key;

  // "None of these" is a first-class answer, exactly as it is in pick.py.
  // the reviewer: "there is no option for 'they all failed' or 'null'". Without it the
  // only way to move past a bad image is to star something you would never
  // send to a customer.
  const none = document.createElement('button');
  none.className = 'nothing';
  none.type = 'button';
  none.dataset.none = img.key;
  none.textContent = 'Nothing usable';
  none.addEventListener('click', () => markFailed(img.key));

  // The default slate remains the fast path.  Burn Map is a deliberate,
  // image-scoped escape hatch for the cases where colour polarity or a
  // mid-tone transition needs a human's correction.
  const refine = document.createElement('button');
  refine.className = 'nothing refine-entry';
  refine.type = 'button';
  refine.textContent = 'Fix interpretation';
  refine.title = 'Open the optional connected-region colour editor';
  refine.addEventListener('click', (e) => { e.stopPropagation(); openRefine(img); });

  // Close clears the card from the working view only. The traced files stay on
  // disk and the run stays in History, so this is "I am done with this one",
  // not "delete it" - Delete lives in History and says so.
  const close = document.createElement('button');
  close.className = 'close';
  close.type = 'button';
  close.title = 'Clear from this view — stays in History';
  close.setAttribute('aria-label', `Close ${img.stem}`);
  close.innerHTML = '&times;';
  close.addEventListener('click', () => closeCard(img.key));

  // Item 1: fold a card away without closing it. Deliberately NOT the same
  // action as Close - a collapsed card keeps its picks and keeps showing
  // "5 shippable, saving otsu" in the head, so a long batch can be worked
  // through top to bottom while the finished ones stay decided and out of the
  // way. Clicking anywhere on the head bar toggles it; the caret is the
  // affordance.
  const caret = document.createElement('button');
  caret.className = 'caret';
  caret.type = 'button';
  caret.title = 'Collapse — keeps your picks';
  caret.setAttribute('aria-expanded', 'true');
  caret.innerHTML = '<span>&#9662;</span>';
  caret.addEventListener('click', (e) => { e.stopPropagation(); toggleCollapse(img.key); });

  const right = document.createElement('div');
  right.style.cssText = 'display:flex;align-items:center;gap:12px';
  right.append(st, none, refine);
  // Developer mode only. Both are about the RECORD, not about the image:
  // Exclude retracts this run from the statistics and hands the image back to
  // the calibration queue; Delete erases it everywhere. Neither is anything a
  // customer tracing a logo should be able to press by accident, which is why
  // they only exist while the toggle is on.
  if (settings.dev_mode) {
    const ex = document.createElement('button');
    ex.className = 'nothing dev';
    ex.type = 'button';
    ex.textContent = 'Exclude';
    ex.title = 'Do not record this run — image returns to the unjudged queue';
    ex.addEventListener('click', (e) => { e.stopPropagation(); excludeImage(img.key); });
    const del = document.createElement('button');
    del.className = 'nothing dev danger';
    del.type = 'button';
    del.textContent = 'Delete…';
    del.title = 'Erase this image and every trace of it, permanently';
    del.addEventListener('click', (e) => { e.stopPropagation(); purgeImage(img.key); });
    right.append(ex, del);
  }
  right.append(caret, close);
  head.append(h2, right);
  head.style.cursor = 'pointer';
  head.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    toggleCollapse(img.key);
  });

  const strip = document.createElement('div');
  strip.className = 'strip';

  if (img.source) {
    const t = document.createElement('div');
    t.className = 'tile source';
    t.innerHTML = `<div class="plate"><img alt="original ${esc(img.stem)}" src="${img.source}"></div>
      <div class="tile-foot"><span class="tile-name">Original</span></div>
      <div class="tile-head"></div>`;
    strip.appendChild(t);
  }

  img.candidates.forEach((c) => strip.appendChild(tileFor(img.key, c)));

  const note = document.createElement('p');
  note.className = 'failnote';
  note.dataset.note = img.key;
  note.hidden = true;
  note.innerHTML = 'Marked unusable &mdash; nothing will be saved for this one. '
    + '<b>What usually works next:</b> clean the source first (crop tight, raise '
    + 'contrast, flatten it to solid black on white) and drop it in again. '
    + 'About 1 logo in 14 still beats every strategy; those are the ones worth '
    + 'redrawing by hand.';

  card.dataset.card = img.key;
  card.append(head, strip, note);
  return card;
}

function toggleCollapse(key, force) {
  const card = document.querySelector(`[data-card="${cssEsc(key)}"]`);
  if (!card) return;
  const now = force === undefined ? !card.classList.contains('collapsed') : force;
  card.classList.toggle('collapsed', now);
  const caret = card.querySelector('.caret');
  if (caret) {
    caret.setAttribute('aria-expanded', String(!now));
    caret.title = now ? 'Expand' : 'Collapse — keeps your picks';
  }
  refreshAction();
}

$('#collapseall').addEventListener('click', () => {
  // One button, two directions: if anything is open it closes everything,
  // otherwise it opens everything. Two separate buttons for this would be one
  // more thing on a bar that already has the only button that matters.
  const anyOpen = [...document.querySelectorAll('.card')]
    .some((c) => !c.classList.contains('collapsed'));
  state.images.forEach((i) => toggleCollapse(i.key, anyOpen));
});

function closeCard(key) {
  state.images = state.images.filter((i) => i.key !== key);
  delete state.sel[key];
  delete state.fav[key];
  delete state.failed[key];
  delete state.touched[key];
  delete state.notes[key];
  const card = document.querySelector(`[data-card="${cssEsc(key)}"]`);
  if (card) card.remove();
  if (!state.images.length) $('#results').hidden = true;
  refreshAction();
}

/* In developer mode, "Nothing usable" asks WHY before it records anything.
 * A bare failure row says the tracer lost; it does not say whether the input
 * was a photograph of a patch on a jacket, a 90 px JPEG, or a genuine miss
 * worth fixing - and by the time the log is analysed the image is a stem in a
 * file. Cancelling the box cancels the verdict rather than recording a blank
 * one. Un-marking never asks; only the Escape hatch needs a reason. */
function markFailed(key) {
  if (!state.failed[key] && settings.dev_mode) {
    askText({
      title: 'Why was nothing shippable?',
      sub: 'Recorded on this image\'s row in the log, next to the verdict. '
         + 'Free text - what was wrong with the source, or with every result.',
      placeholder: 'e.g. source is a photo of a patch, heavy shadow on the left',
      confirm: 'Mark unusable',
    }).then((note) => {
      if (note === null) return;                 // cancelled: no verdict at all
      state.notes[key] = note;
      applyFailed(key, true);
    });
    return;
  }
  applyFailed(key, !state.failed[key]);
}

function applyFailed(key, failed) {
  state.failed[key] = failed;
  if (!failed) delete state.notes[key];
  if (failed) {
    state.sel[key] = new Set();
    state.fav[key] = null;
  }
  const card = document.querySelector(`[data-card="${cssEsc(key)}"]`);
  if (card) card.classList.toggle('failed', failed);
  const btn = document.querySelector(`[data-none="${cssEsc(key)}"]`);
  if (btn) { btn.classList.toggle('on', failed); btn.textContent = failed ? 'Unusable' : 'Nothing usable'; }
  const note = document.querySelector(`[data-note="${cssEsc(key)}"]`);
  if (note) note.hidden = !failed;
  paint(key);
  // Recorded here and not only on Save, because an image where nothing worked
  // is never saved - and that row is the single most useful thing this log
  // collects, especially from other people's installs.
  const img = state.images.find((i) => i.key === key);
  if (img) recordJudgement([img]);
}

function tileFor(stem, c) {
  const t = document.createElement('div');
  t.className = 'tile';
  t.dataset.stem = stem;
  t.dataset.name = c.name;
  t.tabIndex = 0;
  t.innerHTML = `
    <div class="plate"><img alt="${esc(c.name)} result" loading="lazy" src="${c.preview}"></div>
    <button class="star" type="button" title="Make this the one that gets saved"
            aria-label="Star ${esc(c.name)}">&#9733;</button>
    <button class="expand" type="button" title="View full size"
            aria-label="View ${esc(c.name)} full size"><svg viewBox="0 0 16 16"
      width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6"
      stroke-linecap="round" aria-hidden="true"><path d="M6 2H2v4M10 14h4v-4"/>
      </svg></button>
    <div class="tile-foot">
      <span class="tile-name">${c.n}. ${esc(c.name)}</span>
      <span class="tile-score">${c.score}</span>
    </div>
    <div class="tile-head">${esc(c.headline || '')}</div>`;

  t.querySelector('.expand').addEventListener('click', (e) => {
    e.stopPropagation();
    openViewer(stem, c.name);
  });

  t.addEventListener('click', (e) => {
    if (e.target.closest('.star') || e.target.closest('.expand')) return;
    toggle(stem, c.name);
  });
  t.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(stem, c.name); }
  });
  t.querySelector('.star').addEventListener('click', (e) => {
    e.stopPropagation();
    star(stem, c.name);
  });
  return t;
}

// The top candidate is pre-starred as a sensible default, but until the user
// has actually touched this image that default is a GUESS, not a choice.
// Clicking a different tile used to ADD to it, leaving two selected and forcing
// the reviewer to go back and deselect the first. The first interaction now replaces
// the guess; every one after that adds or removes as normal.
function clearGuess(key) {
  if (state.touched[key]) return;
  state.touched[key] = true;
  state.sel[key] = new Set();
  state.fav[key] = null;
}

function toggle(stem, name) {
  if (state.failed[stem]) markFailed(stem);   // choosing one un-fails the image
  clearGuess(stem);
  const set = (state.sel[stem] ||= new Set());
  if (set.has(name)) {
    set.delete(name);
    // Unstarring by deselecting would leave nothing to save, so hand the star
    // to whatever is still selected rather than silently dropping the image.
    if (state.fav[stem] === name) state.fav[stem] = [...set][0] || null;
  } else {
    set.add(name);
    if (!state.fav[stem]) state.fav[stem] = name;
  }
  paint(stem);
}

function star(stem, name) {
  if (state.failed[stem]) markFailed(stem);
  clearGuess(stem);
  (state.sel[stem] ||= new Set()).add(name);
  state.fav[stem] = name;
  paint(stem);
}

function paint(stem) {
  const sel = state.sel[stem] || new Set();
  document.querySelectorAll(`.tile[data-stem="${cssEsc(stem)}"]`).forEach((t) => {
    const n = t.dataset.name;
    t.classList.toggle('on', sel.has(n));
    t.classList.toggle('fav', state.fav[stem] === n);
  });
  const st = document.querySelector(`.state[data-for="${cssEsc(stem)}"]`);
  if (st) {
    const fav = state.fav[stem];
    st.textContent = state.failed[stem] ? 'nothing usable'
      : fav ? `${sel.size} shippable · saving ${fav}` : 'none selected';
    st.classList.toggle('picked', !!fav && !state.failed[stem]);
  }
  refreshAction();
}

/* ---------------- saving ---------------- */
// Item 8: the dropzone's size is now DERIVED from state on every refresh
// rather than toggled by hand at the one place that happened to remember to.
// The upload path did shrink it correctly; History -> Reopen did not, so a
// reopened run showed a full-height 207px dropzone shoving the results down
// the page. Deriving it means every route in and out - upload, reopen, close
// the last card, switch tabs - lands on the same size for the same state, and
// it can never grow while results are on screen.
function syncDropzone() {
  drop.classList.toggle('compact', state.images.length > 0);
}

function refreshAction() {
  syncDropzone();
  const favs = state.images.filter((img) => state.fav[img.key] && !state.failed[img.key]);
  const bad = state.images.filter((img) => state.failed[img.key]).length;
  const ready = favs.length > 0 && $('#dest').value.trim().length > 0;
  $('#actionbar').hidden = !state.images.length;
  $('#collapseall').textContent =
    [...document.querySelectorAll('.card')].some((c) => !c.classList.contains('collapsed'))
      ? 'Collapse all' : 'Expand all';
  $('#save').disabled = !ready;
  const sp = $('#savepicks');
  sp.hidden = !settings.dev_mode || !state.images.length;
  sp.disabled = !state.images.length;
  const tail = bad ? ` · ${bad} marked unusable` : '';
  $('#summary').textContent = !state.images.length ? ''
    : favs.length === 0 ? `Star one candidate per logo to save it.${tail}`
    : !$('#dest').value.trim() ? `${favs.length} ready — choose a folder to save to.${tail}`
    : `${favs.length} of ${state.images.length} ready to save.${tail}`;
}

$('#save').addEventListener('click', async () => {
  // Images accumulate across drops, and the save endpoint is per job, so the
  // favourites are grouped by the job each image came from and posted
  // separately. Keys are display keys; the server wants the real stem.
  const byJob = {};
  state.images.forEach((img) => {
    const fav = state.fav[img.key];
    if (fav && !state.failed[img.key]) (byJob[img.jobId] ||= {})[img.stem] = fav;
  });

  $('#save').disabled = true;
  const dest = $('#dest').value.trim();
  let written = 0;
  const failures = [];
  try {
    for (const [jobId, picks] of Object.entries(byJob)) {
      const r = await apiFetch(`/api/jobs/${jobId}/save`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dest, picks }),
      });
      const j = await r.json();
      if (!r.ok) { failures.push(j.detail || r.statusText); continue; }
      written += j.written.length;
    }
    if (failures.length) toast(failures[0], true);
    else toast(`Saved ${written} SVG${written === 1 ? '' : 's'} to ${dest}`);
    // Pressing Save is the endorsement, so everything on screen is recorded -
    // including images left on the pre-starred default, because choosing to
    // save that default IS choosing it. Images marked unusable go in too.
    await recordJudgement(state.images);
    saveSettings({ dest });
  } catch (err) {
    toast(String(err.message || err), true);
  } finally {
    refreshAction();
  }
});

/* Developer mode: log the batch, write nothing.
 *
 * Normally the record is a side effect of Save - which is right for real work,
 * where judging an image and saving its SVG are the same act. Judging a
 * CALIBRATION batch is not: the verdicts are the whole point and the SVGs are
 * junk, so the only way to record 20 images was to write 20 files somewhere
 * and delete them. This records exactly what Save would record, and saves no
 * files, so the cards can then be closed harmlessly. */
$('#savepicks').addEventListener('click', async () => {
  const btn = $('#savepicks');
  btn.disabled = true;
  try {
    await recordJudgement(state.images);
    const n = state.images.length;
    const bad = state.images.filter((i) => state.failed[i.key]).length;
    toast(`Logged ${n} judgement${n === 1 ? '' : 's'}`
      + (bad ? ` (${bad} unusable)` : '') + ' — no files written');
  } catch (err) {
    toast(String(err.message || err), true);
  } finally {
    refreshAction();
  }
});


/* ---------------- utils ---------------- */
let toastTimer;
function toast(msg, isErr) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.toggle('err', !!isErr);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 5200);
}
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const cssEsc = (s) => (window.CSS?.escape ? CSS.escape(s) : String(s).replace(/["\\]/g, '\\$&'));


/* ---------------- history ----------------
 * Every traced job stays on disk under .work/, and the server indexes them in
 * job.json. Opening one loads its candidates back into the Current view, so a
 * saved SVG that turns out wrong on the machine can be swapped for a different
 * strategy without re-tracing. the reviewer's reason for asking, verbatim: "in case
 * the one i download doesnt work, i can grab the another one from a previous
 * run." */
const TABS = { current: $('#tab-current'), history: $('#tab-history'),
               stats: $('#tab-stats'), settings: $('#tab-settings') };

function showTab(which) {
  Object.entries(TABS).forEach(([name, el]) => {
    el.classList.toggle('on', name === which);
    el.setAttribute('aria-selected', String(name === which));
  });
  const cur = which === 'current';
  $('#history').hidden = which !== 'history';
  $('#stats').hidden = which !== 'stats';
  $('#settings').hidden = which !== 'settings';
  $('#drop').hidden = !cur;
  $('#howto').hidden = !cur;
  $('#results').hidden = !cur || !state.images.length;
  $('#actionbar').hidden = !cur || !state.images.length;
  if (which === 'history') loadHistory();
  if (which === 'stats') loadStats();
  if (which === 'settings') loadSettingsTab();
}
Object.keys(TABS).forEach((name) =>
  TABS[name].addEventListener('click', () => showTab(name)));

async function loadHistory() {
  const box = $('#history');
  box.innerHTML = '<p class="hempty">Loading…</p>';
  try {
    const { jobs } = await (await apiFetch('/api/history')).json();
    const open = new Set(state.images.map((i) => i.jobId));
    if (!jobs.length) {
      box.innerHTML = '<p class="hempty">Nothing traced yet.</p>';
      return;
    }
    box.innerHTML = '';
    jobs.forEach((j) => {
      const row = document.createElement('div');
      row.className = 'hrow';
      const when = new Date(j.created);
      row.innerHTML = `
        <span class="when">${isNaN(when) ? j.created : when.toLocaleString()}</span>
        <span class="what">${esc((j.names || []).join(', ')) || '—'}</span>
        <span class="count">${j.total}</span>`;
      const openBtn = document.createElement('button');
      openBtn.className = 'ghost';
      openBtn.type = 'button';
      openBtn.textContent = open.has(j.id) ? 'Open' : 'Reopen';
      openBtn.addEventListener('click', async () => {
        if (!open.has(j.id)) await loadResults(j.id);
        showTab('current');
      });
      const del = document.createElement('button');
      del.className = 'ghost';
      del.type = 'button';
      del.textContent = 'Delete';
      del.title = 'Remove this run and its traced files from disk';
      del.addEventListener('click', async () => {
        await apiFetch(`/api/jobs/${j.id}`, { method: 'DELETE' });
        state.images = state.images.filter((i) => i.jobId !== j.id);
        document.querySelectorAll('.card').forEach((c) => {
          if (!state.images.some((i) => i.key === c.dataset.card)) c.remove();
        });
        refreshAction();
        loadHistory();
      });
      row.append(openBtn, del);
      box.appendChild(row);
    });
  } catch (err) {
    box.innerHTML = '<p class="hempty">Could not load history.</p>';
  }
}


/* ---------------- what the user decided ----------------
 * Written in pick.py's row shape so the two logs can be read by the same eye,
 * but to a SEPARATE file - see the long note in server.py. Short version, in
 * app rows are useful, and they are not ground truth: what gets dropped into a
 * desktop app is not necessarily a logo at all. */
function judgementFor(img) {
  const fav = state.fav[img.key] || null;
  const sel = [...(state.sel[img.key] || new Set())];
  // Favourite first, matching pick.py: the first entry is the one that ships.
  const ordered = fav ? [fav, ...sel.filter((n) => n !== fav)] : sel;
  const rankOf = (nm) => (img.candidates.find((c) => c.name === nm) || {}).n || 0;
  return {
    stem: img.stem,
    pick: fav,
    shippable: ordered,
    shippable_ranks: ordered.map(rankOf),
    of: img.candidates.length,
    failed: !!state.failed[img.key],
    // Dev mode only - the reason box on 'Nothing usable'. Lands on the row
    // itself so a later reader of the log can tie the verdict to WHY, the
    // same way pick.py's rows tie a pick to the image it was made on.
    note: state.notes[img.key] || null,
  };
}

async function recordJudgement(imgs) {
  const byJob = {};
  imgs.forEach((img) => { (byJob[img.jobId] ||= []).push(judgementFor(img)); });
  for (const [jobId, rows] of Object.entries(byJob)) {
    try {
      await apiFetch(`/api/jobs/${jobId}/judge`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rows),
      });
    } catch (_) { /* never let bookkeeping break the actual work */ }
  }
  // A verdict is what takes an image out of the calibration queue, so the
  // count is only ever right if it is re-read here.
  loadIntake();
}


/* ---------------- developer mode ----------------
 * Two problems, one switch, and both of them only exist because this app is
 * also the instrument used to MEASURE the tracer:
 *
 *   1. A test run pollutes the record. Drop the wrong file in, or re-trace
 *      something to watch the live view, and the log now carries a verdict
 *      that describes an experiment rather than the tracer's real hit rate.
 *   2. A bare failure row loses the reason. "nothing usable" six weeks later
 *      is unanalysable without knowing what was wrong with it.
 *
 * Exclude solves (1) softly: the row is written with excluded:true, the
 * server drops it from the statistics AND from the judged set, so the image
 * comes back in the calibration queue - and the file still records that it
 * happened, and why. Delete solves it the hard way, with no undo.
 * ------------------------------------------------------------------------ */

/* One dialog, promise-shaped, built here rather than in index.html so a
 * customer install never ships the markup for controls it cannot reach.
 * Resolves to the entered text (possibly empty) or null for cancelled. */
function askText({ title, sub, placeholder, confirm, danger, body }) {
  return new Promise((resolve) => {
    const wrap = document.createElement('div');
    wrap.className = 'modal';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-modal', 'true');
    wrap.innerHTML = `<div class="modal-box" style="width:min(560px,100%)">
        <div class="modal-head"><h2>${esc(title)}</h2>
          <button class="close" type="button" aria-label="Close">&times;</button></div>
        <div class="modal-body">
          <p>${esc(sub)}</p>
          ${body || ''}
          ${placeholder === null ? '' : '<textarea class="devnote" rows="3"></textarea>'}
          <div class="share-actions" style="margin-top:12px">
            <button class="ghost cancel" type="button">Cancel</button>
            <button class="primary go ${danger ? 'danger' : ''}" type="button">${esc(confirm)}</button>
          </div>
        </div></div>`;
    const ta = wrap.querySelector('.devnote');
    if (ta) ta.placeholder = placeholder;
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      document.removeEventListener('keydown', onKey);
      wrap.remove();
      resolve(value);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') finish(null);
      // Ctrl/Cmd+Enter submits, so a reason can be typed and filed without
      // leaving the keyboard in the middle of a batch.
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) finish(ta ? ta.value.trim() : '');
    };
    wrap.querySelector('.cancel').addEventListener('click', () => finish(null));
    wrap.querySelector('.close').addEventListener('click', () => finish(null));
    wrap.querySelector('.go').addEventListener('click', () => finish(ta ? ta.value.trim() : ''));
    wrap.addEventListener('click', (e) => { if (e.target === wrap) finish(null); });
    document.addEventListener('keydown', onKey);
    document.body.appendChild(wrap);
    (ta || wrap.querySelector('.go')).focus();
  });
}

async function excludeImage(key) {
  const img = state.images.find((i) => i.key === key);
  if (!img) return;
  const note = await askText({
    title: `Exclude ${img.stem} from the record`,
    sub: 'Nothing is deleted. The run is written to the log marked excluded, '
       + 'kept out of every statistic, and the image goes back into the '
       + 'unjudged calibration queue so it can be traced again. This also '
       + 'retracts a verdict already recorded for it.',
    placeholder: 'Why (optional) — e.g. wrong file, re-traced to test live view',
    confirm: 'Exclude',
  });
  if (note === null) return;
  const row = { ...judgementFor(img), excluded: true, note: note || null };
  try {
    await apiFetch(`/api/jobs/${img.jobId}/judge`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify([row]),
    });
    toast(`${img.stem} excluded — back in the unjudged queue`);
  } catch (err) {
    toast(String(err.message || err), true);
    return;
  }
  closeCard(key);
  loadIntake();
}

/* Delete is two round trips on purpose: /api/purge/plan first, so the exact
 * paths about to go are on screen before the second click. There is no undo,
 * no recycle bin, and the source artwork is included - the requested scope
 * asked for, in his words "fully deletes out of the corpus, picks.py, and
 * anywhere else it exists in this project". */
async function purgeImage(key) {
  const img = state.images.find((i) => i.key === key);
  if (!img) return;
  let plan;
  try {
    plan = await (await apiFetch('/api/purge/plan', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stem: img.stem }),
    })).json();
  } catch (err) {
    toast(String(err.message || err), true);
    return;
  }
  const rows = [
    plan.rows.app ? `${plan.rows.app} row(s) in app-picks.jsonl` : null,
    plan.rows.calibration ? `${plan.rows.calibration} row(s) in picks.jsonl` : null,
  ].filter(Boolean);
  const list = plan.paths.map((p) => `<li><span class="what">${esc(p.what)}</span>`
    + `${esc(p.path)}${p.dir ? '/' : ''}</li>`).join('');
  const ok = await askText({
    title: `Delete ${img.stem} permanently`,
    // Files and log rows counted separately: one "6 items" line over a list
    // of 4 paths reads like the list is incomplete.
    sub: `${plan.paths.length} file(s) and ${plan.rows.app + plan.rows.calibration} `
       + 'log row(s) will be erased. This cannot be undone and the source image '
       + 'is included — nothing in this project will remember it.',
    placeholder: null,
    confirm: 'Delete permanently',
    danger: true,
    body: `<ul class="purgelist">${list || '<li>no files found</li>'}</ul>`
        + (rows.length ? `<p class="sub">Also removed: ${esc(rows.join(', '))}</p>` : ''),
  });
  if (ok === null) return;
  try {
    const r = await (await apiFetch('/api/purge', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stem: img.stem }),
    })).json();
    if (r.failed && r.failed.length) toast(r.failed[0], true);
    else toast(`${img.stem} deleted — ${r.removed.length} path(s), `
      + `${r.rows.app + r.rows.calibration} log row(s)`);
  } catch (err) {
    toast(String(err.message || err), true);
    return;
  }
  closeCard(key);
  loadIntake();
  loadHistory?.();
}


/* ---------------- how-it-works modal (item 2) ---------------- */
const modal = $('#modal');
const openModal = () => { modal.hidden = false; $('#modalclose').focus(); };
const closeModal = () => { modal.hidden = true; $('#infobtn').focus(); };
$('#infobtn').addEventListener('click', openModal);
$('#modalclose').addEventListener('click', closeModal);
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !modal.hidden) closeModal();
});


/* ---------------- statistics (item 3) ----------------
 * Two populations, drawn side by side and never averaged into one number.
 * `calibration` is picks.jsonl: the reviewer at a contact sheet over a corpus chosen
 * to represent the work. `this install` is whatever got dropped into the app.
 *
 * The headline is the RECENT window, not the lifetime figure, and that is a
 * correctness decision rather than flattery. Lifetime, 35 of 79 calibration
 * images produced nothing usable - 44%. Over the last two days of that same
 * log it is 4 of 30 - 13%. The gap is the tracer improving, so a single
 * blended number would describe a pipeline that no longer exists and would
 * understate the current one by about four times. Both are shown, labelled. */
async function loadStats() {
  const box = $('#stats');
  box.innerHTML = '<p class="hempty">Loading…</p>';
  let d;
  try {
    d = await (await apiFetch('/api/stats')).json();
  } catch (_) {
    box.innerHTML = '<p class="hempty">Could not load statistics.</p>';
    return;
  }
  box.innerHTML = '';
  box.appendChild(statGroup(
    'This install', d.app, d.strategy_order,
    'Every image you have judged in the app. Recorded when you save, and when '
    + 'you mark one “Nothing usable”.'));
  if (d.calibration.available) {
    box.appendChild(statGroup(
      'Calibration corpus', d.calibration, d.strategy_order,
      'Reviewed on the command line against a corpus chosen to represent real '
      + 'work. This is the set the scoring is tuned against — kept separate '
      + 'from app runs on purpose, never merged.'));
  }
  box.appendChild(shareGroup(d));
}

function pct(n, of) { return of ? `${Math.round((n / of) * 100)}%` : '—'; }

function bignum(v, k, detail, cls) {
  const el = document.createElement('div');
  el.className = 'bignum';
  el.innerHTML = `<div class="v ${cls || ''}">${esc(v)}</div>
                  <div class="k">${esc(k)}</div>
                  <div class="d">${esc(detail || '')}</div>`;
  return el;
}

function statGroup(title, s, order, sub) {
  const g = document.createElement('section');
  g.className = 'statgroup';
  const h = document.createElement('h2');
  h.textContent = title;
  const p = document.createElement('p');
  p.className = 'sub';
  p.textContent = sub;
  g.append(h, p);

  if (!s.judged) {
    const empty = document.createElement('p');
    empty.className = 'hempty';
    empty.textContent = 'Nothing judged yet.';
    g.appendChild(empty);
    return g;
  }

  const r = s.recent || { judged: 0, nothing_usable: 0 };
  const recentOk = r.judged - r.nothing_usable;
  const nums = document.createElement('div');
  nums.className = 'bignums';
  nums.append(
    bignum(pct(recentOk, r.judged), 'shippable — recent',
           `${recentOk} of the last ${r.judged} judged`,
           recentOk / (r.judged || 1) >= 0.8 ? 'good' : 'warn'),
    bignum(pct(s.usable, s.judged), 'shippable — all time',
           `${s.usable} of ${s.judged} ever judged`),
    bignum(String(s.nothing_usable), 'nothing usable',
           `${pct(s.nothing_usable, s.judged)} of all runs`),
    bignum(s.avg_shippable == null ? '—' : String(s.avg_shippable),
           'usable per image',
           s.sets ? `over ${s.sets} fully-marked images` : 'no full sets yet'),
    bignum(pct(s.top1_shippable, s.sets), 'top pick was usable',
           `${s.top1_shippable} of ${s.sets} fully-marked`),
  );
  g.appendChild(nums);

  // Per-strategy. The denominator is images that recorded a FULL set, not all
  // judged images, because a row from before multi-select could only name one
  // winner - counting those would silently punish every strategy that was not
  // picked on a day when only one could be.
  const known = new Set(order);
  const rows = Object.entries(s.strategies)
    .filter(([, v]) => v.shippable || v.starred)
    .sort((a, b) => b[1].shippable - a[1].shippable || b[1].starred - a[1].starred);

  if (rows.length) {
    const t = document.createElement('table');
    t.className = 'stat';
    t.innerHTML = `<thead><tr>
      <th>Method</th><th>Shippable</th>
      <th class="num">Rate</th><th class="num">Starred</th></tr></thead>`;
    const tb = document.createElement('tbody');
    rows.forEach(([name, v]) => {
      const tr = document.createElement('tr');
      if (!known.has(name)) tr.className = 'demoted';
      const rate = s.sets ? Math.round((v.shippable / s.sets) * 100) : 0;
      tr.innerHTML = `
        <td class="name">${esc(name)}${known.has(name) ? '' : ' <span class="d">(retired)</span>'}</td>
        <td>${v.shippable} of ${s.sets}
            <span class="meter"><i style="width:${rate}%"></i></span></td>
        <td class="num">${s.sets ? `${rate}%` : '—'}</td>
        <td class="num">${v.starred}</td>`;
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    g.appendChild(t);
  }

  // Trend by day. This is the table that makes the two headline numbers above
  // legible instead of contradictory.
  const dates = Object.entries(s.by_date || {});
  if (dates.length > 1) {
    const t2 = document.createElement('table');
    t2.className = 'stat';
    t2.style.marginTop = '24px';
    t2.innerHTML = `<thead><tr><th>Day</th><th class="num">Judged</th>
      <th class="num">Nothing usable</th><th class="num">Shippable</th></tr></thead>`;
    const tb2 = document.createElement('tbody');
    dates.forEach(([day, v]) => {
      const ok = v.judged - v.nothing_usable;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="name">${esc(day)}</td>
        <td class="num">${v.judged}</td>
        <td class="num">${v.nothing_usable}</td>
        <td class="num">${pct(ok, v.judged)}</td>`;
      tb2.appendChild(tr);
    });
    t2.appendChild(tb2);
    g.appendChild(t2);
  }
  return g;
}


/* ---------------- sharing statistics (item 4) ----------------
 * Opt-in, aggregate-only, and never transmitted by the app. The button builds
 * the JSON, shows it in full, and opens a prefilled GitHub issue; the user
 * pastes and submits it themselves. Nothing leaves the machine without a human
 * pressing send, and there is no server to trust because there is no server. */
const REPO_URL = 'https://github.com/JudgeBreddd/t-tracer';

function shareGroup(d) {
  const g = document.createElement('section');
  g.className = 'statgroup';
  g.innerHTML = `<h2>Help improve the tracer</h2>
    <p class="sub">Off by default. When on, you can build an anonymous summary
      of the numbers above and send it as a GitHub issue. It contains counts
      only — no images, no file names, no folders, nothing that identifies you
      or your machine. You see the exact text before anything is sent, and the
      app never transmits it for you.</p>`;

  const wrap = document.createElement('div');
  wrap.className = 'share';

  const label = document.createElement('label');
  label.className = 'optin';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = !!settings.share_stats;
  const span = document.createElement('span');
  span.textContent = 'Let me share anonymous usage statistics';
  label.append(cb, span);

  const actions = document.createElement('div');
  actions.className = 'share-actions';
  const build = document.createElement('button');
  build.className = 'primary';
  build.type = 'button';
  build.textContent = 'Build report';
  build.disabled = !cb.checked;

  const copy = document.createElement('button');
  copy.className = 'ghost';
  copy.type = 'button';
  copy.textContent = 'Copy';
  copy.hidden = true;

  const issue = document.createElement('a');
  issue.className = 'ghost';
  issue.target = '_blank';
  issue.rel = 'noopener';
  issue.textContent = 'Open a GitHub issue';
  issue.hidden = true;
  issue.style.textDecoration = 'none';

  const box = document.createElement('textarea');
  box.id = 'reportbox';
  box.readOnly = true;
  box.hidden = true;

  cb.addEventListener('change', () => {
    saveSettings({ share_stats: cb.checked });
    build.disabled = !cb.checked;
    if (!cb.checked) { box.hidden = copy.hidden = issue.hidden = true; }
  });

  build.addEventListener('click', async () => {
    try {
      const r = await apiFetch('/api/stats/report');
      const j = await r.json();
      if (!r.ok) { toast(j.detail || 'Could not build the report.', true); return; }
      const text = JSON.stringify(j, null, 2);
      box.value = text;
      box.hidden = copy.hidden = issue.hidden = false;
      const title = encodeURIComponent('Usage stats');
      const body = encodeURIComponent('```json\n' + text + '\n```');
      issue.href = `${REPO_URL}/issues/new?title=${title}&body=${body}`;
    } catch (err) {
      toast(String(err.message || err), true);
    }
  });

  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(box.value);
      toast('Report copied.');
    } catch (_) {
      box.select();
      toast('Press Ctrl+C to copy.');
    }
  });

  actions.append(build, copy, issue);
  wrap.append(label, actions, box);
  g.appendChild(wrap);
  return g;
}


/* ---------------- settings tab (item 7: history storage controls) ----------------
 * "Settings shows storage used; Clear generated history button; retention
 * option (Forever / 90 days / 30 days, default Forever)." Cleanup itself
 * lives entirely server-side (select_jobs_to_clean in server.py) - this is
 * just the display and the two controls that trigger it. */
function humanBytes(n) {
  if (n == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

async function loadSettingsTab() {
  const box = $('#settings');
  box.innerHTML = '<p class="hempty">Loading…</p>';
  let storageInfo;
  try {
    storageInfo = await (await apiFetch('/api/storage')).json();
  } catch (_) {
    box.innerHTML = '<p class="hempty">Could not load settings.</p>';
    return;
  }
  // Re-fetch settings rather than trusting the module-level cache: this tab
  // can be the first thing opened after a fresh /api/settings PUT elsewhere,
  // and it needs 'platform' too, which loadSettings() at boot already fetched
  // but the auto-update toggle below reads off `settings` directly.
  try { settings = await (await apiFetch('/api/settings')).json(); } catch (_) {}
  let updateState = { status: 'idle', detail: '' };
  try { updateState = await (await apiFetch('/api/update-status')).json(); } catch (_) {}

  box.innerHTML = '';
  box.appendChild(storageGroup(storageInfo));
  box.appendChild(autoUpdateGroup(updateState));
  box.appendChild(liveViewGroup());
  box.appendChild(devModeGroup());
}

/* Developer mode. One switch, and everything it turns on is about the RECORD
 * rather than about tracing: the per-image Exclude and Delete controls, and
 * the reason box on 'Nothing usable'. Off on a normal install, and the server
 * refuses the delete endpoints outright while it is off - so this is the
 * on/off for a capability, not just for some buttons. Takes effect on the
 * next job shown; cards already on screen keep the controls they were built
 * with. */
function devModeGroup() {
  const g = document.createElement('section');
  g.className = 'statgroup';
  g.innerHTML = `<h2>Developer mode</h2>
    <p class="sub">For testing the tracer rather than using it. Adds to every
      image: <b>Exclude</b> — keep this run out of the statistics and put the
      image back in the unjudged queue, which also retracts a verdict already
      recorded by mistake; <b>Delete</b> — erase the image, its traced output
      and its log rows everywhere in the project, permanently; and a
      <b>reason box</b> on "Nothing usable" so a failure row says why. Off by
      default. New cards only — re-trace or reopen an image to see the
      controls appear or disappear.</p>`;
  const wrap = document.createElement('div');
  wrap.className = 'share';
  const label = document.createElement('label');
  label.className = 'optin';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = !!settings.dev_mode;
  const span = document.createElement('span');
  span.textContent = 'Turn on developer mode';
  label.append(cb, span);
  cb.addEventListener('change', () => saveSettings({ dev_mode: cb.checked }));
  wrap.appendChild(label);
  g.appendChild(wrap);
  return g;
}

/* Live view: a diagnostic, not a customer feature. Both switches only write
 * settings; the server decides per job at creation time whether to attach
 * an observer, so flipping them mid-trace changes the NEXT job, not this one. */
function liveViewGroup() {
  const g = document.createElement('section');
  g.className = 'statgroup';
  g.innerHTML = `<h2>Live view (debug)</h2>
    <p class="sub">Show each strategy's steps while a job traces - about three
      frames a second, latest frame only, nothing queued. Off by default. If a
      step already looks wrong mid-run, that is the step that introduced it.
      Single-image jobs only.</p>`;
  const wrap = document.createElement('div');
  wrap.className = 'share';
  const mk = (key, text) => {
    const label = document.createElement('label');
    label.className = 'optin';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!settings[key];
    const span = document.createElement('span');
    span.textContent = text;
    label.append(cb, span);
    cb.addEventListener('change', () => saveSettings({ [key]: cb.checked }));
    return label;
  };
  wrap.appendChild(mk('live_view', 'Show the live view while tracing'));
  wrap.appendChild(mk('live_keep',
    'Keep every snapshot until the job ends, then ask whether to save them '
    + '(into the job\'s own history folder) or discard them'));
  g.appendChild(wrap);
  return g;
}

/* Item 3.5: the toggle only ever writes settings.auto_update=true/false - the
 * actual download/verify/install runs server-side, once per launch, in the
 * background (server.py _auto_update_once). This group just shows the
 * result of the last attempt (update-status) and lets Windows users opt in. */
function autoUpdateGroup(updateState) {
  const g = document.createElement('section');
  g.className = 'statgroup';
  g.innerHTML = '<h2>Automatic updates</h2>';

  const sub = document.createElement('p');
  sub.className = 'sub';
  const isWindows = settings.platform === 'win32';
  sub.textContent = isWindows
    ? 'When on, T-Tracer checks for a newer release at startup, downloads it, '
      + 'verifies it against the checksum published with the release, and '
      + 'installs it automatically - the running app closes to let the '
      + 'installer run. Off by default. A failed or missing checksum is never '
      + 'run; a declined or offline check is silent.'
    : 'Available on Windows installs only - T-Tracer never downloads or runs '
      + 'an installer on this platform.';
  g.appendChild(sub);

  const wrap = document.createElement('div');
  wrap.className = 'share';
  const label = document.createElement('label');
  label.className = 'optin';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = !!settings.auto_update;
  cb.disabled = !isWindows;
  const span = document.createElement('span');
  span.textContent = 'Automatically download and install updates';
  label.append(cb, span);
  cb.addEventListener('change', () => saveSettings({ auto_update: cb.checked }));
  wrap.appendChild(label);

  if (updateState && updateState.status === 'error') {
    const err = document.createElement('p');
    err.className = 'sub';
    err.style.color = 'var(--accent-red-bright)';
    err.textContent = updateState.detail;
    wrap.appendChild(err);
  } else if (updateState && updateState.status === 'installing') {
    const info = document.createElement('p');
    info.className = 'sub';
    info.textContent = updateState.detail;
    wrap.appendChild(info);
  }

  g.appendChild(wrap);
  return g;
}

function storageGroup(info) {
  const g = document.createElement('section');
  g.className = 'statgroup';
  g.innerHTML = `<h2>History storage</h2>
    <p class="sub">Every traced job's source images, previews and candidate
      SVGs, kept so History can reopen them. Settings, your judged picks and
      the calibration statistics are never touched by anything on this page.</p>`;

  const nums = document.createElement('div');
  nums.className = 'bignums';
  nums.append(
    bignum(humanBytes(info.bytes), 'used', `${info.jobs} job${info.jobs === 1 ? '' : 's'} in history`),
  );
  g.appendChild(nums);

  const wrap = document.createElement('div');
  wrap.className = 'share';

  const retLabel = document.createElement('p');
  retLabel.className = 'sub';
  retLabel.style.margin = '0 0 4px';
  retLabel.textContent = 'Keep history:';
  const retWrap = document.createElement('div');
  retWrap.style.cssText = 'display:flex; gap:16px; flex-wrap:wrap';
  const options = [['forever', 'Forever'], ['90d', '90 days'], ['30d', '30 days']];
  options.forEach(([value, text]) => {
    const label = document.createElement('label');
    label.className = 'optin';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'retention';
    input.value = value;
    input.checked = (settings.retention || 'forever') === value;
    input.addEventListener('change', () => { if (input.checked) saveSettings({ retention: value }); });
    const span = document.createElement('span');
    span.textContent = text;
    label.append(input, span);
    retWrap.appendChild(label);
  });

  const actions = document.createElement('div');
  actions.className = 'share-actions';
  const clearBtn = document.createElement('button');
  clearBtn.className = 'ghost';
  clearBtn.type = 'button';
  clearBtn.textContent = 'Clear generated history';
  clearBtn.title = 'Deletes finished jobs\' files. Active/queued jobs, settings, '
    + 'and your judged picks are never touched.';
  clearBtn.addEventListener('click', async () => {
    clearBtn.disabled = true;
    try {
      const r = await apiFetch('/api/history/clear', { method: 'POST' });
      const j = await r.json();
      toast(`Cleared ${j.cleared} job${j.cleared === 1 ? '' : 's'} — freed ${humanBytes(j.bytes_freed)}.`);
      // Anything cleared may still be open in Current/History - drop it from
      // both rather than leaving a card pointing at files that no longer exist.
      loadSettingsTab();
    } catch (err) {
      toast(String(err.message || err), true);
    } finally {
      clearBtn.disabled = false;
    }
  });

  actions.appendChild(clearBtn);
  wrap.append(retLabel, retWrap, actions);
  g.appendChild(wrap);
  return g;
}


/* ---------------- update check (item 6) ----------------
 * Reports, never installs. One call to the GitHub releases API at startup; if
 * it fails, is rate limited, or there are no releases yet, the bar simply does
 * not appear. Downloading is a link the user clicks, so nothing rewrites
 * itself behind their back. */
async function checkUpdate() {
  let d;
  try {
    d = await (await apiFetch('/api/update-check')).json();
  } catch (_) { return; }
  if (!d.update) return;
  let dismissed = null;
  try { dismissed = localStorage.getItem('tt.skipver'); } catch (_) {}
  if (dismissed === d.latest) return;
  $('#updatetext').textContent =
    `Version ${d.latest} is available — you have ${d.current}.`;
  $('#updatelink').href = d.url || REPO_URL + '/releases/latest';
  $('#updatebar').hidden = false;
  $('#updatedismiss').addEventListener('click', () => {
    // Per-version dismissal, so "Not now" does not silence the NEXT release
    // too. localStorage is fine for this one: losing it just re-shows a bar.
    try { localStorage.setItem('tt.skipver', d.latest); } catch (_) {}
    $('#updatebar').hidden = true;
  });
}


/* ---------------- boot ---------------- */
loadSettings();
checkUpdate();
loadIntake();


/* ---------------- optional Burn Map refinement ----------------
 * A refinement editor is created only after the user asks for it.  Its state
 * is job/image scoped on the server, so a second image cannot inherit edits
 * from the first and normal candidate saving is unaffected. */
const refine = { img: null, base: '', payload: null, generation: 0, busy: false };

function refineClose() {
  refine.generation += 1;
  $('#refine-modal').hidden = true;
  refine.img = null; refine.payload = null; refine.base = ''; refine.busy = false;
}
$('#refine-close').addEventListener('click', refineClose);
$('#refine-modal').addEventListener('click', (e) => {
  if (e.target === $('#refine-modal')) refineClose();
});

async function openRefine(img) {
  const generation = ++refine.generation;
  refine.img = img;
  refine.base = `/api/jobs/${encodeURIComponent(img.jobId)}/refine/${encodeURIComponent(img.stem)}`;
  const base = refine.base;
  $('#refine-title').textContent = `Fix interpretation — ${img.stem}`;
  $('#refine-sub').textContent = 'Optional rescue editor · the normal candidate slate is unchanged';
  $('#refine-modal').hidden = false;
  $('#refine-body').innerHTML = '<p class="meta">Preparing the optional editor…</p>';
  await refreshRefine(generation, base);
}

async function refreshRefine(generation = refine.generation, base = refine.base) {
  if (!base) return;
  try {
    const r = await apiFetch(base);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const payload = await r.json();
    if (generation !== refine.generation || base !== refine.base) return;
    refine.payload = payload;
    renderRefine();
  } catch (err) {
    if (generation !== refine.generation || base !== refine.base) return;
    $('#refine-body').innerHTML = `<p class="meta">Could not open refinement: ${esc(String(err.message || err))}</p>`;
  }
}

function refineButton(label, handler, extra = '') {
  const b = document.createElement('button'); b.type = 'button'; b.className = extra;
  b.textContent = label; b.addEventListener('click', handler); return b;
}

function renderRefine() {
  const p = refine.payload, body = $('#refine-body');
  const m = p.metrics || {};
  body.innerHTML = `<div class="refine-plates">
    <figure class="refine-plate"><img alt="original source"><figcaption>Original</figcaption></figure>
    <figure class="refine-plate"><img alt="editable region map"><figcaption>Editable regions</figcaption></figure>
    <figure class="refine-plate"><img alt="corrected black and white preview"><figcaption>Corrected preview</figcaption></figure>
    <figure class="refine-plate"><img alt="correction difference"><figcaption>Difference from starting interpretation</figcaption></figure>
  </div><aside class="refine-controls"><div class="refine-notice">This is a review surface, not an automatic truth claim. Choose a palette interpretation, toggle a colour layer, or correct one connected region.</div>
    <section><h3>Interpretations</h3><div class="refine-proposals"></div></section>
    <section><h3>Colour layers</h3><div class="refine-groups"></div></section>
    <section><h3>Largest connected regions</h3><div class="refine-list"></div></section>
    <section class="refine-row"><button class="primary refine-download" type="button">Download SVG</button><button class="primary refine-save" type="button">Save corrected SVG</button></section>
    <p class="meta refine-metrics">${esc(String(p.backend || 'legacy'))} · ${Number(m.regions || 0)} regions · ${((Number(m.ink_fraction || 0)) * 100).toFixed(1)}% black</p>
  </aside>`;
  const imgs = body.querySelectorAll('.refine-plate img');
  ['source.png', 'overlay.png', 'result.png', 'diff.png'].forEach((name, i) => { imgs[i].src = withToken(p.assets[name]); });
  // The region overlay and result are exact hit maps. Scale the displayed
  // coordinate back to the bounded working raster before asking the server to
  // toggle that one spatial region.
  [imgs[1], imgs[2]].forEach((image) => {
    image.classList.add('refine-clickable');
    image.title = 'Click a connected region to toggle it';
    image.addEventListener('click', (event) => {
      const box = image.getBoundingClientRect();
      if (!box.width || !box.height) return;
      const x = Math.max(0, Math.min(p.width - 1, Math.floor((event.clientX - box.left) * p.width / box.width)));
      const y = Math.max(0, Math.min(p.height - 1, Math.floor((event.clientY - box.top) * p.height / box.height)));
      runRefineAction({ x, y });
    });
  });
  const proposals = body.querySelector('.refine-proposals');
  (p.proposals || []).forEach((proposal) => proposals.appendChild(refineButton(
    `${proposal.label} · fills ${(Number(proposal.black_fraction) * 100).toFixed(1)}%`,
    () => runRefineAction({ proposal_id: proposal.id }))));
  if (!p.proposals?.length) proposals.innerHTML = '<p class="meta">No stable palette grouping found; use region controls.</p>';
  const groups = body.querySelector('.refine-groups');
  (p.groups || []).forEach((group) => groups.appendChild(refineButton(
    `Layer ${group.colour_label} · ${group.state} · ${group.regions} regions`,
    () => runRefineAction({ colour_label: group.colour_label }))));
  const regions = body.querySelector('.refine-list');
  [...(p.regions || [])].sort((a, b) => b.area - a.area).slice(0, 160).forEach((region) => regions.appendChild(refineButton(
    `#${region.region_id} · layer ${region.colour_label} · ${Number(region.area).toLocaleString()} px · ${region.solved_ink ? 'BLACK' : 'WHITE'}`,
    () => runRefineAction({ region_id: region.region_id }))));
  body.querySelector('.refine-download').addEventListener('click', () => { window.open(withToken(p.download), '_blank', 'noopener'); });
  body.querySelector('.refine-save').addEventListener('click', async () => {
    const dest = $('#dest').value.trim();
    if (!dest) return toast('Choose a Save to folder first.', true);
    const generation = refine.generation, base = refine.base;
    const r = await apiFetch(`${base}/save`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dest }) });
    const j = await r.json();
    if (generation !== refine.generation || base !== refine.base) return;
    if (!r.ok) return toast(j.detail || 'Could not save corrected SVG.', true);
    saveSettings({ dest }); toast(`Saved ${j.written?.[0] || 'corrected SVG'}.`);
  });
}

async function runRefineAction(action) {
  if (refine.busy || !refine.base) return;
  const generation = refine.generation, base = refine.base;
  refine.busy = true;
  try {
    const r = await apiFetch(`${base}/action`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(action) });
    const j = await r.json();
    if (generation !== refine.generation || base !== refine.base) return;
    if (!r.ok) return toast(j.detail || 'Could not apply refinement.', true);
    refine.payload = j; renderRefine();
  } finally {
    if (generation === refine.generation && base === refine.base) refine.busy = false;
  }
}

/* ---------------- full-size candidate viewer ----------------
 * the reviewer: "its hard to see the images to pick from when theyre in the pallet."
 * True and structural - a 150px plate is enough to rank a slate at a glance and
 * nowhere near enough to see whether a hairline survived or a counter closed,
 * which is the decision the tile is asking him to make.
 *
 * Two choices worth stating. It shows the candidate BESIDE THE ORIGINAL rather
 * than alone, because "is this right" is always a comparison and flipping back
 * and forth from memory is what the small tiles already force. And it can
 * select and star from inside the viewer, with arrow keys to move along the
 * slate - otherwise judging seven candidates means seven open/close cycles. */
const viewer = { key: null, idx: 0 };

function viewerImg(key) {
  return state.images.find((i) => i.key === key);
}

function openViewer(key, name) {
  const img = viewerImg(key);
  if (!img) return;
  viewer.key = key;
  viewer.idx = Math.max(0, img.candidates.findIndex((c) => c.name === name));
  $('#viewer').hidden = false;
  paintViewer();
  $('#v-close').focus();
}

function closeViewer() {
  $('#viewer').hidden = true;
  const t = document.querySelector(
    `.tile[data-stem="${cssEsc(viewer.key)}"][data-name="${cssEsc(
      (viewerImg(viewer.key)?.candidates[viewer.idx] || {}).name || '')}"] .expand`);
  if (t) t.focus();
  viewer.key = null;
}

function stepViewer(d) {
  const img = viewerImg(viewer.key);
  if (!img) return;
  viewer.idx = (viewer.idx + d + img.candidates.length) % img.candidates.length;
  paintViewer();
}

function paintViewer() {
  const img = viewerImg(viewer.key);
  if (!img) return closeViewer();
  const c = img.candidates[viewer.idx];
  const sel = state.sel[img.key] || new Set();

  $('#v-title').textContent = img.stem;
  $('#v-score').textContent = `${c.name} · score ${c.score}`;
  $('#v-img').src = c.preview;
  $('#v-img').alt = `${c.name} result for ${img.stem}`;
  $('#v-src').src = img.source || '';
  $('#v-src').closest('.v-pane').hidden = !img.source;
  $('#v-cap').textContent = `${c.n}. ${c.name}${c.headline ? ` — ${c.headline}` : ''}`;
  const setIdx = state.images.findIndex((i) => i.key === img.key);
  $('#v-pos').textContent = `${viewer.idx + 1} of ${img.candidates.length}`
    + (state.images.length > 1 ? ` · set ${setIdx + 1} of ${state.images.length}` : '');
  const last = setIdx >= state.images.length - 1;
  $('#v-nextset').disabled = last;
  $('#v-nextset').title = last ? 'This is the last set' : `Next: ${state.images[setIdx + 1].stem}`;

  const picked = sel.has(c.name);
  const starred = state.fav[img.key] === c.name;
  $('#v-pick').textContent = picked ? 'Shippable ✓' : 'Mark shippable';
  $('#v-pick').classList.toggle('on', picked);
  $('#v-star').textContent = starred ? 'Starred ★' : 'Star this one';
  $('#v-star').classList.toggle('on', starred);
}

$('#v-close').addEventListener('click', closeViewer);
$('#v-prev').addEventListener('click', () => stepViewer(-1));
$('#v-next').addEventListener('click', () => stepViewer(1));
/* Next set: the next IMAGE's slate, from its first candidate. Previous/Next
 * wrap within one image's candidates; this is how a batch is walked without
 * closing the viewer between logos. Stops at the last set rather than
 * wrapping, so reaching the end of a batch is visible. */
function nextSet() {
  const i = state.images.findIndex((x) => x.key === viewer.key);
  const nxt = state.images[i + 1];
  if (!nxt) return;
  // A collapsed card is still part of the batch; open it so closing the
  // viewer lands on the card being judged.
  toggleCollapse(nxt.key, false);
  viewer.key = nxt.key;
  viewer.idx = 0;
  paintViewer();
}
$('#v-nextset').addEventListener('click', nextSet);
$('#v-pick').addEventListener('click', () => {
  const img = viewerImg(viewer.key);
  toggle(img.key, img.candidates[viewer.idx].name);
  paintViewer();
});
$('#v-star').addEventListener('click', () => {
  const img = viewerImg(viewer.key);
  star(img.key, img.candidates[viewer.idx].name);
  paintViewer();
});

document.addEventListener('keydown', (e) => {
  if ($('#viewer').hidden) return;
  const keys = {
    Escape: closeViewer,
    ArrowLeft: () => stepViewer(-1),
    ArrowRight: () => stepViewer(1),
  };
  if (keys[e.key]) { e.preventDefault(); keys[e.key](); return; }
  // Space marks shippable, Enter stars - the two actions this view exists for,
  // reachable without going back to the mouse.
  if (e.key === ' ') { e.preventDefault(); $('#v-pick').click(); }
  if (e.key === 'Enter') { e.preventDefault(); $('#v-star').click(); }
});
