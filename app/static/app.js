/* T-Tracer - frontend.
 *
 * Interaction model, from the reviewer: click a candidate to mark it shippable, star
 * one as the favourite, and the favourite is what gets written to the chosen
 * folder. Marking several is deliberate - across three calibration batches the
 * average image had ~5 shippable candidates, and forcing a single choice threw
 * that away. */

const $ = (s) => document.querySelector(s);
// `images` ACCUMULATES across drops. Each entry carries the job it came from,
// because preview URLs and the save endpoint are both job-scoped. The first
// version replaced the list on every upload, so adding a second logo silently
// erased the first one along with its picks.
const state = { jobId: null, images: [], sel: {}, fav: {}, failed: {},
                touched: {} };

/* ---------------- destination folder ----------------
 * Kept SERVER-side rather than in localStorage, and that is the entire fix for
 * "the folder is forgotten every time I open it".
 *
 * The old code wrote localStorage correctly and it still never survived a
 * restart: main.py binds a FREE PORT on every launch, localStorage is keyed by
 * ORIGIN, and http://127.0.0.1:41337 is a different origin from
 * http://127.0.0.1:39112. Every start was a fresh, empty store. A settings
 * file next to the work directory does not care which port the app got. */
let settings = { dest: '', share_stats: false, role: 'user' };

async function loadSettings() {
  try {
    settings = await (await fetch('/api/settings')).json();
    $('#dest').value = settings.dest || '';
  } catch (_) { /* defaults are fine */ }
  refreshAction();
}

async function saveSettings(patch) {
  Object.assign(settings, patch);
  try {
    await fetch('/api/settings', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  } catch (_) { /* a lost preference is not worth an error */ }
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
    const r = await fetch('/api/pick-folder');
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
    const r = await fetch('/api/jobs', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const j = await r.json();
    state.jobId = j.job_id;
    if (j.rejected?.length) toast(`Skipped ${j.rejected.length} non-image file(s).`, true);
    poll(j.job_id);
  } catch (err) {
    $('#progress').hidden = true;
    toast(String(err.message || err), true);
  }
}

/* ---------------- progress ---------------- */
async function poll(jobId) {
  const r = await fetch(`/api/jobs/${jobId}`);
  const j = await r.json();
  const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
  $('#barfill').style.width = `${Math.max(pct, 3)}%`;
  $('#progresstext').textContent =
    j.status === 'ready' ? `Traced ${j.total} image${j.total === 1 ? '' : 's'}`
                         : `Tracing ${j.done} of ${j.total}…`;
  if (j.status === 'tracing') return setTimeout(() => poll(jobId), 700);
  if (j.errors?.length) toast(j.errors[0], true);
  setTimeout(() => { $('#progress').hidden = true; }, 900);
  loadResults(jobId);
}

/* ---------------- results ---------------- */
async function loadResults(jobId) {
  const r = await fetch(`/api/jobs/${jobId}/results`);
  const fresh = (await r.json()).images;
  const box = $('#results');
  box.hidden = false;

  fresh.forEach((img) => {
    img.jobId = jobId;
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
  right.append(st, none, caret, close);
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
  const card = document.querySelector(`[data-card="${cssEsc(key)}"]`);
  if (card) card.remove();
  if (!state.images.length) $('#results').hidden = true;
  refreshAction();
}

function markFailed(key) {
  const failed = !state.failed[key];
  state.failed[key] = failed;
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
      const r = await fetch(`/api/jobs/${jobId}/save`, {
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
               stats: $('#tab-stats') };

function showTab(which) {
  Object.entries(TABS).forEach(([name, el]) => {
    el.classList.toggle('on', name === which);
    el.setAttribute('aria-selected', String(name === which));
  });
  const cur = which === 'current';
  $('#history').hidden = which !== 'history';
  $('#stats').hidden = which !== 'stats';
  $('#drop').hidden = !cur;
  $('#howto').hidden = !cur;
  $('#results').hidden = !cur || !state.images.length;
  $('#actionbar').hidden = !cur || !state.images.length;
  if (which === 'history') loadHistory();
  if (which === 'stats') loadStats();
}
Object.keys(TABS).forEach((name) =>
  TABS[name].addEventListener('click', () => showTab(name)));

async function loadHistory() {
  const box = $('#history');
  box.innerHTML = '<p class="hempty">Loading…</p>';
  try {
    const { jobs } = await (await fetch('/api/history')).json();
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
        await fetch(`/api/jobs/${j.id}`, { method: 'DELETE' });
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
  };
}

async function recordJudgement(imgs) {
  const byJob = {};
  imgs.forEach((img) => { (byJob[img.jobId] ||= []).push(judgementFor(img)); });
  for (const [jobId, rows] of Object.entries(byJob)) {
    try {
      await fetch(`/api/jobs/${jobId}/judge`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rows),
      });
    } catch (_) { /* never let bookkeeping break the actual work */ }
  }
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
    d = await (await fetch('/api/stats')).json();
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
      const r = await fetch('/api/stats/report');
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


/* ---------------- update check (item 6) ----------------
 * Reports, never installs. One call to the GitHub releases API at startup; if
 * it fails, is rate limited, or there are no releases yet, the bar simply does
 * not appear. Downloading is a link the user clicks, so nothing rewrites
 * itself behind their back. */
async function checkUpdate() {
  let d;
  try {
    d = await (await fetch('/api/update-check')).json();
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
  $('#v-pos').textContent = `${viewer.idx + 1} of ${img.candidates.length}`;

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
