/**
 * deck-kit.js — presenter layer that works on top of every template.
 *
 * Drop it in after the deck's own script:
 *   <script src="deck-kit.js"></script>
 *
 * It finds the deck in one of three ways, in this order:
 *   1. <deck-stage>          — uses the element's goTo()/index API.
 *   2. window.deckAdapter    — { go(i), index(), slides?, nativeScroll? } set by
 *                              templates that ship their own navigation.
 *   3. generic fallback      — top-level `.slide` elements, current slide = the
 *                              one with .active / .is-active; moves by
 *                              synthetic arrow keys.
 *
 * Presenting
 *   ← → Space PgUp/PgDn Home End 1–9   navigation (deck-kit owns it for
 *        adapter/generic decks, so presses during a template's transition are
 *        queued instead of dropped)
 *   P    presenter view: second window with current + next slide, notes,
 *        timer; the original window becomes the clean audience screen
 *   N    speaker notes panel (on this screen)     M  slide menu
 *   T    talk timer (Shift+T resets, click to pause)
 *   B/W  black / white screen    L  laser pointer    D  draw on the slide
 *        (C clears the slide's ink, Backspace undoes a stroke, Esc stops)
 *   F    fullscreen    ?  shortcuts    Esc  close panels
 *   #N in the URL opens slide N. Swipe on touch screens.
 *   Ctrl/Cmd+P visits every slide first (so charts drawn on entry are
 *   filled in), then prints one slide per page with canvases kept.
 *
 * Slide metadata (all optional)
 *   data-title  data-sec  data-min   and   <aside class="notes">…</aside>
 *
 * Interactive slides
 *   DeckKit.widget(target, { enter, leave, draw, tick(dt), print })
 *   DeckKit.canvas(cv) → { ctx, w, h }     crisp canvas in CSS pixels
 *   DeckKit.bind(inputs, fn)               sliders → fn(values), live <output>s
 *   DeckKit.plot(cv, { x:[a,b], y:[c,d] }) axes, functions, points, gaussians…
 *   DeckKit.palette(el)                    the template's ink / bg / accents
 *   [data-reveal="id"]  [data-countdown="5:00" data-autostart]
 *
 * UI strings: Vietnamese when the deck is lang="vi" or the browser prefers
 * Vietnamese, English otherwise (<meta name="deck-kit-lang"> overrides).
 */

(() => {
  if (window.DeckKit) return;

  // Windows opened by the presenter view: ?dk=presenter (notes screen) and
  // ?dk=mirror (slide previews inside it).
  const ROLE = (new URLSearchParams(location.search).get('dk') || 'main').toLowerCase();
  // A copy published by deploy.mjs: no file to save into, so no editing or export tools.
  const PUBLISHED = !!document.querySelector('meta[name="deck-kit-published"]');

  const STR = {
    en: {
      notes: 'Speaker notes', noNotes: 'No notes for this slide.', next: 'Next', end: 'End of deck',
      menu: 'Slides', min: 'min', planned: 'planned {m} min', startsAt: 'starts at min {s}',
      elapsed: 'Elapsed', ahead: 'ahead', behind: 'behind', onTime: 'on time', paused: 'paused',
      rotate: 'Rotate your phone to landscape for the best view', close: 'Close',
      reveal: 'Show answer', hide: 'Hide answer',
      help: 'Shortcuts', helpHint: 'Press ? for presenter shortcuts',
      printing: 'Preparing print… {i}/{n}', popup: 'Allow pop-ups to open the presenter view',
      presenter: 'Presenter view', waiting: 'Waiting for the audience window…', lost: 'Audience window closed',
      current: 'Current', drawOn: 'Drawing — C clears, Esc stops', laserOn: 'Laser pointer — L to stop',
      edit: '✎ Edit', done: '✓ Done', saveFile: '💾 Save to file', unsaved: '{n} unsaved', saved: 'Notes saved to {f}',
      nothing: 'No note changes to save', restored: 'Restored {n} unsaved note edit(s) — press P, then Save',
      otherFile: 'You picked "{a}" but this deck is "{b}". Write to it anyway?',
      downloaded: 'Saved notes.json — merge it with: node scripts/apply-changes.mjs <deck.html> notes.json',
      saveFailed: 'Could not save: {e}', pickDeck: 'Pick this deck\'s own HTML file to save into',
      editBar: '✎ Editing — click to select · drag to move · handles resize · double-click to type · Del hides · Ctrl+Z undo · <b>E</b> saves & exits',
      editN: '{n} change(s)', editSaved: 'Saved {n} change(s) to {f}', editNotSaved: 'Not saved — your edits are kept as a draft',
      editRestored: 'Restored {n} unsaved edit(s) — press E, then E again to save',
      editDownloaded: 'Saved edits.json — merge it with: node scripts/apply-changes.mjs <deck.html> edits.json',
      pick: 'Click an element to copy its location for your agent (Esc cancels)', copied: 'Copied: {t}',
      share: 'Export & share', printNow: 'Print PDF now', print: 'Print', exportFiles: 'Export files', deploy: 'Deploy',
      fPdf: 'PDF', fCompact: 'Small PDF', fPng: 'PNG', fHandout: 'Handout', fPptx: 'PPTX (images)', fPptxEd: 'PPTX (editable)',
      previewOpt: 'Preview link', nameLbl: 'Name', copyCmd: 'Copy command', pickFormat: 'Tick at least one format',
      cmdCopied: 'Command copied — paste it in a terminal or send it to your agent',
      stale: 'The deck file changed on disk after you opened it (the agent may have edited it). Reload the page and save again — your changes are kept as a draft.',
      editPartial: 'Saved {n} change(s) to {f}; {m} element(s) were created by the template\'s script and are not in the file, so they could not be saved',
      keepNotes: 'Include speaker notes',
      step: 'step', copy: 'Copy', copiedCode: 'Copied', play: 'Play', pause: 'Pause', stepNext: 'Next step', resetT: 'Back to start',
      compare: 'Before / after slider', zoomHint: 'Drag a box (or click) to zoom in — Z or Esc to zoom out',
      shareNote: 'Exports and deploys run with Node on your computer; deploying makes the deck public.',
      keys: [['← → 1–9', 'Navigate (→ also reveals the next step)'], ['P', 'Presenter view (2nd window)'], ['N', 'Speaker notes'], ['M', 'Slide menu'],
        ['T', 'Talk timer (Shift+T reset)'], ['B / W', 'Black / white screen'], ['L', 'Laser pointer'],
        ['D', 'Draw (C clear, Backspace undo)'], ['E', 'Edit mode (E again saves)'], ['K', 'Copy an element\'s location for your agent'], ['X', 'Export & share (PDF, PPTX, deploy)'], ['Z', 'Zoom into part of the slide'], ['F', 'Fullscreen'], ['Ctrl+P', 'Print every slide'], ['Esc', 'Close panels'], ['?', 'This help']],
    },
    vi: {
      notes: 'Ghi chú người nói', noNotes: 'Slide này không có ghi chú.', next: 'Tiếp theo', end: 'Hết bài',
      menu: 'Các slide', min: 'phút', planned: 'dự kiến {m} phút', startsAt: 'bắt đầu ở phút {s}',
      elapsed: 'Đã nói', ahead: 'nhanh hơn', behind: 'chậm hơn', onTime: 'đúng giờ', paused: 'tạm dừng',
      rotate: 'Xoay ngang điện thoại để xem rõ hơn', close: 'Đóng',
      reveal: 'Xem đáp án', hide: 'Ẩn đáp án',
      help: 'Phím tắt', helpHint: 'Bấm ? để xem phím tắt trình chiếu',
      printing: 'Đang chuẩn bị in… {i}/{n}', popup: 'Hãy cho phép cửa sổ bật lên (pop-up) để mở màn hình người trình bày',
      presenter: 'Màn hình người trình bày', waiting: 'Đang chờ cửa sổ trình chiếu…', lost: 'Cửa sổ trình chiếu đã đóng',
      current: 'Hiện tại', drawOn: 'Đang vẽ — C xoá, Esc thoát', laserOn: 'Con trỏ laser — bấm L để tắt',
      edit: '✎ Sửa', done: '✓ Xong', saveFile: '💾 Lưu vào file', unsaved: '{n} chưa lưu', saved: 'Đã lưu ghi chú vào {f}',
      nothing: 'Không có ghi chú nào thay đổi', restored: 'Đã khôi phục {n} ghi chú chưa lưu — bấm P rồi Lưu',
      otherFile: 'Bạn chọn "{a}" nhưng deck này là "{b}". Vẫn ghi vào file đó?',
      downloaded: 'Đã tải notes.json — ghép vào deck bằng: node scripts/apply-changes.mjs <deck.html> notes.json',
      saveFailed: 'Không lưu được: {e}', pickDeck: 'Chọn chính file HTML của deck này để lưu',
      editBar: '✎ Đang chỉnh sửa — click để chọn · kéo để di chuyển · kéo núm để đổi cỡ · double-click để gõ · Del ẩn · Ctrl+Z hoàn tác · <b>E</b> lưu & thoát',
      editN: '{n} thay đổi', editSaved: 'Đã lưu {n} thay đổi vào {f}', editNotSaved: 'Chưa lưu — thay đổi được giữ làm bản nháp',
      editRestored: 'Đã khôi phục {n} chỉnh sửa chưa lưu — bấm E rồi E lần nữa để lưu',
      editDownloaded: 'Đã tải edits.json — ghép vào deck bằng: node scripts/apply-changes.mjs <deck.html> edits.json',
      pick: 'Click vào phần tử để chép vị trí cho agent (Esc huỷ)', copied: 'Đã chép: {t}',
      share: 'Xuất & chia sẻ', printNow: 'In PDF ngay', print: 'In', exportFiles: 'Xuất file', deploy: 'Deploy',
      fPdf: 'PDF', fCompact: 'PDF nén', fPng: 'PNG', fHandout: 'Handout', fPptx: 'PPTX ảnh', fPptxEd: 'PPTX sửa được',
      previewOpt: 'Bản xem thử', nameLbl: 'Tên', copyCmd: 'Chép lệnh', pickFormat: 'Hãy tick ít nhất một định dạng',
      cmdCopied: 'Đã chép lệnh — dán vào terminal hoặc gửi cho agent chạy',
      stale: 'File deck đã thay đổi trên ổ đĩa sau khi bạn mở (có thể agent vừa sửa). Hãy tải lại trang rồi lưu lại — thay đổi của bạn vẫn được giữ làm bản nháp.',
      editPartial: 'Đã lưu {n} thay đổi vào {f}; {m} phần tử do script của mẫu tự tạo nên không có trong file, không lưu được',
      keepNotes: 'Kèm ghi chú người nói',
      step: 'bước', copy: 'Chép', copiedCode: 'Đã chép', play: 'Phát', pause: 'Tạm dừng', stepNext: 'Bước tiếp', resetT: 'Về đầu',
      compare: 'Thanh kéo so sánh trước / sau', zoomHint: 'Kéo khung (hoặc click) để phóng to — Z hoặc Esc để thu nhỏ',
      shareNote: 'Xuất file và deploy chạy bằng Node trên máy bạn; deploy sẽ công khai deck.',
      keys: [['← → 1–9', 'Chuyển slide (→ cũng hiện bước tiếp theo)'], ['P', 'Màn hình người trình bày (cửa sổ 2)'], ['N', 'Ghi chú người nói'], ['M', 'Menu các slide'],
        ['T', 'Đồng hồ (Shift+T đặt lại)'], ['B / W', 'Màn hình đen / trắng'], ['L', 'Con trỏ laser'],
        ['D', 'Vẽ lên slide (C xoá, Backspace hoàn tác)'], ['E', 'Chỉnh sửa (E lần nữa để lưu)'], ['K', 'Chép vị trí phần tử cho agent'], ['X', 'Xuất & chia sẻ (PDF, PPTX, deploy)'], ['Z', 'Phóng to một vùng của slide'], ['F', 'Toàn màn hình'], ['Ctrl+P', 'In mọi slide'], ['Esc', 'Đóng bảng'], ['?', 'Bảng này']],
    },
  };
  // Panel language: Vietnamese when the deck is in Vietnamese or the
  // presenter's browser prefers Vietnamese (the controls are for the person
  // presenting); English otherwise. <meta name="deck-kit-lang" content="en|vi">
  // forces one.
  const uiLang = (() => {
    const forced = document.querySelector('meta[name="deck-kit-lang"]');
    if (forced && STR[forced.content]) return forced.content;
    const deck = (document.documentElement.lang || '').slice(0, 2).toLowerCase();
    if (deck === 'vi') return 'vi';
    const browser = ((navigator.languages && navigator.languages[0]) || navigator.language || '').slice(0, 2).toLowerCase();
    return browser === 'vi' ? 'vi' : 'en';
  })();
  const L = STR[uiLang];
  const fmt = (s, o) => s.replace(/\{(\w+)\}/g, (_, k) => o[k]);
  const mmss = (sec) => {
    const s = Math.max(0, Math.round(sec));
    return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  };
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Page-level rules: hide authored notes, print helpers, countdown look.
  const css = `
    aside.notes { display: none !important; }
    .dk-print-img { display: none; }
    [data-countdown] { cursor: pointer; font-variant-numeric: tabular-nums; }
    [data-countdown].dk-done { opacity: .55; }
    html.dk-laser, html.dk-laser * { cursor: none !important; }
    html.dk-pick, html.dk-pick * { cursor: crosshair !important; }
    html.dk-editing [contenteditable] { outline: none; cursor: text; }
    /* steps */
    :where([data-step], [data-steps] > *) { transition: opacity .35s ease, visibility 0s; }
    .dk-step-off { opacity: 0 !important; visibility: hidden !important; pointer-events: none !important; transition: opacity .2s ease, visibility 0s linear .2s; }
    /* math */
    .dk-math[data-display] { display: block; text-align: center; margin: .45em 0; }
    /* code */
    :where(pre.dk-code) { position: relative; }
    .dk-code .dk-tk-kw { color: var(--dk-c-kw); } .dk-code .dk-tk-str { color: var(--dk-c-str); }
    .dk-code .dk-tk-num { color: var(--dk-c-num); } .dk-code .dk-tk-fn { color: var(--dk-c-fn); }
    .dk-code .dk-tk-com { color: var(--dk-c-com); font-style: italic; } .dk-code .dk-tk-attr { color: var(--dk-c-attr); }
    .dk-ln { transition: opacity .3s ease; } .dk-ln.dk-dim { opacity: .28; }
    .dk-copy { position: absolute; top: .5em; right: .5em; z-index: 1; font: 600 max(12px, .45em)/1 system-ui, sans-serif; padding: .45em .8em;
      border: 0; border-radius: 6px; background: rgba(127,127,127,.22); color: inherit; cursor: pointer; opacity: 0; transition: opacity .2s; }
    pre:hover > .dk-copy, .dk-copy:focus-visible { opacity: 1; }
    @media (hover: none) { .dk-copy { opacity: .75; } }
    /* quiz */
    .dk-opt { cursor: pointer; transition: opacity .2s, box-shadow .2s, background-color .2s; }
    .dk-quiz.dk-answered .dk-opt:not(.dk-right):not(.dk-wrong) { opacity: .5; }
    .dk-opt.dk-right { box-shadow: inset 0 0 0 3px var(--dk-right, #16a34a); background-color: color-mix(in srgb, var(--dk-right, #16a34a) 14%, transparent); }
    .dk-opt.dk-wrong { box-shadow: inset 0 0 0 3px var(--dk-wrong, #dc2626); background-color: color-mix(in srgb, var(--dk-wrong, #dc2626) 12%, transparent); }
    .dk-opt.dk-right::after { content: " ✓"; color: var(--dk-right, #16a34a); font-weight: 700; }
    .dk-opt.dk-wrong::after { content: " ✗"; color: var(--dk-wrong, #dc2626); font-weight: 700; }
    .dk-quiz:not(.dk-answered) [data-explain] { display: none; }
    /* charts */
    :where(canvas[data-chart], canvas.dk-chart) { display: block; width: 100%; aspect-ratio: 16 / 9; }
    .dk-chart-box { width: 100%; } .dk-chart-box[style*="height"] canvas { height: 100%; aspect-ratio: auto; }
    .dk-chart-src { position: absolute !important; left: 0 !important; top: 0 !important; transform: scale(0) !important; transform-origin: 0 0 !important;
      opacity: 0 !important; pointer-events: none !important; }   /* tables can't shrink below their content; scale it away, keep it for screen readers */
    /* stepper */
    .dk-stepper { display: flex; gap: .4em; align-items: center; margin-top: .5em; font: 600 max(14px, .5em)/1 system-ui, sans-serif; }
    .dk-stepper button { font: inherit; min-width: 2.6em; padding: .5em .7em; border: 1.5px solid currentColor; border-radius: 6px;
      background: transparent; color: inherit; cursor: pointer; opacity: .85; }
    .dk-stepper button:hover { opacity: 1; } .dk-stepper button:disabled { opacity: .35; cursor: default; }
    .dk-stepper-n { margin-left: .4em; font-variant-numeric: tabular-nums; opacity: .7; }
    .dk-frame-off { display: none !important; }
    /* before / after */
    :where(.dk-cmp) { position: relative; display: grid; overflow: hidden; }
    .dk-cmp > .dk-cmp-before, .dk-cmp > .dk-cmp-after { grid-area: 1 / 1; width: 100%; height: 100%; object-fit: cover; }
    .dk-cmp > .dk-cmp-after { clip-path: inset(0 0 0 var(--dk-cmp, 50%)); }
    .dk-cmp-range { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize; z-index: 2; }
    .dk-cmp-handle { position: absolute; top: 0; bottom: 0; left: var(--dk-cmp, 50%); width: 3px; margin-left: -1.5px; background: #fff;
      box-shadow: 0 0 0 1px rgba(0,0,0,.25); pointer-events: none; z-index: 1; }
    .dk-cmp-handle::after { content: "↔"; position: absolute; top: 50%; left: 50%; width: 44px; height: 44px; margin: -22px 0 0 -22px;
      border-radius: 50%; background: #fff; color: #111; display: grid; place-items: center; font: 700 20px/1 system-ui, sans-serif;
      box-shadow: 0 2px 10px rgba(0,0,0,.3); }
    .dk-cmp-range:focus-visible ~ .dk-cmp-handle::after { outline: 3px solid #3B82F6; }
    .dk-cmp-label { position: absolute; top: .6em; z-index: 1; padding: .2em .6em; border-radius: 4px; background: rgba(0,0,0,.55); color: #fff;
      font: 600 max(13px, .5em)/1.2 system-ui, sans-serif; pointer-events: none; }
    .dk-cmp-l { left: .6em; } .dk-cmp-r { right: .6em; }
    /* hotspots */
    :where(.dk-hs) { position: relative; }
    .dk-hs-dot { position: absolute; left: var(--dk-x); top: var(--dk-y); z-index: 2; width: 2em; height: 2em; margin: -1em 0 0 -1em; padding: 0;
      border-radius: 50%; border: 2px solid var(--dk-hs-fg, #fff); background: var(--dk-hs-bg, #2563EB); color: var(--dk-hs-fg, #fff);
      font: 700 max(13px, .5em)/1 system-ui, sans-serif; display: grid; place-items: center; cursor: pointer;
      box-shadow: 0 2px 8px rgba(0,0,0,.3); animation: dk-pulse 2.4s ease-out infinite; }
    .dk-hs-dot[aria-expanded="true"] { animation: none; outline: 3px solid var(--dk-hs-bg, #2563EB); outline-offset: 2px; }
    @keyframes dk-pulse { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--dk-hs-bg, #2563EB) 55%, transparent); } 70%, 100% { box-shadow: 0 0 0 .7em transparent; } }
    .dk-hs-tip { position: absolute; left: var(--dk-x); top: var(--dk-y); z-index: 3; width: max-content; max-width: min(45%, 24em); margin: 0;
      transform: translate(1.4em, -50%); padding: .55em .85em; border-radius: 8px; background: var(--dk-tip-bg, #fff); color: var(--dk-tip-fg, #111);
      box-shadow: 0 6px 24px rgba(0,0,0,.28); transition: opacity .2s, visibility 0s; }
    .dk-hs-tip.dk-hs-left { transform: translate(calc(-100% - 1.4em), -50%); }
    .dk-hs-tip:not(.dk-open) { opacity: 0; visibility: hidden; pointer-events: none; transition: opacity .2s, visibility 0s linear .2s; }
    /* QR */
    :where([data-qr]) { display: inline-block; width: 8em; aspect-ratio: 1; }
    [data-qr] > svg.dk-qr { display: block; width: 100%; height: 100%; }
    .dk-qr-fallback { font: .5em/1.3 ui-monospace, monospace; word-break: break-all; }
    @media (prefers-reduced-motion: reduce) {
      :where([data-step], [data-steps] > *), .dk-step-off, .dk-ln, .dk-hs-tip { transition: none !important; }
      .dk-hs-dot { animation: none; }
    }
    @media print {
      * { transition: none !important; }
      .dk-copy, .dk-stepper, .dk-cmp-range { display: none !important; }
      .dk-hs-dot { animation: none !important; }
      #deck-kit { display: none !important; }
      .dk-print-src { display: none !important; }
      .dk-print-img { display: block !important; }
    }
  `;
  // The presenter window keeps the deck loaded (for notes) but shows only our UI.
  const presenterPageCss = `
    html, body { overflow: hidden !important; background: #0b1220 !important; }
    body > *:not(#deck-kit) { visibility: hidden !important; }
  `;

  // Presenter UI lives in a shadow root so template CSS (button, h2…) can't restyle it.
  const uiCss = `
    :host { all: initial; font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; color: #fff; }
    * { box-sizing: border-box; }
    .dk-progress { position: fixed; left: 0; top: 0; height: 3px; width: 0; z-index: 2147483100;
      background: #3B82F6; transition: width 300ms ease; pointer-events: none; }
    .dk-panel { position: fixed; z-index: 2147483200; background: rgba(15, 23, 42, .96); color: #fff;
      box-shadow: 0 12px 40px rgba(0,0,0,.35); display: none; }
    .dk-panel.dk-show { display: block; }
    .dk-notes { left: 0; right: 0; bottom: 0; max-height: 38vh; overflow: auto; padding: 14px 28px 18px; }
    .dk-hd { display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; color: #93C5FD; font-weight: 600; margin-bottom: 6px; }
    .dk-hd small { color: #94A3B8; font-weight: 400; }
    .dk-tx { font-size: 18px; line-height: 1.5; max-width: 1100px; }
    .dk-tx p + p { margin-top: .6em; }
    .dk-next { margin-top: 10px; color: #94A3B8; font-size: 14px; }
    .dk-menu { inset: 0; overflow: auto; padding: 32px 40px; background: rgba(15, 23, 42, .97); }
    .dk-menu h2 { font-size: 22px; margin: 0 0 14px; }
    .dk-menu .dk-items { columns: 2 340px; column-gap: 36px; }
    .dk-menu .dk-sec { color: #93C5FD; font-weight: 700; margin: 12px 0 2px; break-after: avoid; }
    .dk-menu button { display: flex; gap: 12px; width: 100%; text-align: left; font: inherit; color: #E2E8F0;
      background: none; border: 0; border-bottom: 1px solid rgba(148,163,184,.2); padding: 6px 4px; cursor: pointer; break-inside: avoid; }
    .dk-menu button:hover, .dk-menu button:focus-visible { background: rgba(59,130,246,.18); outline: none; }
    .dk-menu button.dk-cur { color: #fff; background: rgba(59,130,246,.32); }
    .dk-menu button span:first-child { width: 64px; flex: none; color: #94A3B8; font-variant-numeric: tabular-nums; }
    .dk-timer { top: 12px; right: 12px; border-radius: 10px; padding: 6px 12px; cursor: pointer; user-select: none;
      font-variant-numeric: tabular-nums; display: none; }
    .dk-timer.dk-show { display: flex; gap: 10px; align-items: baseline; }
    .dk-timer b { font-size: 22px; }
    .dk-ahead { color: #4ADE80; } .dk-behind { color: #F87171; } .dk-ontime { color: #93C5FD; } .dk-paused { color: #FBBF24; }
    /* Dialogs (? help, X export & share) share one look. */
    .dk-dialog { left: 50%; top: 50%; transform: translate(-50%, -50%); border-radius: 14px; padding: 18px 24px 16px;
      width: min(600px, 92vw); max-height: 88vh; overflow: auto; }
    .dk-dialog h3 { margin: 0; padding-bottom: 12px; font-size: 18px; font-weight: 700; border-bottom: 1px solid rgba(148,163,184,.25); }
    .dk-dialog .dk-row { display: grid; grid-template-columns: 120px 1fr; gap: 8px 14px; align-items: start;
      padding: 9px 0; border-bottom: 1px solid rgba(148,163,184,.12); }
    .dk-dialog .dk-row:last-of-type { border-bottom: 0; }
    .dk-dialog .dk-row > b { color: #93C5FD; font-weight: 600; padding-top: 2px; }
    .dk-dialog button { font: inherit; font-weight: 600; color: #fff; background: rgba(148,163,184,.16); border: 0;
      border-radius: 8px; padding: 6px 14px; cursor: pointer; }
    .dk-dialog button:hover { background: rgba(148,163,184,.3); }
    .dk-dialog .dk-note { color: #94A3B8; font-size: 12px; margin-top: 10px; }
    .dk-help .dk-row { padding: 6px 0; }
    .dk-help .dk-row > b kbd { color: #93C5FD; }
    kbd { font: 600 13px ui-monospace, Menlo, Consolas, monospace; background: rgba(255,255,255,.14); border-radius: 5px; padding: 2px 7px; }
    .dk-toast { left: 50%; top: 16px; transform: translateX(-50%); border-radius: 999px; padding: 7px 16px; font-size: 14px; pointer-events: none; }
    .dk-rotate { left: 8px; right: 8px; top: 8px; border-radius: 10px; padding: 10px 14px; font-weight: 600; }
    .dk-rotate button { margin-left: 10px; font: inherit; border: 0; border-radius: 6px; padding: 2px 10px; }
    .dk-blank { position: fixed; inset: 0; z-index: 2147483150; display: none; cursor: none; }
    .dk-blank.dk-show { display: block; }
    .dk-ink { position: fixed; inset: 0; z-index: 2147483140; pointer-events: none; width: 100vw; height: 100vh; }
    .dk-ink.dk-drawing { pointer-events: auto; cursor: crosshair; }
    .dk-laser { position: fixed; z-index: 2147483160; width: 18px; height: 18px; margin: -9px 0 0 -9px; border-radius: 50%;
      background: radial-gradient(circle, #fff 0 18%, #ff2a2a 30% 55%, rgba(255,42,42,0) 72%); pointer-events: none; display: none;
      box-shadow: 0 0 14px 6px rgba(255,42,42,.45); }
    .dk-laser.dk-show { display: block; }
    .dk-zl { position: fixed; inset: 0; z-index: 2147483130; display: none; cursor: zoom-in; touch-action: none; }
    .dk-zl.dk-show { display: block; }
    .dk-zb { position: fixed; display: none; border: 2px dashed #3B82F6; background: rgba(59,130,246,.12); pointer-events: none; }
    /* export & share */
    .dk-share .dk-row { padding: 12px 0; }
    .dk-share .dk-opts { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
    .dk-share label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
    .dk-share input[type=text] { font: inherit; color: #fff; background: rgba(148,163,184,.14); border: 1px solid rgba(148,163,184,.3); border-radius: 6px; padding: 3px 8px; width: 180px; }
    .dk-share code { display: block; margin-top: 8px; font: 12px/1.45 ui-monospace, Menlo, monospace; color: #CBD5E1; background: rgba(0,0,0,.25);
      border-radius: 6px; padding: 6px 8px; white-space: pre-wrap; word-break: break-all; }
    /* edit mode */
    .dk-edbar { top: 0; left: 50%; transform: translateX(-50%); border-radius: 0 0 12px 12px; padding: 8px 16px; font-size: 14px; white-space: nowrap; }
    .dk-edbar.dk-show { display: flex; gap: 14px; align-items: center; }
    .dk-edbar b { color: #93C5FD; } .dk-edbar .dk-edn { color: #FBBF24; }
    .dk-hov { position: fixed; z-index: 2147483120; border: 1.5px dashed #3B82F6; pointer-events: none; display: none; box-sizing: border-box; }
    .dk-sel { position: fixed; z-index: 2147483125; border: 2px solid #3B82F6; display: none; cursor: move; box-sizing: border-box; touch-action: none; }
    .dk-sel.dk-typing { pointer-events: none; border-style: dotted; }
    .dk-sel .dk-h { position: absolute; width: 12px; height: 12px; margin: -7px 0 0 -7px; background: #fff; border: 2px solid #3B82F6; border-radius: 3px; touch-action: none; }
    .dk-sel.dk-typing .dk-h { display: none; }
    .dk-sel .dk-tag { position: absolute; left: -2px; top: -24px; font: 600 11px ui-monospace, Menlo, monospace; background: #3B82F6; color: #fff;
      padding: 2px 6px; border-radius: 4px; white-space: nowrap; pointer-events: none; }
    /* presenter window */
    .dk-pv { position: fixed; inset: 0; z-index: 2147483300; background: #0b1220; display: grid; gap: 18px; padding: 18px;
      grid-template-columns: minmax(0, 1.7fr) minmax(280px, 1fr); grid-template-rows: auto minmax(0, 1fr) auto; }
    .dk-pv .dk-lbl { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #94A3B8; margin-bottom: 6px; }
    .dk-pv .dk-top { grid-column: 1 / -1; display: flex; align-items: center; gap: 18px; }
    .dk-pv .dk-top .dk-title { font-size: 18px; font-weight: 600; flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .dk-pv .dk-clock { font: 600 30px ui-monospace, Menlo, Consolas, monospace; cursor: pointer; }
    .dk-pv .dk-status { font-size: 15px; }
    .dk-pv .dk-main, .dk-pv .dk-side { min-height: 0; display: flex; flex-direction: column; }
    .dk-pv .dk-side { gap: 14px; }
    .dk-pv .dk-frame { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 10px; overflow: hidden;
      box-shadow: 0 0 0 1px rgba(148,163,184,.25); }
    .dk-pv .dk-frame iframe { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; border: 0; transform-origin: 0 0; pointer-events: none; }
    .dk-pv .dk-cur iframe { pointer-events: auto; } /* drive sliders/buttons from the laptop */
    .dk-pv .dk-pvzl { position: absolute; inset: 0; display: none; cursor: zoom-in; touch-action: none; background: rgba(59,130,246,.06); }
    .dk-pv .dk-pvzl.dk-show { display: block; }
    .dk-pv .dk-pvzl i { position: absolute; display: none; border: 2px dashed #3B82F6; background: rgba(59,130,246,.15); pointer-events: none; }
    .dk-pv .dk-frame .dk-endcard { position: absolute; inset: 0; display: none; place-items: center; color: #94A3B8; font-size: 18px; }
    .dk-pv .dk-frame.dk-end .dk-endcard { display: grid; }
    .dk-pv .dk-frame.dk-end iframe { visibility: hidden; }
    .dk-pv .dk-pnotes { flex: 1; min-height: 0; overflow: auto; background: rgba(148,163,184,.08); border-radius: 10px; padding: 14px 18px; }
    .dk-pv .dk-pnotes .dk-tx { font-size: 22px; min-height: 2em; outline: none; }
    .dk-pv .dk-pnotes.dk-editing { box-shadow: inset 0 0 0 2px #3B82F6; background: rgba(59,130,246,.08); }
    .dk-pv .dk-nrow { display: flex; align-items: baseline; gap: 10px; }
    .dk-pv .dk-nrow .dk-lbl { margin: 0; flex: 1; }
    .dk-pv .dk-nrow button { font: 600 13px system-ui, sans-serif; color: #fff; background: rgba(148,163,184,.16); border: 0;
      border-radius: 6px; padding: 4px 10px; cursor: pointer; }
    .dk-pv .dk-nrow button:hover { background: rgba(148,163,184,.3); }
    .dk-pv .dk-nstat { font-size: 12px; color: #FBBF24; }
    .dk-pv .dk-bar { grid-column: 1 / -1; display: flex; gap: 10px; align-items: center; }
    .dk-pv .dk-bar button { font: 600 15px inherit; font-family: inherit; color: #fff; background: rgba(148,163,184,.16); border: 0;
      border-radius: 8px; padding: 8px 16px; cursor: pointer; }
    .dk-pv .dk-bar button:hover { background: rgba(148,163,184,.3); }
    .dk-pv .dk-bar .dk-pos { margin-left: auto; color: #94A3B8; font-variant-numeric: tabular-nums; }
    .dk-pv .dk-msg { color: #FBBF24; font-size: 14px; }
  `;

  // Generic print: lay every slide out as its own page. Only used when the
  // deck isn't a <deck-stage> (which already prints one slide per page).
  const genericPrintCss = `
    @page { size: 1920px 1080px; margin: 0; }
    @media print {
      html, body { width: auto !important; height: auto !important; overflow: visible !important;
        margin: 0 !important; padding: 0 !important; }
      * { -webkit-print-color-adjust: exact; print-color-adjust: exact;
        animation-duration: 0s !important; animation-delay: 0s !important; transition: none !important; }
      .dk-print-flow { position: static !important; transform: none !important; width: auto !important;
        height: auto !important; overflow: visible !important; display: block !important; translate: none !important; }
      .dk-print-slide { position: relative !important; inset: auto !important; left: auto !important; top: auto !important;
        transform: none !important; translate: none !important; width: 100vw !important; height: 100vh !important;
        min-height: 0 !important; opacity: 1 !important; visibility: visible !important; overflow: hidden !important;
        flex: none !important; break-after: page; page-break-after: always; break-inside: avoid; }
      .dk-print-slide:last-of-type { break-after: auto; page-break-after: auto; }
      .dk-print-hide { display: none !important; }
    }
  `;

  // ── State ────────────────────────────────────────────────────────────────
  let adapter = null;
  let slides = [];
  let cur = -1;
  let ui = null;
  const widgets = [];         // { target, hooks, slide }
  const plan = { start: [], mins: [], total: 0, has: false };
  let timer = { started: null, pausedAt: null, offset: 0 };
  let pending = null;         // { i, until, next } — retried jump (templates lock nav while animating)
  let synthetic = false;      // true while the generic adapter dispatches its own arrow keys
  let presenterWin = null;    // main window: the presenter popup
  let blank = null;           // 'black' | 'white' | null
  let pinned = null;          // mirror: the slide the presenter told us to show

  // ── Deck discovery ───────────────────────────────────────────────────────
  const topLevelSlides = () =>
    Array.from(document.querySelectorAll('.slide')).filter((s) => !s.parentElement.closest('.slide'));
  const isActive = (s) =>
    s.classList.contains('active') || s.classList.contains('is-active') || s.hasAttribute('data-deck-active');

  function findAdapter() {
    const ds = document.querySelector('deck-stage');
    if (ds) {
      return {
        mode: 'deck-stage',
        slides: () => Array.from(ds.children).filter((e) => !/^(SCRIPT|STYLE|TEMPLATE)$/.test(e.tagName)),
        index: () => ds.index,
        go: (i) => ds.goTo(i),
        ownsHash: true,
        ownsKeys: true,
      };
    }
    const a = window.deckAdapter;
    if (a && typeof a.go === 'function' && typeof a.index === 'function') {
      return {
        mode: 'adapter',
        slides: a.slides ? () => Array.from(typeof a.slides === 'function' ? a.slides() : a.slides) : topLevelSlides,
        index: () => a.index(),
        go: (i) => a.go(i),
        ownsHash: !!a.ownsHash,
        nativeScroll: !!a.nativeScroll,
      };
    }
    // Last resort: step with synthetic arrow keys, which every template handles.
    let guess = 0;
    return {
      mode: 'generic',
      slides: topLevelSlides,
      index: () => {
        const k = slides.findIndex(isActive);
        return k < 0 ? guess : k;
      },
      go: (i) => {
        const from = adapter.index();
        const key = i > from ? 'ArrowRight' : 'ArrowLeft';
        synthetic = true;
        try {
          for (let n = Math.abs(i - from); n > 0; n--) {
            document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
          }
        } finally { synthetic = false; }
        guess = i;
      },
    };
  }

  /** Jump to slide i. Some templates ignore input during their transition,
   *  so the loop retries until the deck lands there (or 1.5 s pass). */
  function goto(i, steps = 0) {
    if (!slides.length) return;
    i = Math.max(0, Math.min(slides.length - 1, i));
    if (ROLE === 'presenter') { send(window.opener, { dk: 'go', index: i }); return; }
    // Steps revealed on arrival: none for a jump, all when stepping back into a slide.
    lastKitGo = { i, at: performance.now() };
    const back = { slide: stepState.slide, n: stepState.n };
    setSteps(i, steps);
    const now = performance.now();
    pending = { i, until: now + 1500, next: now + 150, back: pending ? pending.back : back };
    adapter.go(i);
  }
  /** Relative move that stacks while a jump is still landing. */
  const step = (d) => goto((pending ? pending.i : cur) + d);

  // ── Slide metadata ───────────────────────────────────────────────────────
  function titleOf(s, i) {
    const t = s.getAttribute('data-title') || s.getAttribute('data-label');
    if (t) return t;
    // Headings first, in rank order; then title-ish classes; then any text block.
    let h = null;
    for (const sel of ['h1', 'h2', 'h3', 'h4', '[class*="title"]', '[class*="headline"]', 'blockquote', 'p']) {
      h = Array.from(s.querySelectorAll(sel)).find((e) => !e.closest('aside.notes') && e.textContent.trim());
      if (h) break;
    }
    let txt = '';
    if (h) {
      // <br> has no text; turn it into a space so "two<br>lines" reads right.
      const c = sourceClone(h);
      c.querySelectorAll('br').forEach((b) => b.replaceWith(' '));
      txt = c.textContent.replace(/\s+/g, ' ').trim();
    }
    return txt ? txt.slice(0, 70) : 'Slide ' + (i + 1);
  }
  const secOf = (s) => s.getAttribute('data-sec') || s.getAttribute('data-section') || '';

  let jsonNotes = null;
  function notesOf(s, i) {
    const aside = s.querySelector('aside.notes');
    if (aside) return aside.innerHTML.trim();
    if (s.hasAttribute('data-notes')) return esc(s.getAttribute('data-notes'));
    if (jsonNotes === null) {
      jsonNotes = [];
      const tag = document.getElementById('speaker-notes');
      if (tag) try { const p = JSON.parse(tag.textContent || '[]'); if (Array.isArray(p)) jsonNotes = p; } catch (e) {}
    }
    return jsonNotes[i] ? esc(jsonNotes[i]) : '';
  }

  function buildPlan() {
    let acc = 0;
    plan.start = []; plan.mins = [];
    slides.forEach((s, i) => {
      const m = parseFloat(s.getAttribute('data-min') || s.getAttribute('data-minutes'));
      plan.mins[i] = isFinite(m) && m > 0 ? m : 0;
      plan.start[i] = acc;
      acc += plan.mins[i];
    });
    plan.total = acc;
    plan.has = acc > 0;
  }

  // ── UI ───────────────────────────────────────────────────────────────────
  function buildUI() {
    const style = el('style');
    style.id = 'deck-kit-style';
    style.textContent = css + (adapter.mode === 'deck-stage' ? '' : genericPrintCss) + (ROLE === 'presenter' ? presenterPageCss : '');
    document.head.appendChild(style);

    const host = el('div');
    host.id = 'deck-kit';
    host.setAttribute('data-noncommentable', '');
    host.classList.add('export-hidden');
    const root = host.attachShadow({ mode: 'open' });
    root.appendChild(el('style', '', uiCss));
    const progress = el('div', 'dk-progress');
    const notes = el('div', 'dk-panel dk-notes', '<div class="dk-hd"></div><div class="dk-tx"></div><div class="dk-next"></div>');
    const menu = el('div', 'dk-panel dk-menu', `<h2>${esc(L.menu)}</h2><div class="dk-items"></div>`);
    const tmr = el('div', 'dk-panel dk-timer');
    const help = el('div', 'dk-panel dk-dialog dk-help',
      `<h3>${esc(L.help)}</h3>${L.keys.filter(([k]) => !(PUBLISHED && /^[EKX]$/.test(k))).map(([k, d]) => `<div class="dk-row"><b><kbd>${esc(k)}</kbd></b><span>${esc(d)}</span></div>`).join('')}`);
    const toast = el('div', 'dk-panel dk-toast');
    const blankEl = el('div', 'dk-blank');
    const edbar = el('div', 'dk-panel dk-edbar', `<span>${L.editBar}</span><span class="dk-edn"></span>`);
    const chk = (k, label, on) => `<label><input type="checkbox" data-f="${k}"${on ? ' checked' : ''}>${esc(label)}</label>`;
    const share = el('div', 'dk-panel dk-dialog dk-share', `
      <h3>${esc(L.share)}</h3>
      <div class="dk-row"><b>${esc(L.printNow)}</b><div class="dk-opts"><button type="button" data-a="print">${esc(L.print)}</button></div></div>
      <div class="dk-row"><b>${esc(L.exportFiles)}</b><div>
        <div class="dk-opts">${chk('pdf', L.fPdf)}${chk('compact', L.fCompact)}${chk('png', L.fPng)}${chk('handout', L.fHandout)}${chk('pptx', L.fPptx, true)}${chk('pptx-editable', L.fPptxEd)}</div>
        <div class="dk-opts" style="margin-top:10px"><button type="button" data-a="export">${esc(L.copyCmd)}</button></div><code data-c="export"></code></div></div>
      <div class="dk-row"><b>${esc(L.deploy)}</b><div>
        <div class="dk-opts">${['vercel', 'cloudflare', 'github'].map((t, i) => `<label><input type="radio" name="dk-to" value="${t}"${i ? '' : ' checked'}>${t === 'vercel' ? 'Vercel' : t === 'cloudflare' ? 'Cloudflare' : 'GitHub'}</label>`).join('')}</div>
        <div class="dk-opts" style="margin-top:8px"><label><input type="checkbox" data-o="preview">${esc(L.previewOpt)}</label>
          <label><input type="checkbox" data-o="notes">${esc(L.keepNotes)}</label>
          <label>${esc(L.nameLbl)} <input type="text" data-o="name" spellcheck="false"></label></div>
        <div class="dk-opts" style="margin-top:10px"><button type="button" data-a="deploy">${esc(L.copyCmd)}</button></div><code data-c="deploy"></code></div></div>
      <div class="dk-note">${esc(L.shareNote)}</div>`);
    const hov = el('div', 'dk-hov');
    const selbox = el('div', 'dk-sel', '<span class="dk-tag"></span>' +
      ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'].map((d) => `<i class="dk-h" data-dir="${d}"></i>`).join(''));
    const HPOS = { nw: [0, 0, 'nwse'], n: [50, 0, 'ns'], ne: [100, 0, 'nesw'], e: [100, 50, 'ew'], se: [100, 100, 'nwse'], s: [50, 100, 'ns'], sw: [0, 100, 'nesw'], w: [0, 50, 'ew'] };
    selbox.querySelectorAll('.dk-h').forEach((h) => { const [x, y, c] = HPOS[h.dataset.dir]; h.style.left = x + '%'; h.style.top = y + '%'; h.style.cursor = c + '-resize'; });
    const ink = el('canvas', 'dk-ink');
    const laser = el('div', 'dk-laser');
    const zoomLayer = el('div', 'dk-zl');
    const zoomBox = el('div', 'dk-zb');
    zoomLayer.appendChild(zoomBox);
    if (plan.has && ROLE === 'main') root.appendChild(progress);
    root.append(ink, zoomLayer, hov, selbox, notes, menu, tmr, help, toast, blankEl, laser, edbar, share);
    document.body.appendChild(host);

    tmr.addEventListener('click', () => timerCmd('pause'));
    // Keep clicks, wheel and touches on our panels away from the template's
    // own handlers (several decks navigate on any click or wheel).
    ['click', 'mousedown', 'pointerdown', 'touchstart', 'touchend', 'wheel'].forEach((t) =>
      host.addEventListener(t, (e) => e.stopPropagation(), { passive: true }));
    ui = { host, root, progress, notes, menu, tmr, help, toast, blankEl, ink, laser, edbar, hov, selbox, share, zoomLayer, zoomBox };
    setupShare();
    buildMenu();
  }

  function buildMenu() {
    const box = ui.menu.querySelector('.dk-items');
    box.textContent = '';
    let last = null;
    slides.forEach((s, i) => {
      const sec = secOf(s);
      if (sec && sec !== last) { box.appendChild(el('div', 'dk-sec', esc(sec))); last = sec; }
      const lead = plan.has ? `${L.min} ${Math.round(plan.start[i])}` : String(i + 1).padStart(2, '0');
      const b = el('button', '', `<span>${esc(lead)}</span><span>${plan.has ? (i + 1) + '. ' : ''}${esc(titleOf(s, i))}</span>`);
      b.type = 'button';
      b.addEventListener('click', () => { show('menu', false); goto(i); });
      box.appendChild(b);
    });
  }

  function show(name, on) {
    const p = ui[name];
    const want = on === undefined ? !p.classList.contains('dk-show') : on;
    p.classList.toggle('dk-show', want);
    if (name === 'menu' && want) {
      const b = ui.menu.querySelectorAll('button')[cur];
      if (b) b.focus({ preventScroll: false });
    }
  }

  let toastTimer = null;
  function flash(msg, ms = 2600) {
    ui.toast.textContent = msg;
    show('toast', true);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => show('toast', false), ms);
  }

  function notesHtml(i) {
    const n = slides[i] ? notesOf(slides[i], i) : '';
    return n || `<em>${esc(L.noNotes)}</em>`;
  }
  function metaText(i) {
    return plan.has ? fmt(L.planned, { m: plan.mins[i] }) + ' · ' + fmt(L.startsAt, { s: Math.round(plan.start[i]) }) : '';
  }
  function renderNotes() {
    const s = slides[cur];
    if (!s) return;
    const meta = metaText(cur);
    ui.notes.querySelector('.dk-hd').innerHTML =
      `<span>${cur + 1}/${slides.length} · ${esc(titleOf(s, cur))}</span>${meta ? `<small>${esc(meta)}</small>` : ''}<small class="dk-el"></small>`;
    ui.notes.querySelector('.dk-tx').innerHTML = notesHtml(cur);
    const nx = slides[cur + 1];
    ui.notes.querySelector('.dk-next').textContent = nx ? `${L.next}: ${titleOf(nx, cur + 1)}` : L.end;
  }

  // ── Timer ────────────────────────────────────────────────────────────────
  const elapsed = () => {
    if (!timer.started) return 0;
    const now = timer.pausedAt || Date.now();
    return (now - timer.started - timer.offset) / 1000;
  };
  function startTimer() { if (!timer.started) timer.started = Date.now(); }
  function timerCmd(cmd) {
    if (ROLE === 'presenter') { send(window.opener, { dk: 'timer', cmd }); return; }
    if (cmd === 'reset') { timer = { started: null, pausedAt: null, offset: 0 }; }
    else if (cmd === 'pause') {
      if (!timer.started) startTimer();
      else if (timer.pausedAt) { timer.offset += Date.now() - timer.pausedAt; timer.pausedAt = null; }
      else timer.pausedAt = Date.now();
    }
    renderTimer();
    broadcast();
  }
  function timerStatus(e) {
    if (timer.pausedAt) return `<span class="dk-paused">${esc(L.paused)}</span>`;
    if (!plan.has || !timer.started || cur < 0) return '';
    const lo = plan.start[cur] * 60, hi = (plan.start[cur] + plan.mins[cur]) * 60;
    if (e < lo) return `<span class="dk-ahead">${esc(L.ahead)} ${mmss(lo - e)}</span>`;
    if (e > hi) return `<span class="dk-behind">${esc(L.behind)} ${mmss(e - hi)}</span>`;
    return `<span class="dk-ontime">${esc(L.onTime)}</span>`;
  }
  let lastTimerText = '';
  function renderTimer() {
    const e = elapsed();
    const total = plan.has ? ` / ${mmss(plan.total * 60)}` : '';
    const html = `<b>${mmss(e)}</b><span>${total}</span>${timerStatus(e)}`;
    if (html !== lastTimerText) {
      lastTimerText = html;
      ui.tmr.innerHTML = html;
      const tag = ui.notes.querySelector('.dk-el');
      if (tag) tag.textContent = timer.started ? `${L.elapsed} ${mmss(e)}${total}` : '';
      if (pv) { pv.clock.textContent = mmss(e) + total; pv.status.innerHTML = timerStatus(e); }
    }
  }

  // ── Widgets ──────────────────────────────────────────────────────────────
  function resolveSlide(target) {
    let node = target;
    if (typeof target === 'string') node = document.getElementById(target.replace(/^#/, ''));
    if (!node) return null;
    return slides.find((s) => s === node || s.contains(node)) || null;
  }
  const ctxFor = (w) => ({ slide: w.slide, index: slides.indexOf(w.slide), kit: api });
  function call(w, hook, ...args) {
    const f = w.hooks[hook];
    if (!f) return;
    try { f(...args, ctxFor(w)); } catch (e) { console.error('[deck-kit] widget ' + hook + ' failed:', e); }
  }
  const widgetsOn = (s) => widgets.filter((w) => w.slide === s);

  /** Size a canvas for its current CSS box (and the stage scale) and return
   *  a 2D context whose units are CSS pixels. */
  function fitCanvas(cv) {
    const w = cv.offsetWidth, h = cv.offsetHeight;
    const r = cv.getBoundingClientRect();
    const scale = w ? r.width / w : 1;
    const k = Math.min(4, Math.max(1, (window.devicePixelRatio || 1) * (scale || 1)));
    const bw = Math.max(1, Math.round(w * k)), bh = Math.max(1, Math.round(h * k));
    if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
    const ctx = cv.getContext('2d');
    ctx.setTransform(k, 0, 0, k, 0, 0);
    return { ctx, w, h };
  }

  // ── Widget helpers: palette, bind, plot ──────────────────────────────────
  const FALLBACK = ['#2563EB', '#E8770E', '#0F9D6B', '#C2255C', '#7C3AED', '#0891B2'];
  let probe = null;
  /** Normalise any CSS colour to rgb(a) via the browser; null if not a colour. */
  function toRgb(v) {
    if (!v || /^(transparent|inherit|initial|unset|none)$/i.test(v.trim())) return null;
    if (!CSS.supports('color', v.trim())) return null;
    if (!probe) { probe = el('span'); probe.style.display = 'none'; document.body.appendChild(probe); }
    probe.style.color = ''; probe.style.color = v.trim();
    const c = getComputedStyle(probe).color;
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const [r, g, b, a = 1] = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    return a < 0.5 ? null : { r, g, b, css: `rgb(${r}, ${g}, ${b})` };
  }
  const lum = ({ r, g, b }) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const dist = (a, b) => Math.abs(a.r - b.r) + Math.abs(a.g - b.g) + Math.abs(a.b - b.b);

  /** The template's colours as seen from `node`: { ink, bg, mute, accents[] }.
   *  Accents come from the page's colour custom properties, in declaration
   *  order, keeping only ones that stand out from the background. */
  function palette(node) {
    const start = typeof node === 'string' ? document.getElementById(node) : node;
    const at = start || slides[cur] || document.body;
    let bg = null;
    for (let n = at; n && n.nodeType === 1 && !bg; n = n.parentElement) bg = toRgb(getComputedStyle(n).backgroundColor);
    bg = bg || { r: 255, g: 255, b: 255, css: 'rgb(255, 255, 255)' };
    const ink = toRgb(getComputedStyle(at).color) || { r: 17, g: 17, b: 17, css: 'rgb(17, 17, 17)' };
    const seen = [bg, ink];
    const accents = [];
    const scopes = [document.documentElement, slides[cur], at].filter(Boolean);
    for (const sc of scopes) {
      const cs = getComputedStyle(sc);
      for (let k = 0; k < cs.length; k++) {
        const name = cs[k];
        if (!name.startsWith('--')) continue;
        const c = toRgb(cs.getPropertyValue(name));
        if (!c || contrast(c, bg) < 1.6 || seen.some((s) => dist(s, c) < 40)) continue;
        seen.push(c);
        accents.push(c.css);
      }
    }
    FALLBACK.forEach((f) => { if (accents.length < 4) { const c = toRgb(f); if (contrast(c, bg) >= 1.6) accents.push(c.css); } });
    const mute = `rgba(${ink.r}, ${ink.g}, ${ink.b}, .45)`;
    return { ink: ink.css, bg: bg.css, mute, grid: `rgba(${ink.r}, ${ink.g}, ${ink.b}, .12)`, accents, dark: lum(bg) < 0.25 };
  }

  /** Wire inputs to a callback. `inputs`: ids, elements, a selector or a
   *  container. Calls fn(values) now and on every change; values are keyed by
   *  id (or name) and numeric for range/number inputs. Matching <output for=id>
   *  or an <output> in the same label shows the live value. */
  function bind(inputs, fn) {
    let list = [];
    const add = (x) => {
      if (!x) return;
      if (typeof x === 'string') {
        const byId = document.getElementById(x.replace(/^#/, ''));
        if (byId) return add(byId);
        return document.querySelectorAll(x).forEach(add);
      }
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(x.tagName)) list.push(x);
      else x.querySelectorAll('input, select, textarea').forEach((e) => list.push(e));
    };
    (Array.isArray(inputs) ? inputs : [inputs]).forEach(add);
    list = [...new Set(list)];
    const outFor = (inp) => (inp.id && document.querySelector(`output[for~="${inp.id}"]`)) ||
      (inp.closest('label') && inp.closest('label').querySelector('output'));
    const decimals = (inp) => inp.dataset.d !== undefined ? +inp.dataset.d : ((inp.step || '1').split('.')[1] || '').length;
    const values = () => {
      const v = {};
      list.forEach((inp) => {
        const key = inp.id || inp.name;
        v[key] = inp.type === 'checkbox' ? inp.checked : /^(range|number)$/.test(inp.type) ? +inp.value : inp.value;
        const o = outFor(inp);
        if (o && /^(range|number)$/.test(inp.type)) o.textContent = (+inp.value).toFixed(decimals(inp));
      });
      return v;
    };
    const run = () => fn(values());
    list.forEach((inp) => inp.addEventListener('input', run));
    run();
    return values;
  }

  /** Chart helper on a canvas, in data coordinates.
   *    const p = DeckKit.plot(cv, { x: [-5, 5], y: [0, 0.6] });
   *    p.clear().axes({ xlabel: 'x' }).fn(x => …, { fill: true }).points(pts)
   *  Every call to clear() refits the canvas, so it is safe in draw()/tick(). */
  function plot(cv, opts = {}) {
    const o = Object.assign({ x: [0, 1], y: [0, 1], pad: [18, 18, 42, 54] }, opts);
    const pad = Array.isArray(o.pad) ? o.pad : [o.pad, o.pad, o.pad, o.pad]; // top right bottom left
    let ctx, W, H, pal;
    const p = {
      get ctx() { return ctx; }, get w() { return W; }, get h() { return H; }, get palette() { return pal; },
      X: (x) => pad[3] + (x - o.x[0]) / (o.x[1] - o.x[0]) * (W - pad[1] - pad[3]),
      Y: (y) => H - pad[2] - (y - o.y[0]) / (o.y[1] - o.y[0]) * (H - pad[0] - pad[2]),
      range(x, y) { if (x) o.x = x; if (y) o.y = y; return p; },
      color(i) { return typeof i === 'string' ? i : pal.accents[(i || 0) % pal.accents.length]; },
      clear() {
        ({ ctx, w: W, h: H } = fitCanvas(cv));
        pal = palette(cv);
        ctx.clearRect(0, 0, W, H);
        const font = getComputedStyle(cv).fontFamily || 'sans-serif';
        ctx.font = `${o.fontSize || 16}px ${font}`;
        return p;
      },
      axes({ xlabel = '', ylabel = '', ticks = 5, grid = true } = {}) {
        ctx.save();
        ctx.lineWidth = 1;
        ctx.fillStyle = pal.mute;
        ctx.textBaseline = 'top';
        const nice = (a, b) => { const s = 10 ** Math.floor(Math.log10((b - a) / ticks)); const m = (b - a) / ticks / s; return (m > 5 ? 10 : m > 2 ? 5 : m > 1 ? 2 : 1) * s; };
        const fmtN = (v, s) => (Math.abs(v) < 1e-9 ? '0' : v.toFixed(Math.max(0, -Math.floor(Math.log10(s)))));
        const sx = nice(o.x[0], o.x[1]), sy = nice(o.y[0], o.y[1]);
        ctx.textAlign = 'center';
        for (let v = Math.ceil(o.x[0] / sx) * sx; v <= o.x[1] + 1e-9; v += sx) {
          if (grid) { ctx.strokeStyle = pal.grid; ctx.beginPath(); ctx.moveTo(p.X(v), pad[0]); ctx.lineTo(p.X(v), H - pad[2]); ctx.stroke(); }
          ctx.fillText(fmtN(v, sx), p.X(v), H - pad[2] + 6);
        }
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let v = Math.ceil(o.y[0] / sy) * sy; v <= o.y[1] + 1e-9; v += sy) {
          if (grid) { ctx.strokeStyle = pal.grid; ctx.beginPath(); ctx.moveTo(pad[3], p.Y(v)); ctx.lineTo(W - pad[1], p.Y(v)); ctx.stroke(); }
          ctx.fillText(fmtN(v, sy), pad[3] - 8, p.Y(v));
        }
        ctx.strokeStyle = pal.mute; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(pad[3], pad[0]); ctx.lineTo(pad[3], H - pad[2]); ctx.lineTo(W - pad[1], H - pad[2]); ctx.stroke();
        if (xlabel) { ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(xlabel, W - pad[1], H - pad[2] - 6); }
        if (ylabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(ylabel, pad[3] + 8, pad[0]); }
        ctx.restore();
        return p;
      },
      /** y = f(x) across the x range. { color, width, dash, fill } */
      fn(f, s = {}) {
        const pts = [];
        const n = s.samples || Math.max(60, Math.round(W / 3));
        for (let k = 0; k <= n; k++) { const x = o.x[0] + (o.x[1] - o.x[0]) * k / n; pts.push([x, f(x)]); }
        return p.line(pts, s);
      },
      /** Polyline through [[x, y], …]. */
      line(pts, s = {}) {
        const col = p.color(s.color);
        ctx.save();
        ctx.beginPath();
        pts.forEach(([x, y], k) => (k ? ctx.lineTo(p.X(x), p.Y(y)) : ctx.moveTo(p.X(x), p.Y(y))));
        if (s.fill) {
          ctx.save();
          ctx.lineTo(p.X(pts[pts.length - 1][0]), p.Y(Math.max(o.y[0], 0)));
          ctx.lineTo(p.X(pts[0][0]), p.Y(Math.max(o.y[0], 0)));
          ctx.closePath();
          ctx.globalAlpha = typeof s.fill === 'number' ? s.fill : 0.22;
          ctx.fillStyle = col; ctx.fill();
          ctx.restore();
          ctx.beginPath();
          pts.forEach(([x, y], k) => (k ? ctx.lineTo(p.X(x), p.Y(y)) : ctx.moveTo(p.X(x), p.Y(y))));
        }
        ctx.strokeStyle = col; ctx.lineWidth = s.width || 3; ctx.lineJoin = 'round';
        if (s.dash) ctx.setLineDash(Array.isArray(s.dash) ? s.dash : [8, 6]);
        ctx.globalAlpha = s.alpha || 1;
        ctx.stroke();
        ctx.restore();
        return p;
      },
      /** Dots at [[x, y], …]. { color, r, shape: 'circle' | 'square' | 'x' } */
      points(pts, s = {}) {
        const col = p.color(s.color), r = s.r || 5;
        ctx.save();
        ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.globalAlpha = s.alpha || 1;
        pts.forEach(([x, y]) => {
          const X = p.X(x), Y = p.Y(y);
          ctx.beginPath();
          if (s.shape === 'square') ctx.rect(X - r, Y - r, 2 * r, 2 * r);
          else if (s.shape === 'x') { ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X + r, Y - r); ctx.lineTo(X - r, Y + r); ctx.stroke(); return; }
          else ctx.arc(X, Y, r, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
        return p;
      },
      /** Normal density N(mu, sigma²) as a curve. */
      gauss(mu, sigma, s = {}) {
        return p.fn((x) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)), s);
      },
      vline(x, s = {}) { return p.line([[x, o.y[0]], [x, o.y[1]]], Object.assign({ width: 1.5, dash: true, color: pal.mute }, s)); },
      hline(y, s = {}) { return p.line([[o.x[0], y], [o.x[1], y]], Object.assign({ width: 1.5, dash: true, color: pal.mute }, s)); },
      /** 2×2 covariance ellipse at k standard deviations. */
      ellipse(cx, cy, cov, s = {}) {
        const [[a, b], [, d]] = cov, k = s.k || 2;
        const tr = a + d, det = a * d - b * b, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const ang = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
        const pts = [];
        for (let t = 0; t <= 64; t++) {
          const u = t / 64 * Math.PI * 2, ex = k * Math.sqrt(l1) * Math.cos(u), ey = k * Math.sqrt(Math.max(0, l2)) * Math.sin(u);
          pts.push([cx + ex * Math.cos(ang) - ey * Math.sin(ang), cy + ex * Math.sin(ang) + ey * Math.cos(ang)]);
        }
        return p.line(pts, s);
      },
      text(x, y, str, s = {}) {
        ctx.save();
        ctx.fillStyle = s.color ? p.color(s.color) : pal.ink;
        ctx.textAlign = s.align || 'left'; ctx.textBaseline = s.baseline || 'middle';
        if (s.size) ctx.font = `${s.weight || 400} ${s.size}px ${getComputedStyle(cv).fontFamily}`;
        ctx.fillText(str, p.X(x) + (s.dx || 0), p.Y(y) + (s.dy || 0));
        ctx.restore();
        return p;
      },
      /** Coloured legend in the top-right corner: [['label', colorOrIndex], …] */
      legend(items) {
        ctx.save();
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        items.forEach(([label, c], k) => {
          const y = pad[0] + 12 + k * 24;
          ctx.fillStyle = pal.ink; ctx.fillText(label, W - pad[1] - 24, y);
          ctx.fillStyle = p.color(c); ctx.fillRect(W - pad[1] - 16, y - 6, 16, 12);
        });
        ctx.restore();
        return p;
      },
    };
    return p;
  }

  // ── Countdown & reveal ───────────────────────────────────────────────────
  const parseDur = (v) => {
    const m = String(v || '').trim().match(/^(\d+):(\d{1,2})$/);
    return m ? +m[1] * 60 + +m[2] : (parseFloat(v) || 0);
  };
  function cdState(e) {
    if (!e._dk) e._dk = { left: parseDur(e.getAttribute('data-countdown')), running: false };
    return e._dk;
  }
  function cdRender(e) { e.textContent = mmss(cdState(e).left); e.classList.toggle('dk-done', cdState(e).left <= 0); }
  function cdTick(dt) {
    document.querySelectorAll('[data-countdown]').forEach((e) => {
      const st = cdState(e);
      if (!st.running) return;
      st.left = Math.max(0, st.left - dt);
      if (st.left <= 0) st.running = false;
      cdRender(e);
    });
  }
  function onDocClick(e) {
    const cd = e.target.closest && e.target.closest('[data-countdown]');
    if (cd) {
      const st = cdState(cd);
      if (e.detail === 2) { st.left = parseDur(cd.getAttribute('data-countdown')); st.running = false; }
      else if (st.left > 0) st.running = !st.running;
      cdRender(cd);
    }
    const rv = e.target.closest && e.target.closest('[data-reveal]');
    if (rv) {
      const id = rv.getAttribute('data-reveal');
      const a = document.getElementById(id);
      if (a) { setReveal(rv, !a.hidden); route({ dk: 'reveal', id, hidden: a.hidden }); }
    }
  }
  function setReveal(rv, hidden) {
    const a = document.getElementById(rv.getAttribute('data-reveal'));
    if (!a) return;
    a.hidden = hidden;
    rv.setAttribute('aria-expanded', String(!hidden));
    const showT = rv.getAttribute('data-reveal-text') || L.reveal;
    const hideT = rv.getAttribute('data-hide-text') || L.hide;
    const txt = rv.textContent.trim();
    if (txt === showT || txt === hideT || !txt) rv.textContent = hidden ? showT : hideT;
  }

  // ── Sync: audience screen ⇄ presenter previews ───────────────────────────
  // Slider / input values and revealed answers travel both ways, so moving a
  // slider on either screen moves it on the other. Controls are addressed by
  // slide index + position among that slide's inputs.
  const controlsOf = (s) => Array.from(s.querySelectorAll('input, select, textarea'));
  let applyingRemote = false;   // true while applying a value that came from the other window
  function route(msg) {
    if (applyingRemote) return;
    if (ROLE === 'main') send(presenterWin, msg);
    else if (ROLE === 'mirror') send(window.parent, msg);
  }
  function onControl(e) {
    if (applyingRemote) return;
    const el = e.target;
    const s = slides.find((x) => x.contains(el));
    if (!s) return;
    const n = controlsOf(s).indexOf(el);
    if (n >= 0) route({ dk: 'control', slide: slides.indexOf(s), n, value: el.value, checked: el.checked });
  }
  function applyControl(m) {
    const s = slides[m.slide];
    const el = s && controlsOf(s)[m.n];
    if (!el) return;
    const check = /^(checkbox|radio)$/.test(el.type);
    if (check ? el.checked === m.checked : el.value === m.value) return;
    applyingRemote = true;
    try {
      if (check) el.checked = m.checked; else el.value = m.value;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } finally { applyingRemote = false; }
  }
  function applyReveal(m) {
    const btn = document.querySelector(`[data-reveal="${CSS.escape(m.id)}"]`);
    const a = document.getElementById(m.id);
    if (btn && a && a.hidden !== m.hidden) setReveal(btn, m.hidden);
  }
  function slideState(i) {
    const s = slides[i];
    if (!s) return null;
    const reveals = Array.from(s.querySelectorAll('[data-reveal]')).map((b) => {
      const a = document.getElementById(b.getAttribute('data-reveal'));
      return a ? { id: b.getAttribute('data-reveal'), hidden: a.hidden } : null;
    }).filter(Boolean);
    const acts = actorsOn(s).map((a) => ({ kind: a.kind, n: actorsOn(s, a.kind).indexOf(a), v: a.state() }));
    return { dk: 'sync', slide: i, controls: controlsOf(s).map((el) => ({ value: el.value, checked: el.checked })), reveals, acts };
  }
  function applySync(m) {
    m.controls.forEach((c, n) => applyControl({ slide: m.slide, n, value: c.value, checked: c.checked }));
    m.reveals.forEach(applyReveal);
    (m.acts || []).forEach((a) => applyAct(Object.assign({ slide: m.slide }, a)));
  }
  /** Main window: send the current and next slide's control state to the presenter. */
  function sendSync() {
    if (ROLE !== 'main' || !presenterWin || presenterWin.closed) return;
    [cur, cur + 1].forEach((i) => { const st = slideState(i); if (st) send(presenterWin, st); });
  }

  // ── Classroom tools: blank screen, laser, ink ────────────────────────────
  function setBlank(kind) {
    blank = blank === kind ? null : kind;
    ui.blankEl.style.background = blank === 'white' ? '#fff' : '#000';
    show('blankEl', !!blank);
  }

  let laserOn = false;
  function setLaser(on) {
    laserOn = on === undefined ? !laserOn : on;
    document.documentElement.classList.toggle('dk-laser', laserOn);
    if (!laserOn) show('laser', false);
    else flash(L.laserOn, 1800);
  }
  function onPointerMove(e) {
    if (!laserOn) return;
    ui.laser.style.left = e.clientX + 'px';
    ui.laser.style.top = e.clientY + 'px';
    show('laser', true);
  }

  // Strokes are stored per slide in viewport fractions so they survive resizes.
  const ink = { on: false, strokes: {}, live: null };
  function setDraw(on) {
    ink.on = on === undefined ? !ink.on : on;
    ui.ink.classList.toggle('dk-drawing', ink.on);
    if (ink.on) flash(L.drawOn, 2200);
    drawInk();
  }
  function drawInk() {
    const { ctx, w, h } = fitCanvas(ui.ink);
    ctx.clearRect(0, 0, w, h);
    const list = (ink.strokes[cur] || []).concat(ink.live ? [ink.live] : []);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    list.forEach((s) => {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width;
      ctx.beginPath();
      s.pts.forEach(([x, y], k) => (k ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
      if (s.pts.length === 1) ctx.lineTo(s.pts[0][0] * w + 0.1, s.pts[0][1] * h);
      ctx.stroke();
    });
  }
  function setupInk() {
    const c = ui.ink;
    const pt = (e) => [e.clientX / innerWidth, e.clientY / innerHeight];
    c.addEventListener('pointerdown', (e) => {
      if (!ink.on) return;
      c.setPointerCapture(e.pointerId);
      ink.live = { color: '#ff2a55', width: Math.max(3, innerWidth / 320), pts: [pt(e)] };
      drawInk();
    });
    c.addEventListener('pointermove', (e) => { if (ink.live) { ink.live.pts.push(pt(e)); drawInk(); } });
    const end = () => {
      if (!ink.live) return;
      (ink.strokes[cur] = ink.strokes[cur] || []).push(ink.live);
      ink.live = null;
      drawInk();
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
  }

  // ── Presenter view ───────────────────────────────────────────────────────
  const baseUrl = () => location.href.replace(/[?#].*$/, '');
  function send(win, msg) { try { if (win && !win.closed) win.postMessage(msg, '*'); } catch (e) {} }
  function stateMsg() { return { dk: 'state', index: cur, total: slides.length, timer, blank, steps: stepInfo() }; }
  function broadcast() { if (ROLE === 'main' && presenterWin && !presenterWin.closed) send(presenterWin, stateMsg()); }

  function openPresenter() {
    if (presenterWin && !presenterWin.closed) { presenterWin.focus(); return; }
    presenterWin = window.open(`${baseUrl()}?dk=presenter#${cur + 1}`, 'deck-kit-presenter', 'width=1280,height=800');
    if (!presenterWin) { flash(L.popup, 4000); return; }
    show('notes', false);
  }

  // Presenter: Z picks a box over the current preview; the audience screen zooms there.
  const pvZoom = { on: false, picking: false, x0: 0, y0: 0, drag: false };
  function pvZoomKey() {
    if (pvZoom.on || pvZoom.picking) {
      pvZoom.picking = false;
      pv.zl.classList.remove('dk-show');
      if (pvZoom.on) {
        pvZoom.on = false;
        const m = { dk: 'zoom', slide: cur, off: true };
        send(window.opener, m); send(pv.fc.contentWindow, m);
      }
      return;
    }
    pvZoom.picking = true;
    pv.zl.classList.add('dk-show');
    flash(L.zoomHint, 2400);
  }
  function setupPvZoom() {
    const zl = pv.zl = pv.box.querySelector('.dk-pvzl'), box = zl.firstElementChild;
    const frac = (e) => { const r = zl.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
    zl.addEventListener('pointerdown', (e) => { e.preventDefault(); zl.setPointerCapture(e.pointerId); pvZoom.drag = true; [pvZoom.x0, pvZoom.y0] = frac(e); });
    zl.addEventListener('pointermove', (e) => {
      if (!pvZoom.drag) return;
      const [x, y] = frac(e);
      Object.assign(box.style, { display: 'block', left: Math.min(x, pvZoom.x0) * 100 + '%', top: Math.min(y, pvZoom.y0) * 100 + '%', width: Math.abs(x - pvZoom.x0) * 100 + '%', height: Math.abs(y - pvZoom.y0) * 100 + '%' });
    });
    const end = (e) => {
      if (!pvZoom.drag) return;
      const [x, y] = frac(e);   // measure before the layer hides (a hidden layer has no size)
      pvZoom.drag = false; pvZoom.picking = false;
      box.style.display = 'none';
      zl.classList.remove('dk-show');
      if (e.type !== 'pointerup' || !isFinite(x + y)) return;
      // The preview converts the box to slide coordinates and tells the audience screen.
      send(pv.fc.contentWindow, { dk: 'zoom-pick', x: Math.min(x, pvZoom.x0), y: Math.min(y, pvZoom.y0), w: Math.abs(x - pvZoom.x0), h: Math.abs(y - pvZoom.y0) });
    };
    zl.addEventListener('pointerup', end);
    zl.addEventListener('pointercancel', end);
  }

  let pv = null;
  let pvSteps = null;   // presenter: { n, total } steps shown on the audience screen
  function buildPresenter() {
    const root = ui.root;
    const box = el('div', 'dk-pv', `
      <div class="dk-top"><div class="dk-title"></div><span class="dk-msg"></span><span class="dk-status"></span><span class="dk-clock" title="pause"></span></div>
      <div class="dk-main"><div class="dk-lbl">${esc(L.current)}</div><div class="dk-frame dk-cur"><iframe title="current"></iframe><div class="dk-pvzl"><i></i></div></div></div>
      <div class="dk-side">
        <div><div class="dk-lbl">${esc(L.next)}</div><div class="dk-frame dk-nx"><iframe title="next"></iframe><div class="dk-endcard">${esc(L.end)}</div></div></div>
        <div class="dk-nrow"><div class="dk-lbl">${esc(L.notes)} <span class="dk-meta"></span></div>
          <span class="dk-nstat"></span>${PUBLISHED ? '' : `<button data-n="edit">${esc(L.edit)}</button><button data-n="save">${esc(L.saveFile)}</button>`}</div>
        <div class="dk-pnotes"><div class="dk-tx"></div></div>
      </div>
      <div class="dk-bar"><button data-a="prev">◀</button><button data-a="next">▶</button><button data-a="black">B</button>
        <button data-a="reset">⟲ 0:00</button><span class="dk-pos"></span></div>`);
    root.appendChild(box);
    const [fc, fn] = box.querySelectorAll('iframe');
    const url = (i) => `${baseUrl()}?dk=mirror#${i + 1}`;
    fc.src = url(cur);
    fn.src = url(Math.min(slides.length - 1, cur + 1));
    pv = {
      box, fc, fn,
      title: box.querySelector('.dk-title'), clock: box.querySelector('.dk-clock'), status: box.querySelector('.dk-status'),
      msg: box.querySelector('.dk-msg'), notes: box.querySelector('.dk-pnotes .dk-tx'), meta: box.querySelector('.dk-meta'),
      pos: box.querySelector('.dk-pos'), nxFrame: box.querySelector('.dk-nx'),
      notesBox: box.querySelector('.dk-pnotes'), nstat: box.querySelector('.dk-nstat'), editBtn: box.querySelector('[data-n="edit"]'),
    };
    pv.clock.addEventListener('click', () => timerCmd('pause'));
    setupPvZoom();
    if (pv.editBtn) {
      pv.editBtn.addEventListener('click', () => setEditing(!editing));
      box.querySelector('[data-n="save"]').addEventListener('click', saveNotesToFile);
    }
    let typingTimer = null;
    pv.notes.addEventListener('input', () => {
      clearTimeout(typingTimer);
      typingTimer = setTimeout(() => {
        const html = pv.notes.textContent.trim() ? pv.notes.innerHTML.trim() : '';
        editNotes(cur, html);
        send(window.opener, { dk: 'notes', slide: cur, html });
      }, 250);
    });
    renderNoteStatus();
    box.querySelector('.dk-bar').addEventListener('click', (e) => {
      const a = e.target.closest('button') && e.target.closest('button').dataset.a;
      if (a === 'prev') advance(-1);
      if (a === 'next') advance(1);
      if (a === 'black') send(window.opener, { dk: 'key', key: 'b' });
      if (a === 'reset') timerCmd('reset');
    });
    const fit = () => box.querySelectorAll('.dk-frame').forEach((f) => {
      f.querySelector('iframe').style.transform = `scale(${f.clientWidth / 1600})`;
    });
    new ResizeObserver(fit).observe(box);
    fit();
    // Previews load asynchronously; re-send their slide once they're ready.
    const reloaded = () => setTimeout(() => { renderPresenter(); send(window.opener, { dk: 'want-sync' }); }, 300);
    fc.addEventListener('load', reloaded);
    fn.addEventListener('load', reloaded);
    pv.msg.textContent = window.opener ? L.waiting : L.lost;
    // Say hello until the audience window answers; notice if it goes away.
    let heard = false;
    pv.heard = () => { heard = true; };
    const ping = () => {
      if (!window.opener || window.opener.closed) { pv.msg.textContent = L.lost; return; }
      if (!heard) send(window.opener, { dk: 'hello' });
    };
    ping();
    setInterval(ping, 1500);
    renderPresenter();
  }
  function renderPresenter() {
    if (!pv || cur < 0) return;
    const s = slides[cur];
    pv.title.textContent = `${cur + 1}. ${titleOf(s, cur)}`;
    pv.notes.innerHTML = editing ? (slides[cur] ? notesOf(slides[cur], cur) : '') : notesHtml(cur);
    pv.meta.textContent = metaText(cur) ? '· ' + metaText(cur) : '';
    renderPreviews();
    lastTimerText = '';
    renderTimer();
  }
  /** Position, step counter and the two previews (notes untouched: they may be mid-edit). */
  function renderPreviews() {
    if (!pv || cur < 0) return;
    const si = pvSteps || { n: 0, total: 0 };
    pv.pos.textContent = `${cur + 1} / ${slides.length}` + (si.total ? ` · ${L.step} ${si.n}/${si.total}` : '');
    send(pv.fc.contentWindow, { dk: 'go', index: cur, steps: si.n });
    // "Next" previews the next step while the slide still has some.
    const more = si.n < si.total;
    const end = !more && cur + 1 >= slides.length;
    pv.nxFrame.classList.toggle('dk-end', end);
    if (!end) send(pv.fn.contentWindow, more ? { dk: 'go', index: cur, steps: si.n + 1 } : { dk: 'go', index: cur + 1, steps: 0 });
  }

  function onMessage(e) {
    const m = e.data;
    if (!m || typeof m !== 'object' || !m.dk) return;
    if (ROLE === 'main') {
      // A reloaded audience window re-adopts the presenter that says hello.
      if (m.dk === 'hello' && (!presenterWin || presenterWin.closed)) presenterWin = e.source;
      if (e.source !== presenterWin) return;
      if (m.dk === 'hello') { broadcast(); sendSync(); }
      if (m.dk === 'want-sync') sendSync();
      if (m.dk === 'go') goto(m.index);
      if (m.dk === 'nav') advance(m.d);
      if (m.dk === 'act') applyAct(m);
      if (m.dk === 'zoom') applyZoom(m);
      if (m.dk === 'timer') timerCmd(m.cmd);
      if (m.dk === 'key') handleKey(m.key, m.shift);
      if (m.dk === 'control') applyControl(m);
      if (m.dk === 'reveal') applyReveal(m);
      if (m.dk === 'notes') { editNotes(m.slide, m.html); if (m.slide === cur) renderNotes(); }
      if (m.dk === 'notes-saved') { notesEdits.clear(); clearDraft(); }
      if (m.dk === 'file-saved') lastWrite = Math.max(lastWrite, m.at || Date.now());
    } else if (ROLE === 'presenter') {
      const fromPreview = pv && (e.source === pv.fc.contentWindow || e.source === pv.fn.contentWindow);
      if (fromPreview) {
        // Interaction inside a preview: forward to the audience screen.
        if (m.dk === 'zoom') { pvZoom.on = !m.off; send(window.opener, m); return; }
        if (m.dk === 'control' || m.dk === 'reveal' || m.dk === 'act') {
          send(window.opener, m);
          // Both previews may show this slide (the next one a step further).
          send((e.source === pv.fc.contentWindow ? pv.fn : pv.fc).contentWindow, m);
        }
        if (m.dk === 'key' && !handleKey(m.key, m.shift)) navKey(m.key, m.shift);
        return;
      }
      if (e.source !== window.opener) return;
      if (m.dk === 'file-saved') lastWrite = Math.max(lastWrite, m.at || Date.now());
      if (m.dk === 'state') {
        pv.msg.textContent = '';
        pv.heard();
        timer = m.timer;
        const st = m.steps || { n: 0, total: 0 };
        if (m.index !== cur) { cur = m.index; pvSteps = st; pvZoom.on = false; renderPresenter(); }
        else if (!pvSteps || st.n !== pvSteps.n || st.total !== pvSteps.total) { pvSteps = st; renderPreviews(); }
        lastTimerText = '';
        renderTimer();
      }
      if (m.dk === 'zoom') {
        if (m.slide === cur) { pvZoom.on = !m.off; send(pv.fc.contentWindow, m); }
        else if (m.off) pvZoom.on = false;
      }
      if (m.dk === 'sync' || m.dk === 'control' || m.dk === 'reveal' || m.dk === 'act') {
        // The "next" preview may show this same slide (one step further).
        const nextShows = pvSteps && pvSteps.n < pvSteps.total ? cur : cur + 1;
        if (m.slide === cur || m.dk === 'reveal') send(pv.fc.contentWindow, m);
        if (m.slide === nextShows) send(pv.fn.contentWindow, m);
      }
    } else if (ROLE === 'mirror') {
      if (e.source !== window.parent) return;
      if (m.dk === 'go') { pinned = m.index; goto(m.index, m.steps || 0); }
      if (m.dk === 'sync') applySync(m);
      if (m.dk === 'control') applyControl(m);
      if (m.dk === 'reveal') applyReveal(m);
      if (m.dk === 'act') applyAct(m);
      if (m.dk === 'zoom') applyZoom(m);
      if (m.dk === 'zoom-pick') zoomPick(m);
    }
  }

  // ── Editing speaker notes (presenter view) ───────────────────────────────
  // Edits apply to the live deck at once and are kept as a draft in
  // localStorage. "Save" writes them into the deck's own HTML file through
  // the File System Access API (Chrome/Edge): the original source is read and
  // only the edited <aside class="notes"> are replaced, so nothing from the
  // running presentation leaks into the file. Elsewhere it downloads
  // notes.json for scripts/apply-notes.mjs.
  const notesEdits = new Map();      // slide index → notes HTML ('' = remove)
  const DRAFT = 'deck-kit-notes:' + location.pathname;
  let editing = false;
  let fileHandle = null;
  let lastWrite = 0;        // when this deck (either window) last wrote the file

  function setNotesDom(root, list, i, html) {
    const s = list[i];
    if (!s) return;
    let a = s.querySelector('aside.notes');
    if (!html) { if (a) a.remove(); return; }
    if (!a) {
      a = root.createElement('aside');
      a.className = 'notes';
      s.append(root.createTextNode('\n  '), a, root.createTextNode('\n'));
    }
    a.innerHTML = html;
  }
  function editNotes(i, html) {
    notesEdits.set(i, html);
    setNotesDom(document, slides, i, html);
    try { localStorage.setItem(DRAFT, JSON.stringify(Object.fromEntries(notesEdits))); } catch (e) {}
    renderNoteStatus();
  }
  function clearDraft() { try { localStorage.removeItem(DRAFT); } catch (e) {} renderNoteStatus(); }
  function loadDraft() {
    let d = null;
    try { d = JSON.parse(localStorage.getItem(DRAFT) || 'null'); } catch (e) {}
    if (!d) return 0;
    Object.entries(d).forEach(([i, html]) => { notesEdits.set(+i, html); setNotesDom(document, slides, +i, html); });
    return notesEdits.size;
  }
  function renderNoteStatus(msg) {
    if (!pv) return;
    pv.nstat.textContent = msg || (notesEdits.size ? fmt(L.unsaved, { n: notesEdits.size }) : '');
  }
  function setEditing(on) {
    editing = on;
    pv.notes.contentEditable = on ? 'true' : 'false';
    pv.notesBox.classList.toggle('dk-editing', on);
    pv.editBtn.textContent = on ? L.done : L.edit;
    renderPresenter();
    if (on) pv.notes.focus();
  }

  /** Find each slide's [start, end) in the HTML source text: the offset of
   *  its opening tag and of its closing tag. Mirrors findAdapter(): children
   *  of <deck-stage>, else top-level `.slide` elements. Comments and the raw
   *  text of script/style/textarea are skipped; unclosed tags are tolerated. */
  function slideRanges(src) {
    const VOID = /^(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)$/i;
    const tagRe = /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<![^>]*>|<(\/?)([A-Za-z][\w:.-]*)((?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*(\/?)>/g;
    const hasDeckStage = /<deck-stage[\s>]/i.test(src.replace(/<!--[\s\S]*?-->/g, ''));
    const stack = [], ranges = [];
    let m;
    while ((m = tagRe.exec(src))) {
      if (!m[2]) continue;
      const name = m[2].toLowerCase();
      if (m[1]) { // closing tag: pop to the matching opener
        for (let k = stack.length - 1; k >= 0; k--) {
          if (stack[k].name !== name) continue;
          const [open] = stack.splice(k);
          if (open.slide) ranges.push([open.start, m.index]);
          break;
        }
        continue;
      }
      const cls = (m[3].match(/\sclass\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'>]+))/i) || [])[1] || '';
      const parent = stack[stack.length - 1];
      const slide = hasDeckStage
        ? !!parent && parent.name === 'deck-stage' && !/^(script|style|template)$/.test(name)
        : /(^|\s|["'])slide(\s|["']|$)/.test(cls) && !stack.some((x) => x.slide);
      if (VOID.test(name) || m[4]) { if (slide) ranges.push([m.index, m.index + m[0].length, true]); continue; }
      stack.push({ name, start: m.index, slide });
      if (/^(script|style|textarea|title)$/.test(name)) {
        const close = src.toLowerCase().indexOf('</' + name, tagRe.lastIndex);
        tagRe.lastIndex = close < 0 ? src.length : close;
      }
    }
    return ranges.sort((x, y) => x[0] - y[0]);
  }

  /** Replace only the edited <aside class="notes"> in the source text; every
   *  other byte stays as it was. Returns null if the slides can't be matched. */
  function patchNotesText(src, edits, count) {
    const ranges = slideRanges(src);
    if (ranges.length !== count || ranges.some((r) => r[2])) return null;
    const asideRe = /<aside\b[^>]*\sclass\s*=\s*(?:"[^"]*\bnotes\b[^"]*"|'[^']*\bnotes\b[^']*'|notes\b)[^>]*>/i;
    let out = src;
    // Work from the last slide back so earlier offsets stay valid.
    [...edits.entries()].sort((x, y) => y[0] - x[0]).forEach(([i, html]) => {
      const [start, end] = ranges[i];
      const body = out.slice(start, end);
      const am = body.match(asideRe);
      if (am) {
        const open = start + am.index;
        const close = out.indexOf('</aside>', open);
        if (close < 0 || close > end) return;
        if (html) out = out.slice(0, open + am[0].length) + html + out.slice(close);
        else {
          // Drop the aside together with the whitespace line it sat on.
          const lineStart = out.lastIndexOf('\n', open - 1);
          const from = /^\s*$/.test(out.slice(lineStart + 1, open)) ? lineStart : open;
          out = out.slice(0, from) + out.slice(close + '</aside>'.length);
        }
      } else if (html) {
        const lineStart = out.lastIndexOf('\n', end - 1);
        const indent = (out.slice(lineStart + 1, end).match(/^[ \t]*$/) || [''])[0];
        const inner = indent + (indent.includes('\t') ? '\t' : '  ');
        const atLineStart = /^[ \t]*$/.test(out.slice(lineStart + 1, end));
        out = atLineStart
          ? out.slice(0, lineStart + 1) + `${inner}<aside class="notes">${html}</aside>\n` + out.slice(lineStart + 1)
          : out.slice(0, end) + `<aside class="notes">${html}</aside>` + out.slice(end);
      }
    });
    return out;
  }

  /** Put the edited notes into an HTML source string: a minimal text edit when
   *  the slides can be located in the source, otherwise a parse-and-serialise
   *  (which doesn't run scripts, but normalises formatting). */
  function patchNotes(src, edits, count) {
    const exact = patchNotesText(src, edits, count);
    if (exact != null) return exact;
    const doc = new DOMParser().parseFromString(src, 'text/html');
    const ds = doc.querySelector('deck-stage');
    const list = ds
      ? Array.from(ds.children).filter((e) => !/^(SCRIPT|STYLE|TEMPLATE)$/.test(e.tagName))
      : Array.from(doc.querySelectorAll('.slide')).filter((s) => !s.parentElement.closest('.slide'));
    if (list.length !== count) throw new Error(`the file has ${list.length} slides, the deck shows ${count}`);
    edits.forEach((html, i) => setNotesDom(doc, list, i, html));
    const dt = doc.doctype ? `<!DOCTYPE ${doc.doctype.name}>` : '<!doctype html>';
    return dt + '\n' + doc.documentElement.outerHTML + '\n';
  }

  const deckName = () => decodeURIComponent(location.pathname.split('/').pop() || 'index.html');

  /** Rewrite the deck's own HTML file: transform(sourceText) → new text.
   *  File System Access (Chrome/Edge); the deck file is picked once per
   *  window. Throws {fallback: true} where the API is missing. */
  async function writeDeck(transform, onPick) {
    if (!window.showOpenFilePicker) { const e = new Error('no file access'); e.fallback = true; throw e; }
    if (!fileHandle) {
      if (onPick) onPick();
      [fileHandle] = await window.showOpenFilePicker({ types: [{ description: 'HTML', accept: { 'text/html': ['.html', '.htm'] } }] });
      if (fileHandle.name !== deckName() && !confirm(fmt(L.otherFile, { a: fileHandle.name, b: deckName() }))) {
        fileHandle = null;
        const e = new Error('cancelled'); e.name = 'AbortError'; throw e;
      }
    }
    let perm = await fileHandle.queryPermission({ mode: 'readwrite' });
    if (perm !== 'granted') perm = await fileHandle.requestPermission({ mode: 'readwrite' });
    if (perm !== 'granted') throw new Error('permission denied');
    const file = await fileHandle.getFile();
    // Refuse to patch a file that changed after this page loaded it (e.g. the
    // agent edited it): element positions may no longer match.
    if (file.lastModified > Math.max(performance.timeOrigin, lastWrite) + 2000) {
      const e = new Error(L.stale); e.stale = true; throw e;
    }
    const out = transform(await file.text());
    const w = await fileHandle.createWritable();
    await w.write(out);
    await w.close();
    lastWrite = Math.max(Date.now(), (await fileHandle.getFile()).lastModified);
    // The other window (presenter ⇄ audience) must not mistake this save for an outside edit.
    send(ROLE === 'presenter' ? window.opener : presenterWin, { dk: 'file-saved', at: lastWrite });
    return fileHandle.name;
  }
  function download(name, data) {
    const a = el('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }

  async function saveNotesToFile() {
    if (!notesEdits.size) { renderNoteStatus(L.nothing); return; }
    try {
      const name = await writeDeck((src) => patchNotes(src, notesEdits, slides.length), () => renderNoteStatus(L.pickDeck));
      notesEdits.clear();
      clearDraft();
      send(window.opener, { dk: 'notes-saved' });
      renderNoteStatus(fmt(L.saved, { f: name }));
      return;
    } catch (e) {
      if (e.name === 'AbortError') { renderNoteStatus(); return; }
      if (e.stale) { renderNoteStatus(L.stale); return; }
      if (!e.fallback) { renderNoteStatus(fmt(L.saveFailed, { e: e.message })); if (!/slides/.test(e.message)) return; }
    }
    // No File System Access (Firefox, Safari): hand over a notes.json to merge.
    download('notes.json', { deck: deckName(), slides: slides.length, notes: Object.fromEntries(notesEdits) });
    renderNoteStatus(L.downloaded);
  }

  // ── Source mapping: a live element → its tag in the HTML source ─────────
  // Used to save visual edits and to report line numbers. The element is
  // found by its child-index path from the slide in a script-free parse of
  // the source (checked by tag name, with its original text as a tie-break),
  // then matched to the source tags of that slide in document order.
  const SRC_TAG_RE = /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<![^>]*>|<(\/?)([A-Za-z][\w:.-]*)((?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*(\/?)>/g;
  function tagsIn(src, start, end) {
    const VOID = /^(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)$/i;
    const re = new RegExp(SRC_TAG_RE.source, 'g');
    re.lastIndex = start;
    const out = [], stack = [];
    let m;
    while ((m = re.exec(src)) && m.index < end) {
      if (!m[2]) continue;
      const name = m[2].toLowerCase();
      if (m[1]) {
        for (let k = stack.length - 1; k >= 0; k--) {
          if (stack[k].name !== name) continue;
          const [open] = stack.splice(k);
          open.close = m.index;
          break;
        }
        continue;
      }
      const t = { name, start: m.index, openEnd: m.index + m[0].length, close: -1 };
      out.push(t);
      if (VOID.test(name) || m[4]) continue;
      stack.push(t);
      if (/^(script|style|textarea|title)$/.test(name)) {
        const c = src.toLowerCase().indexOf('</' + name, re.lastIndex);
        re.lastIndex = c < 0 ? src.length : c;
      }
    }
    return out;
  }
  const pathOf = (node, slide) => {
    const p = [];
    for (let e = node; e && e !== slide; e = e.parentElement) p.unshift(realChildren(e.parentElement).indexOf(e));
    return p;
  };
  const byPath = (slide, path) => path.reduce((e, k) => (e ? realChildren(e)[k] : null), slide);
  const normText = (t) => (t || '').replace(/\s+/g, ' ').trim();
  function sourceIndex(src, count = slides.length) {
    const ranges = slideRanges(src);
    if (ranges.length !== count || ranges.some((r) => r[2])) throw new Error(`the file has ${ranges.length} slides, the deck shows ${count}`);
    const doc = new DOMParser().parseFromString(src, 'text/html');
    const ds = doc.querySelector('deck-stage');
    const list = ds
      ? Array.from(ds.children).filter((e) => !/^(SCRIPT|STYLE|TEMPLATE)$/.test(e.tagName))
      : Array.from(doc.querySelectorAll('.slide')).filter((x) => !x.parentElement.closest('.slide'));
    return { src, ranges, list, maps: [] };
  }
  /** Source tag {start, openEnd, close} of the element at `path` on slide i. */
  function locateIn(ix, i, path, tag, text) {
    const ps = ix.list[i];
    if (!ps) return null;
    let pe = byPath(ps, path);
    if (pe && pe.tagName.toLowerCase() !== tag) pe = null;
    if (text != null && (!pe || normText(pe.textContent) !== normText(text))) {
      const same = Array.from(ps.querySelectorAll(tag)).filter((e) => normText(e.textContent) === normText(text));
      if (same.length === 1) pe = same[0];
    }
    if (!pe) return null;
    if (!ix.maps[i]) {
      const [start, end] = ix.ranges[i];
      const tags = tagsIn(ix.src, start, end);
      const map = new Map();
      let j = 0;
      for (const d of [ps, ...ps.querySelectorAll('*')]) {
        const name = d.tagName.toLowerCase();
        let k = j;
        while (k < tags.length && tags[k].name !== name && k - j < 60) k++;
        if (k < tags.length && tags[k].name === name) { map.set(d, tags[k]); j = k + 1; }
      }
      ix.maps[i] = map;
    }
    return ix.maps[i].get(pe) || null;
  }
  function setStyleProps(openTag, props) {
    const m = openTag.match(/\sstyle\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'>]+))/i);
    const before = m ? (m[2] ?? m[3] ?? m[4] ?? '') : '';
    const map = new Map();
    before.replace(/&quot;/g, '"').split(';').forEach((d) => { const k = d.indexOf(':'); if (k > 0) map.set(d.slice(0, k).trim().toLowerCase(), d.slice(k + 1).trim()); });
    Object.entries(props).forEach(([k, v]) => (v ? map.set(k, v) : map.delete(k)));
    const css = [...map].map(([k, v]) => `${k}: ${v}`).join('; ');
    const attr = css ? ` style="${css.replace(/"/g, '&quot;')}"` : '';
    if (m) return openTag.slice(0, m.index) + attr + openTag.slice(m.index + m[0].length);
    return attr ? openTag.replace(/\s*(\/?)>$/, `${attr}$1>`) : openTag;
  }
  /** Apply visual edits [{slide, path, tag, text, style: {prop: value}, html?}]
   *  to an HTML source string, touching only the edited tags. */
  function patchEdits(src, edits, count = slides.length) { return patchEditsReport(src, edits, count).out; }
  /** Like patchEdits, but skips elements that aren't in the source (made by a
   *  template's script at runtime) and reports them instead of failing. */
  function patchEditsReport(src, edits, count = slides.length) {
    const ix = sourceIndex(src, count);
    let found = edits.map((e) => ({ e, t: locateIn(ix, e.slide, e.path, e.tag, e.text) }));
    const skipped = found.filter((x) => !x.t).map((x) => x.e);
    found = found.filter((x) => x.t);
    // A rewritten element already contains its children's state.
    const rewritten = found.filter((x) => x.e.html != null && x.t.close >= 0);
    found = found.filter((x) => !rewritten.some((r) => r !== x && x.t.start > r.t.openEnd && x.t.start < r.t.close));
    let out = src;
    found.sort((a, b) => b.t.start - a.t.start).forEach(({ e, t }) => {
      if (e.html != null && t.close >= 0) out = out.slice(0, t.openEnd) + e.html + out.slice(t.close);
      if (e.style && Object.keys(e.style).length) out = out.slice(0, t.start) + setStyleProps(out.slice(t.start, t.openEnd), e.style) + out.slice(t.openEnd);
    });
    return { out, skipped };
  }
  async function readSource() {
    if (/^https?:/.test(location.protocol)) { try { return await (await fetch(location.href.replace(/#.*$/, ''), { cache: 'no-store' })).text(); } catch (e) {} }
    if (fileHandle) { try { return await (await fileHandle.getFile()).text(); } catch (e) {} }
    return null;
  }

  // ── Edit mode (E) ────────────────────────────────────────────────────────
  // Select, move (CSS translate), resize, type and hide elements on the
  // slides; E again writes only those elements back into the HTML file.
  const TRACK = ['translate', 'width', 'height', 'font-size', 'display'];
  const ed = { on: false, sel: null, touched: new Map(), undo: [], redo: [], typing: null };
  const EDRAFT = 'deck-kit-edits:' + location.pathname;
  const round = (v, d = 0) => +v.toFixed(d);
  const isMedia = (e) => /^(img|canvas|svg|video|picture|iframe|object|embed)$/i.test(e.tagName);
  const isTextual = (e) => {
    if (isMedia(e) || e.querySelector('canvas, img, video, iframe, .dk-stepper, .dk-chart-box, .dk-hs-dot, .dk-cmp-range, .dk-qr')) return false;
    if (Array.from(e.querySelectorAll('svg')).some((v) => !v.closest('.katex'))) return false;
    return !!normText(srcText(e));
  };
  const slideOf = (node) => slides.find((x) => x.contains(node)) || null;
  const editable = (node) => node && node.nodeType === 1 && slides[cur] && slides[cur] !== node && slides[cur].contains(node) && !node.closest('aside.notes') && !node.closest('[data-dk-gen]');

  // HTML is kept in source form (what the author wrote), never the rendered maths / highlighting.
  function snapshot(e) { const o = {}; TRACK.forEach((p) => { o[p] = e.style.getPropertyValue(p); }); o.html = sourceHtml(e); return o; }
  function restoreSnap(e, snap) {
    TRACK.forEach((p) => (snap[p] ? e.style.setProperty(p, snap[p]) : e.style.removeProperty(p)));
    if (!e.getAttribute('style')) e.removeAttribute('style');
    if (sourceHtml(e) !== snap.html) { e.innerHTML = snap.html; enhance(e); applySteps(slides[cur], stepState.n); }
  }
  function touch(e) {
    if (ed.touched.has(e)) return ed.touched.get(e);
    const s = slideOf(e);
    const info = { slide: slides.indexOf(s), path: pathOf(e, s), tag: e.tagName.toLowerCase(), text: srcText(e), orig: snapshot(e), htmlEdited: false };
    ed.touched.set(e, info);
    return info;
  }
  function pendingEdits() {
    const list = [];
    ed.touched.forEach((info, e) => {
      const now = snapshot(e);
      const style = {};
      TRACK.forEach((p) => { if (now[p] !== info.orig[p]) style[p] = now[p]; });
      const item = { slide: info.slide, path: info.path, tag: info.tag, text: info.text, style };
      if (info.htmlEdited && now.html !== info.orig.html) {
        // Typing trims the source's indentation around the text; put it back.
        const lead = info.orig.html.match(/^\s*/)[0], trail = info.orig.html.match(/\s*$/)[0];
        item.html = lead + now.html.trim() + trail;
      }
      if (Object.keys(style).length || item.html != null) list.push(item);
    });
    return list;
  }
  function saveEditDraft() { try { const p = pendingEdits(); if (p.length) localStorage.setItem(EDRAFT, JSON.stringify(p)); else localStorage.removeItem(EDRAFT); } catch (e) {} }
  function loadEditDraft() {
    let d = null;
    try { d = JSON.parse(localStorage.getItem(EDRAFT) || 'null'); } catch (e) {}
    if (!Array.isArray(d)) return 0;
    let n = 0;
    d.forEach((item) => {
      const e = byPath(slides[item.slide], item.path);
      if (!e || e.tagName.toLowerCase() !== item.tag) return;
      const info = touch(e);
      info.text = item.text;
      Object.entries(item.style || {}).forEach(([p, v]) => (v ? e.style.setProperty(p, v) : e.style.removeProperty(p)));
      if (item.html != null) { e.innerHTML = item.html; enhance(e); info.htmlEdited = true; }
      n++;
    });
    return n;
  }
  function commit(e, before) {
    const after = snapshot(e);
    if (JSON.stringify(after) === JSON.stringify(before)) return;
    ed.undo.push({ e, before, after });
    ed.redo = [];
    saveEditDraft();
    renderEditBar();
  }
  function renderEditBar() {
    const n = pendingEdits().length;
    ui.edbar.querySelector('.dk-edn').textContent = n ? fmt(L.editN, { n }) : '';
  }
  function place(box, e) {
    const r = e.getBoundingClientRect();
    Object.assign(box.style, { left: r.left + 'px', top: r.top + 'px', width: r.width + 'px', height: r.height + 'px', display: 'block' });
  }
  function select(e) {
    ed.sel = e;
    if (!e) { ui.selbox.style.display = 'none'; return; }
    const cls = Array.from(e.classList).slice(0, 2).map((c) => '.' + c).join('');
    ui.selbox.querySelector('.dk-tag').textContent = e.tagName.toLowerCase() + cls;
    place(ui.selbox, e);
  }
  function enterEdit() {
    ['menu', 'help', 'notes'].forEach((p) => show(p, false));
    if (ink.on) setDraw(false);
    if (laserOn) setLaser(false);
    unzoom(true);
    // Hidden steps must be reachable while editing.
    ed.hadAllSteps = stepsAll;
    showAllSteps(true);
    ed.on = true;
    document.documentElement.classList.add('dk-editing');
    show('edbar', true);
    renderEditBar();
  }
  async function exitEdit() {
    stopTyping();
    select(null);
    ui.hov.style.display = 'none';
    ed.on = false;
    document.documentElement.classList.remove('dk-editing');
    show('edbar', false);
    showAllSteps(!!ed.hadAllSteps);
    const edits = pendingEdits();
    if (!edits.length) return;
    try {
      let skipped = [];
      const name = await writeDeck((src) => { const r = patchEditsReport(src, edits); skipped = r.skipped; return r.out; });
      ed.touched.forEach((info, e) => { info.orig = snapshot(e); info.htmlEdited = false; info.text = srcText(e); });
      ed.undo = []; ed.redo = [];
      try { localStorage.removeItem(EDRAFT); } catch (x) {}
      flash(skipped.length ? fmt(L.editPartial, { n: edits.length - skipped.length, f: name, m: skipped.length })
        : fmt(L.editSaved, { n: edits.length, f: name }), skipped.length ? 8000 : 3500);
    } catch (e) {
      if (e.name === 'AbortError') { flash(L.editNotSaved, 3500); return; }
      if (e.stale) { flash(L.stale, 9000); return; }
      if (e.fallback) { download('edits.json', { deck: deckName(), slides: slides.length, edits }); flash(L.editDownloaded, 6000); return; }
      flash(fmt(L.saveFailed, { e: e.message }), 6000);
    }
  }
  function startTyping(e) {
    if (!isTextual(e)) return;
    const info = touch(e);
    info.htmlEdited = true;
    ed.typing = { e, before: snapshot(e) };
    // Type into the source: $x^2$ rather than the rendered maths.
    unrender(e);
    try { e.contentEditable = e.firstElementChild ? 'true' : 'plaintext-only'; } catch (x) { e.contentEditable = 'true'; }
    ui.selbox.classList.add('dk-typing');
    e.focus();
    const r = document.createRange();
    r.selectNodeContents(e);
    r.collapse(false);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(r);
  }
  function stopTyping() {
    const t = ed.typing;
    if (!t) return;
    t.e.removeAttribute('contenteditable');
    ui.selbox.classList.remove('dk-typing');
    ed.typing = null;
    enhance(t.e);
    commit(t.e, t.before);
  }
  const translateOf = (e) => {
    const v = e.style.translate || getComputedStyle(e).translate;
    if (!v || v === 'none') return [0, 0];
    const [x = 0, y = 0] = v.split(/\s+/).map(parseFloat);
    return [x || 0, y || 0];
  };
  const cssW = (e, w, axis) => {
    const cs = getComputedStyle(e);
    if (cs.boxSizing === 'border-box') return w;
    const pad = axis === 'x'
      ? parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight) + parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth)
      : parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom) + parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
    return Math.max(1, w - pad);
  };
  function startDrag(ev, mode, dir) {
    const e = ed.sel;
    if (!e) return;
    touch(e);
    const before = snapshot(e);
    const r = e.getBoundingClientRect();
    const k = (e.offsetWidth ? r.width / e.offsetWidth : 1) || 1;   // screen px per CSS px
    const [tx0, ty0] = translateOf(e);
    const w0 = e.offsetWidth || r.width / k, h0 = e.offsetHeight || r.height / k;
    const fs0 = parseFloat(getComputedStyle(e).fontSize);
    const x0 = ev.clientX, y0 = ev.clientY;
    let moved = false;
    const move = (m) => {
      let dx = (m.clientX - x0) / k, dy = (m.clientY - y0) / k;
      if (!moved && Math.hypot(m.clientX - x0, m.clientY - y0) < 3) return;
      moved = true;
      if (mode === 'move') {
        if (m.shiftKey) { if (Math.abs(dx) > Math.abs(dy)) dy = 0; else dx = 0; }
        e.style.translate = `${round(tx0 + dx)}px ${round(ty0 + dy)}px`;
      } else {
        let w = w0, h = h0, tx = tx0, ty = ty0;
        if (dir.includes('e')) w = w0 + dx;
        if (dir.includes('w')) { w = w0 - dx; tx = tx0 + dx; }
        if (dir.includes('s')) h = h0 + dy;
        if (dir.includes('n')) { h = h0 - dy; ty = ty0 + dy; }
        w = Math.max(8, w); h = Math.max(8, h);
        const corner = dir.length === 2, textual = isTextual(e);
        if (corner && (textual || isMedia(e)) && !m.shiftKey) {
          // Corners keep proportions; for text that means scaling the type.
          const ratio = w / w0;
          h = h0 * ratio;
          if (dir.includes('w')) tx = tx0 + (w0 - w);
          if (dir.includes('n')) ty = ty0 + (h0 - h);
          if (textual) e.style.fontSize = round(fs0 * ratio, 1) + 'px';
        }
        if (dir !== 'n' && dir !== 's') e.style.width = round(cssW(e, w, 'x')) + 'px';
        if (dir !== 'e' && dir !== 'w' && !(corner && isTextual(e) && !m.shiftKey)) e.style.height = round(cssW(e, h, 'y')) + 'px';
        e.style.translate = `${round(tx)}px ${round(ty)}px`;
      }
      place(ui.selbox, e);
    };
    const up = (u) => {
      window.removeEventListener('pointermove', move, true);
      window.removeEventListener('pointerup', up, true);
      if (moved) commit(e, before);
      else if (mode === 'move') drill(u);
    };
    window.addEventListener('pointermove', move, true);
    window.addEventListener('pointerup', up, true);
  }
  /** Click (without dragging) on the selection: select the child under the pointer. */
  function drill(ev) {
    // Rendered maths / highlighted code count as their author element.
    const under = document.elementsFromPoint(ev.clientX, ev.clientY).map(liftGen).find((n) => n && n !== ed.sel && ed.sel.contains(n) && n !== ui.host && editable(n));
    if (under) select(under);
  }
  function undoRedo(redo) {
    const from = redo ? ed.redo : ed.undo, to = redo ? ed.undo : ed.redo;
    const step = from.pop();
    if (!step) return;
    restoreSnap(step.e, redo ? step.after : step.before);
    to.push(step);
    if (ed.sel) place(ui.selbox, ed.sel);
    saveEditDraft();
    renderEditBar();
  }
  function nudge(dx, dy) {
    const e = ed.sel;
    touch(e);
    const before = snapshot(e);
    const [tx, ty] = translateOf(e);
    e.style.translate = `${round(tx + dx)}px ${round(ty + dy)}px`;
    place(ui.selbox, e);
    commit(e, before);
  }
  /** Keys while editing. Returns true when the key was used. */
  function editKey(e) {
    const k = e.key;
    if ((e.ctrlKey || e.metaKey) && /^z$/i.test(k)) { undoRedo(e.shiftKey); return true; }
    if ((e.ctrlKey || e.metaKey) && /^y$/i.test(k)) { undoRedo(true); return true; }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    if (k === 'e' || k === 'E') { exitEdit(); return true; }
    if (k === 'm' || k === 'M') { show('menu'); return true; }
    if (k === 'Escape') {
      if (ui.menu.classList.contains('dk-show')) show('menu', false);
      else if (ed.sel) select(editable(ed.sel.parentElement) ? ed.sel.parentElement : null);
      return true;
    }
    if (ed.sel && (k === 'Delete' || k === 'Backspace')) {
      const s = ed.sel;
      touch(s);
      const before = snapshot(s);
      s.style.display = 'none';
      commit(s, before);
      select(null);
      return true;
    }
    if (ed.sel && /^Arrow/.test(k)) {
      const d = e.shiftKey ? 10 : 1;
      nudge(k === 'ArrowLeft' ? -d : k === 'ArrowRight' ? d : 0, k === 'ArrowUp' ? -d : k === 'ArrowDown' ? d : 0);
      return true;
    }
    if (NAV_KEYS.test(k) || /^[0-9]$/.test(k)) return false;   // let navigation through
    return true;                                                // swallow other shortcuts
  }

  // ── Pick an element for the agent (K) ────────────────────────────────────
  let picking = false;
  let eatClickUntil = 0;   // swallow the click that follows a pick, so the deck doesn't react to it
  function cssPathIn(e, s) {
    const parts = [];
    for (let n = e; n && n !== s; n = n.parentElement) {
      let part = n.tagName.toLowerCase();
      const cls = Array.from(n.classList).filter((c) => !/^dk-/.test(c)).slice(0, 2);
      if (n.id) part += '#' + n.id;
      else if (cls.length) part += '.' + cls.join('.');
      const sibs = Array.from(n.parentElement.children).filter((x) => x.tagName === n.tagName);
      if (!n.id && sibs.length > 1) part += `:nth-of-type(${sibs.indexOf(n) + 1})`;
      parts.unshift(part);
    }
    return parts.join(' > ');
  }
  async function describe(e) {
    const s = slideOf(e);
    const i = slides.indexOf(s);
    const file = location.protocol === 'file:' ? decodeURIComponent(location.pathname) : location.href.replace(/[?#].*$/, '');
    const tag = e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + Array.from(e.classList).filter((c) => !/^dk-/.test(c)).map((c) => '.' + c).join('');
    const text = normText(e.textContent);
    const snip = text.length > 90 ? text.slice(0, 87) + '…' : text;
    let line = '';
    try {
      const src = await readSource();
      if (src) {
        const t = locateIn(sourceIndex(src), i, pathOf(e, s), e.tagName.toLowerCase(), e.textContent);
        if (t) line = `, line ${src.slice(0, t.start).split('\n').length}`;
      }
    } catch (x) {}
    return `${file}${line} — slide ${i + 1} "${titleOf(s, i)}" — <${tag}>${snip ? ` "${snip}"` : ''} — selector: ${cssPathIn(e, s) || '(the slide itself)'}`;
  }
  async function copyText(t) {
    try { await navigator.clipboard.writeText(t); return true; } catch (x) {}
    const ta = el('textarea');
    ta.value = t;
    ta.style.cssText = 'position:fixed;left:-9999px';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (x) {}
    ta.remove();
    return ok;
  }
  function startPick() {
    picking = true;
    document.documentElement.classList.add('dk-pick');
    flash(L.pick, 4000);
  }
  function stopPick() {
    picking = false;
    document.documentElement.classList.remove('dk-pick');
    ui.hov.style.display = 'none';
  }

  // Pointer handling shared by edit mode and pick mode (capture phase, so the
  // template's own click handlers and links don't fire).
  function onEditPointer(ev) {
    if (!ed.on && !picking) {
      if (/^(click|mouseup|touchend)$/.test(ev.type) && performance.now() < eatClickUntil) {
        ev.preventDefault(); ev.stopPropagation();
        if (ev.type === 'click') eatClickUntil = 0;
      }
      return;
    }
    const path = ev.composedPath();
    const target = path.includes(ui.host) ? path[0] : liftGen(path[0]);
    if (path.includes(ui.host)) {
      if (ev.type !== 'pointerdown' || !ed.on) return;
      const h = target.closest && target.closest('.dk-h');
      if (h) { ev.preventDefault(); startDrag(ev, 'resize', h.dataset.dir); return; }
      if (target === ui.selbox || (target.closest && target.closest('.dk-sel'))) { ev.preventDefault(); startDrag(ev, 'move'); }
      return;
    }
    if (ed.typing && ed.typing.e.contains(target)) return;   // caret placement while typing
    const inSlide = editable(target);
    if (ev.type === 'pointermove') {
      if (inSlide && target !== ed.sel) place(ui.hov, target); else ui.hov.style.display = 'none';
      return;
    }
    if (!inSlide) {
      if (ev.type === 'pointerdown' && ed.on) { stopTyping(); select(null); }
      return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    if (ev.type !== 'pointerdown') return;
    if (picking) {
      stopPick();
      eatClickUntil = performance.now() + 1000;
      describe(target).then((t) => copyText(t).then(() => flash(fmt(L.copied, { t }), 6000)));
      return;
    }
    stopTyping();
    select(target);
    ui.hov.style.display = 'none';
    startDrag(ev, 'move');
  }
  function onEditDblClick(ev) {
    if (!ed.on || !ed.sel) return;
    const path = ev.composedPath();
    if (path.includes(ui.host) || editable(liftGen(path[0]))) { ev.preventDefault(); ev.stopPropagation(); startTyping(ed.sel); }
  }

  // ── Export & share (X) ───────────────────────────────────────────────────
  // A page can't run programs, so this panel prints directly and builds the
  // export / deploy commands (with this deck's real path) to paste into a
  // terminal or hand to the agent.
  function toolsDir() {
    const meta = document.querySelector('meta[name="deck-kit-tools"]');
    if (meta && meta.content) return meta.content.replace(/\/$/, '');
    // A deck inside the repo itself (templates/…, examples/…).
    const m = localPath().match(/^(.*\/deck-kit)\/(templates|examples)\//);
    if (location.protocol === 'file:' && m) return m[1] + '/scripts';
    return '$HOME/.agents/skills/deck-kit/scripts';
  }
  // file:///C:/x.html has pathname /C:/x.html on Windows; drop the leading slash there.
  const localPath = () => decodeURIComponent(location.pathname).replace(/^\/([A-Za-z]:\/)/, '$1');
  const deckPath = () => (location.protocol === 'file:' ? localPath() : location.href.replace(/[?#].*$/, ''));
  /** Same link name rule as deploy.mjs: the folder for index.html/template.html, else the file name. */
  function defaultName() {
    const parts = decodeURIComponent(location.pathname).split('/').filter(Boolean);
    const file = parts.pop() || 'deck';
    const base = /^(index|template)\.html?$/i.test(file) ? (parts.pop() || 'deck') : file.replace(/\.html?$/i, '');
    return base.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 52) || 'deck';
  }
  // Double quotes keep spaces together and still let the shell expand $HOME.
  const q = (s) => '"' + String(s).replace(/(["\\`])/g, '\\$1') + '"';
  function shareCommand(kind) {
    const box = ui.share;
    if (kind === 'export') {
      const flags = Array.from(box.querySelectorAll('[data-f]:checked')).map((c) => '--' + c.dataset.f);
      return flags.length ? `node ${q(toolsDir() + '/export.mjs')} ${q(deckPath())} ${flags.join(' ')}` : '';
    }
    const to = (box.querySelector('input[name="dk-to"]:checked') || {}).value || 'vercel';
    const name = box.querySelector('[data-o="name"]').value.trim();
    const preview = box.querySelector('[data-o="preview"]').checked;
    const notes = box.querySelector('[data-o="notes"]').checked;
    return `node ${q(toolsDir() + '/deploy.mjs')} ${q(deckPath())} --to=${to}${name ? ` --name=${q(name)}` : ''}${preview ? ' --preview' : ''}${notes ? ' --keep-notes' : ''}`;
  }
  function renderShare() {
    ['export', 'deploy'].forEach((k) => { ui.share.querySelector(`[data-c="${k}"]`).textContent = shareCommand(k) || L.pickFormat; });
  }
  function setupShare() {
    const box = ui.share;
    box.querySelector('[data-o="name"]').value = defaultName();
    box.addEventListener('input', renderShare);
    box.addEventListener('change', renderShare);
    box.addEventListener('click', (e) => {
      const a = e.target.closest('button') && e.target.closest('button').dataset.a;
      if (a === 'print') { show('share', false); api.print(); }
      if (a === 'export' || a === 'deploy') {
        const cmd = shareCommand(a);
        if (!cmd) { flash(L.pickFormat); return; }
        copyText(cmd).then(() => flash(L.cmdCopied, 4000));
      }
    });
    renderShare();
  }

  // ── Links between slides ─────────────────────────────────────────────────
  // <a href="#some-id"> jumps to the slide holding that id (#7 = slide 7).
  // Chrome keeps such links clickable in a printed PDF.
  function slideForHash(h) {
    const id = decodeURIComponent((h || '').replace(/^#/, ''));
    if (!id) return -1;
    if (/^\d+$/.test(id)) return +id - 1;
    let t = null;
    try { t = document.getElementById(id); } catch (x) {}
    return t ? slides.findIndex((s) => s === t || s.contains(t)) : -1;
  }
  function onLinkClick(ev) {
    if (ed.on || picking) return;
    const a = ev.target.closest && ev.target.closest('a[href^="#"]');
    if (!a) return;
    const k = slideForHash(a.getAttribute('href'));
    if (k < 0 || k >= slides.length) return;
    ev.preventDefault();
    ev.stopPropagation();
    goto(k);
  }

  // ── Video, audio and embedded players ────────────────────────────────────
  // Nothing plays by itself: autoplay is ignored and every <video>/<audio>
  // gets the browser's controls (play, seek, volume, mute). Leaving a slide
  // stops its media and rewinds it, so coming back shows it unplayed.
  // YouTube / Vimeo iframes are paused through their player API.
  const YT = /^https?:\/\/(www\.)?(youtube(-nocookie)?\.com|youtu\.be)\//i;
  const VIMEO = /^https?:\/\/player\.vimeo\.com\//i;
  function prepareMedia() {
    slides.forEach((s) => {
      s.querySelectorAll('video, audio').forEach((m) => {
        m.removeAttribute('autoplay');
        m.autoplay = false;
        if (!m.hasAttribute('controls')) m.controls = true;
        if (!m.hasAttribute('preload')) m.preload = 'metadata';
        if (!m.paused) m.pause();
      });
      s.querySelectorAll('iframe[src]').forEach((f) => {
        const src = f.getAttribute('src');
        if (YT.test(src) || VIMEO.test(src)) {
          // No autoplay; YouTube also needs the player API so it can be paused.
          try {
            const u = new URL(src, location.href);
            u.searchParams.delete('autoplay');
            if (YT.test(src)) { u.searchParams.set('enablejsapi', '1'); u.searchParams.set('playsinline', '1'); }
            if (u.href !== src) f.src = u.href;
          } catch (e) {}
        }
        if (!f.hasAttribute('allow')) f.setAttribute('allow', 'fullscreen; picture-in-picture; encrypted-media');
      });
    });
  }
  function stopMedia(s) {
    if (!s) return;
    s.querySelectorAll('video, audio').forEach((m) => {
      try { m.pause(); if (m.currentTime) m.currentTime = 0; } catch (e) {}
    });
    s.querySelectorAll('iframe[src]').forEach((f) => {
      const src = f.getAttribute('src') || '', w = f.contentWindow;
      if (!w) return;
      if (YT.test(src)) {
        // Pause first: a paused YouTube player stays paused after seekTo.
        ['pauseVideo', 'seekTo'].forEach((func) => w.postMessage(JSON.stringify({ event: 'command', func, args: func === 'seekTo' ? [0, true] : [] }), '*'));
      } else if (VIMEO.test(src)) {
        w.postMessage(JSON.stringify({ method: 'pause' }), '*');
        w.postMessage(JSON.stringify({ method: 'setCurrentTime', value: 0 }), '*');
      }
    });
  }

  // ── Navigation hook ──────────────────────────────────────────────────────
  function onChange(i, prev) {
    if (bounceStep(i, prev)) return;
    if (prev >= 0 && slides[prev]) widgetsOn(slides[prev]).forEach((w) => call(w, 'leave'));
    if (prev >= 0 && prev !== i) stopMedia(slides[prev]);
    if (zoom.on || zoom.picking) unzoom(true);
    // Arrived without deck-kit (the template's own click / swipe): forward
    // shows no steps yet, backward shows them all.
    if (stepState.slide !== i) setSteps(i, prev >= 0 && i < prev ? Infinity : 0);
    cur = i;
    const s = slides[i];
    widgetsOn(s).forEach((w) => { call(w, 'enter'); call(w, 'draw'); });
    s.querySelectorAll('[data-countdown][data-autostart]').forEach((e) => { cdState(e).running = cdState(e).left > 0; });
    if (plan.has) ui.progress.style.width = ((plan.start[i] + plan.mins[i]) / plan.total * 100) + '%';
    ui.menu.querySelectorAll('button').forEach((b, k) => b.classList.toggle('dk-cur', k === i));
    if (i > 0 && ROLE === 'main') startTimer();
    if (!adapter.ownsHash && ROLE === 'main') { try { history.replaceState(null, '', '#' + (i + 1)); } catch (e) {} }
    renderNotes();
    renderTimer();
    if (ink.on || ink.strokes[i] || ink.strokes[prev]) drawInk();
    broadcast();
    sendSync();
  }

  let last = performance.now();
  function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (ROLE !== 'presenter') {
      const i = adapter.index();
      if (pending) {
        if (i === pending.i) pending = null;
        else if (now > pending.until) {
          // The template never got there: the step state goes back to the slide on screen.
          const b = pending.back;
          pending = null;
          if (i >= 0 && stepState.slide !== i) setSteps(i, b.slide === i ? b.n : Infinity);
        }
        else if (now >= pending.next) { pending.next = now + 150; adapter.go(pending.i); }
      }
      if (i !== cur && i >= 0 && i < slides.length) onChange(i, cur);
      // A preview follows the audience screen only; undo moves the template made on its own.
      if (ROLE === 'mirror' && pinned != null && !pending && i !== pinned) goto(pinned, stepState.slide === pinned ? stepState.n : 0);
      if (cur >= 0) widgetsOn(slides[cur]).forEach((w) => call(w, 'tick', dt));
      cdTick(dt);
    }
    if (timer.started) renderTimer();
    if (ed.on && ed.sel) { if (ed.sel.isConnected && slides[cur] && slides[cur].contains(ed.sel)) place(ui.selbox, ed.sel); else select(null); }
    requestAnimationFrame(loop);
  }

  let resizeQueued = false;
  function onResize() {
    if (resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => {
      resizeQueued = false;
      if (cur >= 0) widgetsOn(slides[cur]).forEach((w) => call(w, 'draw'));
      drawInk();
    });
  }

  // ── Keyboard ─────────────────────────────────────────────────────────────
  const NAV_KEYS = /^(ArrowLeft|ArrowRight|ArrowUp|ArrowDown|Home|End|PageUp|PageDown| |Spacebar)$/;
  /** Shortcuts shared by the main and presenter windows. Returns true if used. */
  function handleKey(key, shift) {
    if (ROLE === 'presenter' && /^[bBwWlLdDcC.]$|^Backspace$/.test(key)) { send(window.opener, { dk: 'key', key, shift }); return true; }
    switch (key) {
      case 'n': case 'N': if (ROLE === 'main') show('notes'); break;
      case 'm': case 'M': show('menu'); break;
      case 'p': case 'P': if (ROLE === 'main') openPresenter(); break;
      case 'e': case 'E': if (ROLE !== 'main' || PUBLISHED) return false; enterEdit(); break;
      case 'k': case 'K': if (ROLE !== 'main' || PUBLISHED) return false; startPick(); break;
      case 'x': case 'X': if (ROLE !== 'main' || PUBLISHED) return false; renderShare(); show('share'); break;
      case 't': if (ROLE === 'main') { show('tmr'); renderTimer(); } break;
      case 'T': timerCmd('reset'); flash(L.elapsed + ' 0:00'); break;
      case 'f': case 'F': toggleFullscreen(); break;
      case 'z': case 'Z':
        if (ROLE === 'presenter') { if (pv) pvZoomKey(); break; }
        if (ROLE !== 'main' || ed.on) return false;
        startZoom(); break;
      case 'b': case 'B': case '.': setBlank('black'); broadcast(); break;
      case 'w': case 'W': setBlank('white'); broadcast(); break;
      case 'l': case 'L': setLaser(); break;
      case 'd': case 'D': setDraw(); break;
      case 'c': case 'C': if (ink.strokes[cur]) { delete ink.strokes[cur]; drawInk(); } else return false; break;
      case 'Backspace': if (ink.on && ink.strokes[cur] && ink.strokes[cur].length) { ink.strokes[cur].pop(); drawInk(); } else return false; break;
      case '?': show('help'); break;
      case 'Escape':
        if (ROLE === 'presenter' && pv && (pvZoom.on || pvZoom.picking)) pvZoomKey();
        if (picking) stopPick();
        if (zoom.on || zoom.picking) unzoom();
        ['menu', 'help', 'notes', 'share'].forEach((p) => show(p, false));
        if (ink.on) setDraw(false);
        if (laserOn) setLaser(false);
        if (blank) setBlank(blank);
        break;
      default: return false;
    }
    return true;
  }

  function onKey(e) {
    if (synthetic) return;
    // Events from our shadow UI are retargeted to the host; look at the real target.
    const t = (e.composedPath && e.composedPath()[0]) || e.target;
    if (editing && e.key === 'Escape') { setEditing(false); e.preventDefault(); return; }
    if (ed.typing) {
      if (e.key === 'Escape') { stopTyping(); e.preventDefault(); e.stopImmediatePropagation(); return; }
      e.stopImmediatePropagation();   // keys belong to the text being typed
      return;
    }
    if (ed.on) {
      if (editKey(e)) { e.preventDefault(); e.stopImmediatePropagation(); return; }
      if (!NAV_KEYS.test(e.key) && !/^[0-9]$/.test(e.key)) return;
    }
    const typing = t && (t.isContentEditable || /^(TEXTAREA|SELECT)$/.test(t.tagName) ||
      (t.tagName === 'INPUT' && !/^(range|checkbox|radio|button|submit|reset|color)$/.test(t.type)));
    const onControl = t && t.tagName === 'INPUT' && /^(range|checkbox|radio)$/.test(t.type);
    // Let sliders and inputs keep their keys: stop the deck from seeing them.
    if ((typing || onControl) && NAV_KEYS.test(e.key)) { e.stopImmediatePropagation(); return; }
    if (typing) return;

    if ((e.ctrlKey || e.metaKey) && (e.key === 'p' || e.key === 'P') && ROLE === 'main') {
      e.preventDefault(); e.stopImmediatePropagation();
      api.print();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    // While the menu is open, arrow keys move between rows instead of slides.
    if (ui.menu.classList.contains('dk-show') && /^(ArrowUp|ArrowDown)$/.test(e.key)) {
      const bs = Array.from(ui.menu.querySelectorAll('button'));
      const k = bs.indexOf(ui.root.activeElement);
      const nb = bs[Math.max(0, Math.min(bs.length - 1, k + (e.key === 'ArrowDown' ? 1 : -1)))];
      if (nb) nb.focus();
      e.preventDefault(); e.stopImmediatePropagation();
      return;
    }
    if (handleKey(e.key, e.shiftKey)) { e.preventDefault(); e.stopImmediatePropagation(); return; }

    // Navigation: <deck-stage> keeps its own handler in the main window;
    // everywhere else deck-kit drives it so presses queue instead of dropping.
    // (deck-kit still takes the keys that reveal or hide a step.)
    if (adapter.ownsKeys && ROLE === 'main' && !stepKey(e.key)) return;
    if (navKey(e.key, e.shiftKey)) { e.preventDefault(); e.stopImmediatePropagation(); }
  }

  /** Navigation keys. Returns true if the key moved (or tried to move) the deck. */
  function navKey(key, shift) {
    let d = null, abs = null;
    if (/^(ArrowRight|ArrowDown|PageDown| |Spacebar)$/.test(key)) d = 1;
    else if (/^(ArrowLeft|ArrowUp|PageUp)$/.test(key)) d = -1;
    else if (key === 'Home') abs = 0;
    else if (key === 'End') abs = slides.length - 1;
    else if (/^[0-9]$/.test(key)) abs = key === '0' ? 9 : +key - 1;
    else if ((key === 'r' || key === 'R') && !shift) abs = 0;
    if (d === null && abs === null) return false;
    if (d !== null) advance(d); else if (abs < slides.length) goto(abs);
    return true;
  }

  /** Inside a presenter preview: keys belong to the presenter window. */
  function onMirrorKey(e) {
    const t = e.target;
    const typing = t && (t.isContentEditable || /^(TEXTAREA|SELECT)$/.test(t.tagName) ||
      (t.tagName === 'INPUT' && !/^(range|checkbox|radio|button|submit|reset|color)$/.test(t.type)));
    const onControl = t && t.tagName === 'INPUT' && /^(range|checkbox|radio)$/.test(t.type);
    if ((typing || onControl) && NAV_KEYS.test(e.key)) { e.stopImmediatePropagation(); return; }
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    send(window.parent, { dk: 'key', key: e.key, shift: e.shiftKey });
    e.preventDefault(); e.stopImmediatePropagation();
  }

  function toggleFullscreen() {
    try {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) {
        const r = document.documentElement.requestFullscreen();
        if (r && r.catch) r.catch(() => {});
      }
    } catch (e) {}
  }

  // ── Touch ────────────────────────────────────────────────────────────────
  function setupTouch() {
    // deck-stage has its own tap zones; scroll-based decks keep native scrolling.
    if (adapter.nativeScroll) return;
    let x0 = null, y0 = null;
    const skip = 'input, button, select, textarea, a, canvas, label, [data-noswipe], [data-compare], [data-hotspots], .dk-hs, .dk-opt, #deck-kit';
    // Capture phase + stopPropagation: the template's own swipe code would
    // otherwise move a second time.
    const own = adapter.mode !== 'deck-stage';
    window.addEventListener('touchstart', (e) => {
      if (e.target.closest && e.target.closest(skip)) { x0 = null; return; }
      const t = e.changedTouches[0];
      x0 = t.clientX; y0 = t.clientY;
      if (own) e.stopPropagation();
    }, { passive: true, capture: true });
    window.addEventListener('touchend', (e) => {
      if (x0 === null) return;
      const t = e.changedTouches[0], dx = t.clientX - x0, dy = t.clientY - y0;
      x0 = null;
      if (own) e.stopPropagation();
      if (Math.abs(dx) > 50 && Math.abs(dx) > 1.5 * Math.abs(dy)) advance(dx < 0 ? 1 : -1);
      else if (own && Math.abs(dy) > 60 && Math.abs(dy) > 1.5 * Math.abs(dx)) advance(dy < 0 ? 1 : -1);
    }, { passive: true, capture: true });

    if (matchMedia('(orientation: portrait) and (max-width: 700px)').matches) {
      const r = el('div', 'dk-panel dk-rotate dk-show', `${esc(L.rotate)}<button type="button">${esc(L.close)}</button>`);
      r.querySelector('button').addEventListener('click', () => r.remove());
      ui.root.appendChild(r);
    }
  }

  // ── Print ────────────────────────────────────────────────────────────────
  // What each slide looked like while it was on screen during visitAll().
  // Templates often reset charts and counters when a slide is left; printing
  // re-applies these snapshots (inline styles, leaf text, canvas pixels).
  const visitSnaps = new Map();
  const isWidgetCanvas = (cv) => widgets.some((w) => w.target === cv || (w.target && w.target.contains && w.target.contains(cv)) || w.target === cv.id);
  function snapSlide(s) {
    const recs = [];
    s.querySelectorAll('*').forEach((e) => {
      if (e.closest('aside.notes')) return;
      const r = { el: e, style: e.getAttribute('style') };
      if (!e.firstElementChild && !/^(SCRIPT|STYLE|TEXTAREA)$/.test(e.tagName)) r.text = e.textContent;
      if (e.tagName === 'CANVAS' && e.width && !isWidgetCanvas(e)) { try { r.img = e.toDataURL('image/png'); } catch (x) {} }
      recs.push(r);
    });
    visitSnaps.set(s, recs);
  }

  /** Resolve once slide i stops changing: its text and inline styles are the
   *  same on two checks in a row (count-up numbers, growing bars done).
   *  Finite CSS animations must finish; canvas pixels and infinite keyframe
   *  loops are ignored, so live widgets and ambient motion don't hold it up. */
  async function settleSlide(i, { min = 150, max = 3000 } = {}) {
    const s = slides[i];
    if (!s) return;
    // Position too, so scroll-snap decks count as settled only once scrolling stops.
    const sig = () => Math.round(s.getBoundingClientRect().top) + '|' + s.textContent + '|' + Array.from(s.querySelectorAll('[style]'), (e) => e.getAttribute('style')).join(';');
    const t0 = performance.now();
    await sleep(min);
    let prev = sig(), same = 0;
    // Finite CSS animations / transitions on the slide (entrances) must end too;
    // infinite loops (ambient motion) are ignored.
    const running = () => (document.getAnimations ? document.getAnimations() : []).some((an) => {
      const target = an.effect && an.effect.target;
      if (!target || !(s === target || s.contains(target)) || an.playState !== 'running') return false;
      return isFinite(an.effect.getComputedTiming().endTime);
    });
    while ((same < 2 || running() || busy > 0) && performance.now() - t0 < max) {
      await sleep(150);
      const now = sig();
      same = now === prev ? same + 1 : 0;
      prev = now;
    }
  }

  /** Visit every slide once so templates that render on entry (charts,
   *  counters) fill in their content, ending on the slide we started from. */
  async function visitAll({ settle, progress } = {}) {
    const wait = settle != null ? settle : adapter.mode === 'deck-stage' ? 150 : 450;
    const home = cur, homeSteps = stepState.slide === cur ? stepState.n : 0;
    const order = slides.map((_, i) => i).filter((i) => i !== home).concat([home]);
    for (let k = 0; k < order.length; k++) {
      const i = order[k];
      goto(i, i === home ? homeSteps : 0);
      const until = performance.now() + 2500;
      while (adapter.index() !== i && performance.now() < until) await sleep(50);
      const extra = parseFloat(slides[i].getAttribute('data-print-delay'));
      await settleSlide(i, { min: isFinite(extra) ? extra : wait });
      snapSlide(slides[i]);
      if (progress) progress(k + 1, order.length);
    }
  }

  let printUndo = null;
  function preparePrint() {
    if (printUndo) return;
    const undo = [];
    unzoom(true);
    // Printed pages show every step.
    const hadAll = stepsAll;
    showAllSteps(true);
    undo.push(() => showAllSteps(hadAll));
    if (adapter.mode !== 'deck-stage') {
      // Show every slide, flatten the containers that stack or slide them.
      const activeCls = slides.some((s) => s.classList.contains('is-active')) ? 'is-active' : 'active';
      const usesCls = slides.some((s) => s.classList.contains(activeCls));
      const flow = new Set();
      slides.forEach((s) => {
        const before = s.className;
        s.classList.add('dk-print-slide');
        if (usesCls) { s.classList.add(activeCls); s.classList.remove('prev', 'next', 'is-prev', 'is-next'); }
        undo.push(() => { s.className = before; });
        for (let p = s.parentElement; p && p !== document.body; p = p.parentElement) flow.add(p);
      });
      flow.forEach((p) => { p.classList.add('dk-print-flow'); undo.push(() => p.classList.remove('dk-print-flow')); });
      // Fixed chrome (dots, counters, arrows) would repeat on every page.
      Array.from(document.body.querySelectorAll('*')).forEach((n) => {
        if (n.closest('.dk-print-slide') || flow.has(n) || n.closest('#deck-kit')) return;
        if (getComputedStyle(n).position === 'fixed') { n.classList.add('dk-print-hide'); undo.push(() => n.classList.remove('dk-print-hide')); }
      });
    }
    // Bring back what visitAll() saw on slides the template has since reset.
    const snapImg = new Map();
    visitSnaps.forEach((recs, s) => {
      if (s === slides[cur]) return;
      recs.forEach((r) => {
        if (!r.el.isConnected) return;
        const st = r.el.getAttribute('style');
        if (st !== r.style) {
          if (r.style == null) r.el.removeAttribute('style'); else r.el.setAttribute('style', r.style);
          undo.push(() => { if (st == null) r.el.removeAttribute('style'); else r.el.setAttribute('style', st); });
        }
        if (r.text != null && !r.el.firstElementChild && r.el.textContent !== r.text) {
          const tx = r.el.textContent;
          r.el.textContent = r.text;
          undo.push(() => { r.el.textContent = tx; });
        }
        if (r.img) snapImg.set(r.el, r.img);
      });
    });
    // Redraw every widget (not just the current slide), then freeze canvases
    // into images so they survive the print pipeline.
    widgets.forEach((w) => { if (w.slide) { call(w, 'enter'); call(w, w.hooks.print ? 'print' : 'draw'); } });
    slides.forEach((s) => s.querySelectorAll('canvas').forEach((cv) => {
      if (!cv.width || !cv.offsetWidth) return;
      let src = snapImg.get(cv);
      if (!src) { try { src = cv.toDataURL('image/png'); } catch (e) { return; } }
      const img = new Image();
      img.src = src;
      img.className = 'dk-print-img';
      img.alt = '';
      img.style.width = cv.offsetWidth + 'px';
      img.style.height = cv.offsetHeight + 'px';
      cv.after(img);
      cv.classList.add('dk-print-src');
      undo.push(() => { img.remove(); cv.classList.remove('dk-print-src'); });
    }));
    // Widgets stay "entered" until the print is done (cleanupPrint), so
    // entrance states and drawings are what gets printed.
    printUndo = undo;
  }
  function cleanupPrint() {
    if (!printUndo) return;
    printUndo.reverse().forEach((f) => f());
    printUndo = null;
    widgets.forEach((w) => { if (w.slide && w.slide !== slides[cur]) call(w, 'leave'); });
    if (cur >= 0) widgetsOn(slides[cur]).forEach((w) => call(w, 'draw'));
  }

  // ── Interactive features ─────────────────────────────────────────────────
  // Steps (data-step), math ($…$), code (highlighting, data-lines, copy),
  // quizzes, charts, steppers, before/after sliders, hotspots, QR codes.
  // Everything deck-kit adds inside a slide carries data-dk-gen (or a dk-
  // class / --dk- custom property / data-dk- attribute), and sourceClone()
  // strips it again, so edit mode saves only what the author wrote.
  const isGen = (e) => !!e && e.nodeType === 1 && e.hasAttribute('data-dk-gen');
  const realChildren = (e) => Array.prototype.filter.call(e.children, (c) => !isGen(c));
  /** Nearest ancestor-or-self that is the author's own element. */
  const liftGen = (n) => {
    let g;
    while (n && n.nodeType === 1 && n.closest && (g = n.closest('[data-dk-gen]'))) n = g.parentElement;
    // Token and line spans of highlighted code belong to the <code> element.
    const hl = n && n.nodeType === 1 && n.closest && n.closest('[data-dk-hl]');
    return hl || n;
  };
  /** Mark attributes deck-kit added to an author's element, so they are not saved. */
  function addAttr(e, name, value) {
    if (e.hasAttribute(name)) return;
    e.setAttribute(name, value);
    const list = (e.getAttribute('data-dk-attrs') || '').split(' ').filter(Boolean);
    list.push(name);
    e.setAttribute('data-dk-attrs', list.join(' '));
  }
  /** A copy of `node` as the author wrote it: generated nodes removed, math
   *  and highlighted code turned back into text, deck-kit classes, attributes
   *  and custom properties dropped. */
  function sourceClone(node) {
    const c = node.cloneNode(true);
    const all = () => [c, ...c.querySelectorAll('*')];
    c.querySelectorAll('[data-dk-gen="math"]').forEach((m) => m.replaceWith(document.createTextNode(m.getAttribute('data-dk-src') || '')));
    c.querySelectorAll('[data-dk-gen]').forEach((g) => g.remove());
    all().forEach((x) => { if (x.hasAttribute('data-dk-hl')) x.textContent = x.textContent; });
    all().forEach((x) => {
      (x.getAttribute('data-dk-attrs') || '').split(' ').filter(Boolean).forEach((a) => x.removeAttribute(a));
      Array.from(x.attributes).forEach((a) => { if (a.name.startsWith('data-dk-')) x.removeAttribute(a.name); });
      if (x.classList.length) {
        Array.from(x.classList).forEach((k) => { if (k.startsWith('dk-')) x.classList.remove(k); });
        if (!x.classList.length) x.removeAttribute('class');
      }
      if (x.style && x.style.length) {
        Array.from(x.style).forEach((p) => { if (p.startsWith('--dk-')) x.style.removeProperty(p); });
        if (!x.getAttribute('style').trim()) x.removeAttribute('style');
      }
    });
    return c;
  }
  const sourceHtml = (node) => sourceClone(node).innerHTML;
  const srcText = (node) => sourceClone(node).textContent;
  /** Turn rendered maths and highlighted code inside `node` back into plain
   *  text in place (for typing); other descendants and their listeners stay. */
  function unrender(node) {
    node.querySelectorAll('[data-dk-gen="math"]').forEach((m) => m.replaceWith(document.createTextNode(m.getAttribute('data-dk-src') || '')));
    [node, ...node.querySelectorAll('[data-dk-hl]')].forEach((c) => { if (c.hasAttribute('data-dk-hl')) { c.textContent = c.textContent; c.removeAttribute('data-dk-hl'); } });
    node.querySelectorAll('.dk-copy[data-dk-gen]').forEach((b) => b.remove());
    node.normalize();
  }

  /** Stop the template's own click / swipe handlers (many decks advance on
   *  any click) from seeing interaction with deck-kit controls. */
  function guard(node, sel) {
    ['click', 'mousedown', 'pointerdown', 'touchstart', 'touchend', 'wheel'].forEach((t) =>
      node.addEventListener(t, (e) => {
        // Only the deck-kit control itself (sel), so [data-reveal] / [data-countdown]
        // inside the box still reach the document.
        const c = sel && e.target.closest && e.target.closest(sel);
        if (!sel || (c && node.contains(c))) e.stopPropagation();
      }, { passive: true }));
  }
  const reducedMotion = () => !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);

  // Work the deck must finish before it counts as settled (library loads, chart
  // animations); settleSlide() and printing wait for it.
  let busy = 0;
  function track(p) { busy++; let done = false; const end = () => { if (!done) { done = true; busy--; } }; p.then(end, end); return p; }

  // Libraries load on first use from a CDN unless already on the page
  // (bundle.mjs inlines them for offline decks).
  const LIBS = {
    katex: { test: () => window.katex, js: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js', css: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css' },
    qr: { test: () => window.qrcode, js: 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.js' },
  };
  const libLoads = {};
  function loadLib(name) {
    const lib = LIBS[name];
    if (lib.test()) return Promise.resolve();
    if (libLoads[name]) return libLoads[name];
    if (lib.css && !document.querySelector('link[data-dk-lib="' + name + '"], style[data-dk-lib="' + name + '"]')) {
      const l = el('link'); l.rel = 'stylesheet'; l.href = lib.css; l.setAttribute('data-dk-lib', name);
      document.head.appendChild(l);
    }
    libLoads[name] = new Promise((res, rej) => {
      const s = el('script'); s.src = lib.js; s.async = true; s.setAttribute('data-dk-lib', name);
      s.onload = () => (lib.test() ? res() : rej(new Error(name + ' did not load')));
      s.onerror = () => rej(new Error('could not load ' + lib.js + ' (offline? bundle the deck with scripts/bundle.mjs)'));
      document.head.appendChild(s);
    });
    libLoads[name].catch((e) => console.warn('[deck-kit]', e.message));
    return libLoads[name];
  }

  // Shared state for clickable things that both screens show (quizzes,
  // hotspots, steppers): addressed by slide + kind + order on the slide.
  // (Their slide is looked up on use: DeckKit.stepper() may run before init.)
  const actors = [];   // { kind, el, state(), apply(v) }
  function addActor(kind, node, state, apply) {
    const a = { kind, el: node, state, apply };
    actors.push(a);
    return a;
  }
  const actorsOn = (s, kind) => actors.filter((a) => (!kind || a.kind === kind) && a.el.isConnected && s.contains(a.el));
  function sendAct(a) {
    const s = slideOf(a.el);
    const n = s ? actorsOn(s, a.kind).indexOf(a) : -1;
    if (n >= 0) route({ dk: 'act', slide: slides.indexOf(s), kind: a.kind, n, v: a.state() });
  }
  function applyAct(m) {
    const a = slides[m.slide] && actorsOn(slides[m.slide], m.kind)[m.n];
    if (!a || JSON.stringify(a.state()) === JSON.stringify(m.v)) return;
    applyingRemote = true;
    try { a.apply(m.v); } finally { applyingRemote = false; }
  }

  // ── Steps: reveal parts of a slide one at a time ─────────────────────────
  // [data-step] elements appear one per → press, in page order; data-step="2"
  // puts an element in step 2 (same number → together). Children of
  // [data-steps] each count as a step. pre[data-lines="1-2|4"] adds one step
  // per segment, highlighting those lines. ← hides the last step; entering a
  // slide backwards shows all its steps.
  const stepState = { slide: -1, n: 0 };
  let stepsAll = false;   // print, export and edit mode: everything visible
  function parseLines(spec) {
    return String(spec || '').split('|').map((seg) => {
      seg = seg.trim();
      if (!seg || seg === '*' || seg === 'all') return null;
      const set = new Set();
      seg.split(',').forEach((r) => {
        const m = r.trim().match(/^(\d+)(?:\s*-\s*(\d+))?$/);
        if (!m) return;
        const a = +m[1], b = m[2] ? +m[2] : a;
        for (let k = Math.min(a, b); k <= Math.max(a, b) && k - Math.min(a, b) < 5000; k++) set.add(k);
      });
      return set.size ? set : null;   // nothing readable → no dimming
    });
  }
  function stepGroups(s) {
    if (!s) return [];
    const units = [];
    let auto = 0;
    s.querySelectorAll('[data-step], [data-steps] > *, pre[data-lines]').forEach((e) => {
      if (isGen(e) || e.closest('aside.notes, [data-dk-gen]')) return;
      if (e.hasAttribute('data-step')) {
        const v = parseFloat(e.getAttribute('data-step'));
        units.push({ key: isFinite(v) ? v : ++auto, el: e });
      } else if (e.parentElement && e.parentElement.hasAttribute('data-steps') && !e.matches('pre[data-lines]')) {
        units.push({ key: ++auto, el: e });
      }
      if (e.matches('pre[data-lines]')) parseLines(e.getAttribute('data-lines')).forEach((set, seg) => units.push({ key: ++auto, pre: e, seg, set }));
    });
    const keys = [...new Set(units.map((u) => u.key))].sort((a, b) => a - b);
    return keys.map((k) => units.filter((u) => u.key === k));
  }
  function applySteps(s, n) {
    const groups = stepGroups(s);
    const lines = new Map();
    groups.forEach((g, k) => {
      const on = stepsAll || k < n;
      g.forEach((u) => {
        if (u.el) u.el.classList.toggle('dk-step-off', !on);
        else {
          if (!lines.has(u.pre)) lines.set(u.pre, null);
          if (on && !stepsAll) lines.set(u.pre, u.set);
        }
      });
    });
    lines.forEach((set, pre) => pre.querySelectorAll('.dk-ln').forEach((ln, k) => ln.classList.toggle('dk-dim', !!set && !set.has(k + 1))));
    return groups.length;
  }
  function setSteps(i, n) {
    const s = slides[i];
    if (!s) return;
    const total = stepGroups(s).length;
    stepState.slide = i;
    stepState.n = Math.max(0, Math.min(total, n));
    applySteps(s, stepState.n);
    if (i === cur) broadcast();
  }
  const stepInfo = () => ({ n: stepState.slide === cur ? stepState.n : 0, total: cur >= 0 ? stepGroups(slides[cur]).length : 0 });
  function showAllSteps(on) {
    stepsAll = !!on;
    slides.forEach((s, i) => applySteps(s, i === stepState.slide ? stepState.n : 0));
  }
  /** → / ← : reveal or hide a step, or change slide. */
  function advance(d) {
    if (ROLE === 'presenter') { send(window.opener, { dk: 'nav', d }); return; }
    const t = pending ? pending.i : cur;
    if (t < 0) return;
    const total = stepsAll ? 0 : stepGroups(slides[t]).length;
    const n = stepState.slide === t ? stepState.n : 0;
    if (d > 0 && n < total) { setSteps(t, n + 1); return; }
    if (d < 0 && n > 0 && total) { setSteps(t, n - 1); return; }
    const to = t + d;
    if (to < 0 || to >= slides.length) return;
    goto(to, d < 0 ? Infinity : 0);
  }
  // Templates also change slides on their own: clicks anywhere (or on their
  // ‹ › buttons), the wheel, swipes, deck-stage's tap zones. While the slide
  // still has steps, plain clicks and the wheel go through the steps first;
  // anything that still moves the deck one slide is sent back and becomes a
  // step instead (onChange → bounceStep).
  const INTERACTIVE = 'a[href], button, input, select, textarea, label, summary, canvas, video, audio, iframe, pre, ' +
    '[contenteditable], [tabindex], [role="button"], [role="slider"], [onclick], [data-noswipe], [data-countdown], [data-reveal], ' +
    '[data-quiz], [data-compare], [data-hotspots], .dk-stepper';
  let lastKitGo = { i: -1, at: 0 };   // deck-kit's own last goto (vs. the template moving by itself)
  let lastPress = { el: null, at: 0 }; // what the user last clicked / tapped
  /** A slide-picker dot: one of a row of same-tag siblings with one entry per
   *  slide (outside the slides). Clicking it is a jump, not "next". */
  function isSlidePicker(node) {
    for (let e = node && node.nodeType === 1 ? node : node && node.parentElement; e && e !== document.body; e = e.parentElement) {
      if (slides.includes(e) || slides.some((s) => e.contains(s))) return false;
      const p = e.parentElement;
      if (!p) return false;
      const kin = Array.prototype.filter.call(p.children, (c) => c.tagName === e.tagName);
      if (kin.length === slides.length && p.children.length <= slides.length + 2 && kin.length > 1) return true;
    }
    return false;
  }
  let bounced = { from: -1, to: -1, at: 0 };
  const stepsQuiet = () => ROLE !== 'main' || stepsAll || ed.on || picking || ink.on || zoom.on || zoom.picking || cur < 0;
  function onStepClick(ev) {
    if (stepsQuiet() || ev.button !== 0 || ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.altKey) return;
    const path = ev.composedPath();
    // deck-stage's touch tap zones sit over the slide in its shadow DOM.
    const zone = path.find((n) => n.classList && n.classList.contains('tapzone'));
    if (path.includes(ui.host) || !(zone || path.includes(slides[cur]))) return;
    if (path.some((n) => n.nodeType === 1 && n.matches && n.matches(INTERACTIVE))) return;
    if (String(getSelection() || '').trim()) return;   // selecting text, not advancing
    const { n, total } = stepInfo();
    // Left third = back, as deck-stage's taps.
    const back = zone ? zone.classList.contains('tapzone--back') : ev.clientX < innerWidth / 3;
    if (back ? n === 0 : n >= total) return;
    ev.preventDefault();
    ev.stopPropagation();
    advance(back ? -1 : 1);
  }
  let wheelLock = 0, wheelSum = 0;
  function onStepWheel(ev) {
    if (stepsQuiet() || ev.ctrlKey) return;
    const now = performance.now();
    if (now < wheelLock) { ev.preventDefault(); ev.stopPropagation(); wheelLock = Math.max(wheelLock, now + 250); return; }
    const t = ev.composedPath()[0];
    if (t && t.closest && t.closest('#deck-kit, input, select, textarea')) return;
    const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? innerHeight : 1;
    const d = (Math.abs(ev.deltaY) >= Math.abs(ev.deltaX) ? ev.deltaY : ev.deltaX) * unit;
    // A scrollable box on the slide keeps its own scrolling.
    for (let e = t && t.nodeType === 1 ? t : null; e && e !== slides[cur] && e !== document.body; e = e.parentElement) {
      const oy = getComputedStyle(e).overflowY;
      if (/auto|scroll/.test(oy) && e.scrollHeight > e.clientHeight + 1 &&
        (d > 0 ? e.scrollTop + e.clientHeight < e.scrollHeight - 1 : e.scrollTop > 0)) return;
    }
    const { n, total } = stepInfo();
    if (!(d > 0 ? n < total : d < 0 && n > 0)) { wheelSum = 0; return; }
    ev.preventDefault();
    ev.stopPropagation();
    wheelSum += d;
    if (Math.abs(wheelSum) < 30) return;
    wheelSum = 0;
    // One step per flick: trackpad inertia keeps sending events for a while.
    wheelLock = now + 450;
    advance(d > 0 ? 1 : -1);
  }
  /** The template moved one slide by itself while the old slide still had
   *  steps to show (or hide): go back and take a step instead. */
  function bounceStep(i, prev) {
    if (stepsQuiet() || prev < 0 || Math.abs(i - prev) !== 1 || !slides[prev]) return false;
    const now = performance.now();
    if (lastKitGo.i === i && now - lastKitGo.at < 2500) return false;           // deck-kit sent it there
    if (now - lastPress.at < 1500 && isSlidePicker(lastPress.el)) return false;  // a dot in the template's slide picker
    if (pending && pending.i === prev) return true;                             // already on the way back
    if (bounced.from === prev && bounced.to === i && now - bounced.at < 3000) return false;   // template won't come back: accept
    const pn = stepState.slide === prev ? stepState.n : 0, pt = stepGroups(slides[prev]).length;
    if (i > prev ? pn >= pt : pn <= 0) return false;
    bounced = { from: prev, to: i, at: now };
    goto(prev, pn + (i > prev ? 1 : -1));
    return true;
  }
  /** Would this key change the steps on the current slide (rather than the slide)? */
  function stepKey(key) {
    const d = /^(ArrowRight|ArrowDown|PageDown| |Spacebar)$/.test(key) ? 1 : /^(ArrowLeft|ArrowUp|PageUp)$/.test(key) ? -1 : 0;
    if (!d || stepsAll || cur < 0) return false;
    const { n, total } = stepInfo();
    return d > 0 ? n < total : n > 0;
  }

  // ── Math: $…$, $$…$$, \(…\), \[…\] rendered with KaTeX ───────────────────
  // Inline $…$ follows Pandoc's rule so prices don't turn into maths: no
  // space after the opening $, none before the closing $, and no digit right
  // after it ("$5 and $10" stays text). \$ is a literal dollar sign.
  const MATH_SKIP = 'pre, code, kbd, samp, script, style, textarea, svg, math, .katex, aside.notes, [data-no-math], [data-dk-gen], [contenteditable]';
  function findMath(t) {
    const out = [];
    for (let i = 0; i < t.length; i++) {
      const c = t[i];
      if (c === '\\' && t[i + 1] === '$') { out.push({ start: i, end: i + 2, literal: true }); i++; continue; }
      if (c === '\\' && (t[i + 1] === '(' || t[i + 1] === '[')) {
        const close = t[i + 1] === '(' ? '\\)' : '\\]';
        const j = t.indexOf(close, i + 2);
        if (j > i + 2) { out.push({ start: i, end: j + 2, tex: t.slice(i + 2, j), display: t[i + 1] === '[' }); i = j + 1; }
        continue;
      }
      if (c !== '$') continue;
      if (t[i + 1] === '$') {
        const j = t.indexOf('$$', i + 2);
        if (j > i + 2 && t.slice(i + 2, j).trim()) { out.push({ start: i, end: j + 2, tex: t.slice(i + 2, j), display: true }); i = j + 1; }
        else i++;
        continue;
      }
      if (!t[i + 1] || /\s/.test(t[i + 1])) continue;
      let j = i + 1;
      while ((j = t.indexOf('$', j)) > 0 && t[j - 1] === '\\') j++;
      if (j < 0 || /\s/.test(t[j - 1]) || /\d/.test(t[j + 1] || '') || t[j + 1] === '$') continue;
      out.push({ start: i, end: j + 1, tex: t.slice(i + 1, j), display: false });
      i = j;
    }
    return out;
  }
  function scanMath(root) {
    if (document.querySelector('meta[name="deck-kit-math"][content="off"]')) return [];
    const spans = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (/[$\\]/.test(n.data) && n.parentElement && !n.parentElement.closest(MATH_SKIP) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      const t = node.data, found = findMath(t);
      if (!found.length || !found.some((f) => !f.literal)) return;
      const frag = document.createDocumentFragment();
      let at = 0;
      found.forEach((f) => {
        if (f.start > at) frag.appendChild(document.createTextNode(t.slice(at, f.start)));
        const sp = el('span', 'dk-math');
        sp.setAttribute('data-dk-gen', 'math');
        sp.setAttribute('data-dk-src', t.slice(f.start, f.end));
        if (f.literal) sp.textContent = '$';
        else {
          sp.textContent = t.slice(f.start, f.end);
          sp.setAttribute('data-tex', f.tex);
          if (f.display) sp.setAttribute('data-display', '');
          spans.push(sp);
        }
        frag.appendChild(sp);
        at = f.end;
      });
      if (at < t.length) frag.appendChild(document.createTextNode(t.slice(at)));
      node.replaceWith(frag);
    });
    return spans;
  }
  function renderMath(spans) {
    if (!spans.length) return;
    track(loadLib('katex').then(() => spans.forEach((sp) => {
      if (!sp.isConnected && !sp.parentNode) return;
      try { window.katex.render(sp.getAttribute('data-tex'), sp, { displayMode: sp.hasAttribute('data-display'), throwOnError: false }); }
      catch (e) { sp.title = e.message; }
    }))).catch(() => {});
  }

  // ── Code: highlighting, data-lines, copy button ──────────────────────────
  // <pre><code class="language-js">…</code></pre> is highlighted with the
  // template's own accent colours (only plain-text code: hand-written markup
  // inside <code> is left alone). Highlighted blocks, and <pre data-copy>,
  // get a Copy button that shows on hover.
  const KW = 'abstract and as assert async await break case catch class const continue def default defer del delete do elif else enum except export extends false final finally fn for from func function go if impl import in instanceof interface is lambda let loop match mod mut new nil none not null or package pass private protected pub public raise return self static struct super switch this throw throws trait true try type typeof use var void while with yield None True False int float double char bool boolean string long unsigned auto echo fi then done esac local where select insert update into values join on group order by having create table';
  const HASH_LANGS = /^(py|python|sh|bash|shell|zsh|console|rb|ruby|r|yaml|yml|toml|perl|pl|ps1|powershell|make|makefile|dockerfile|ini|conf|nix|julia|jl|elixir|ex|exs|coffee|tcl|cmake)$/;
  const PLAIN_LANGS = /^(none|text|plain|plaintext|txt|output|log)$/;
  function codeLang(code) {
    const m = (code.className || '').match(/\b(?:language|lang)-([\w+#-]+)/i) || [null, code.parentElement && code.parentElement.getAttribute('data-lang')];
    return (m[1] || '').toLowerCase();
  }
  function tokenize(text, lang) {
    const rules = [];
    if (!lang || PLAIN_LANGS.test(lang)) return [['', text]];
    if (/^(html|xml|svg|vue|xhtml)$/.test(lang)) {
      rules.push(['com', /<!--[\s\S]*?-->/], ['kw', /<\/?[A-Za-z][\w:.-]*|\/?>/], ['str', /"[^"]*"|'[^']*'/], ['attr', /[\w:.-]+(?==)/]);
    } else if (/^(css|scss|less)$/.test(lang)) {
      rules.push(['com', /\/\*[\s\S]*?\*\//], ['str', /"[^"\n]*"|'[^'\n]*'/], ['attr', /[\w-]+(?=\s*:[^;{}]*[;}])/], ['num', /#[\da-fA-F]{3,8}\b|-?\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms|deg|fr)?/], ['kw', /@[\w-]+|!important/]);
    } else {
      const hash = HASH_LANGS.test(lang), dash = /^(sql|lua|hs|haskell|ada)$/.test(lang);
      rules.push(['com', hash ? /#.*/ : dash ? /--.*/ : /\/\/.*|\/\*[\s\S]*?\*\//]);
      if (/^(py|python)$/.test(lang)) rules.push(['str', /"""[\s\S]*?"""|'''[\s\S]*?'''/]);
      rules.push(['str', /"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\[\s\S]|[^`\\])*`/]);
      rules.push(['num', /\b(?:0[xX][\da-fA-F]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/]);
      rules.push(['kw', new RegExp('\\b(?:' + KW.split(' ').join('|') + ')\\b', lang === 'sql' ? 'i' : '')]);
      rules.push(['fn', /\b[A-Za-z_$][\w$]*(?=\s*\()/]);
    }
    const re = new RegExp(rules.map(([, r]) => '(' + r.source + ')').join('|'), 'g' + (rules.some(([, r]) => r.flags.includes('i')) ? 'i' : ''));
    const out = [];
    let at = 0, m;
    while ((m = re.exec(text))) {
      if (!m[0]) { re.lastIndex++; continue; }
      if (m.index > at) out.push(['', text.slice(at, m.index)]);
      const g = m.slice(1).findIndex((x) => x !== undefined);
      out.push([rules[g][0], m[0]]);
      at = m.index + m[0].length;
    }
    if (at < text.length) out.push(['', text.slice(at)]);
    return out;
  }
  /** Highlight (and split into lines for data-lines) one <code> element. */
  function renderCode(code, pre) {
    if (code.hasAttribute('data-dk-hl') || code.firstElementChild) return;
    const lang = codeLang(code);
    const wantLines = pre.hasAttribute('data-lines');
    if (!lang && !wantLines) return;
    const text = code.textContent;
    const lead = text.match(/^\n?/)[0];   // a newline right after <code> isn't line 1
    const toks = tokenize(text.slice(lead.length), lang);
    const lines = [[]];
    toks.forEach(([cls, s]) => s.split('\n').forEach((part, k) => {
      if (k) lines.push([]);
      if (part) lines[lines.length - 1].push(cls ? `<span class="dk-tk-${cls}">${esc(part)}</span>` : esc(part));
    }));
    // A trailing newline doesn't start a real last line.
    const trail = lines.length > 1 && !lines[lines.length - 1].length ? lines.pop() && '\n' : '';
    const html = lines.map((l, k) => `<span class="dk-ln">${l.join('')}${k < lines.length - 1 ? '\n' : ''}</span>`).join('');
    code.innerHTML = esc(lead) + html + trail;
    code.setAttribute('data-dk-hl', lang || 'plain');
    const pal = palette(pre);
    const a = pal.accents;
    [['kw', a[0]], ['str', a[1] || a[0]], ['num', a[2] || a[0]], ['fn', a[3] || a[1] || a[0]], ['com', pal.mute], ['attr', a[1] || a[0]]].forEach(([k, c]) => pre.style.setProperty('--dk-c-' + k, c));
  }
  function setupCode(pre) {
    const code = pre.querySelector(':scope > code') || null;
    if (code) renderCode(code, pre);
    else if (pre.hasAttribute('data-lines') && !pre.querySelector('.dk-ln') && !pre.firstElementChild) {
      // Bare <pre data-lines> without <code>: wrap the lines the same way.
      const fake = el('code');
      fake.textContent = pre.textContent;
      renderCode(fake, pre);
      pre.innerHTML = fake.innerHTML;
      pre.setAttribute('data-dk-hl', 'plain');
    }
    const wantCopy = pre.getAttribute('data-copy') !== 'false' && (pre.hasAttribute('data-copy') || (code && codeLang(code)));
    if (wantCopy || pre.hasAttribute('data-dk-hl') || pre.querySelector(':scope > [data-dk-hl]')) pre.classList.add('dk-code');
    if (wantCopy && !pre.querySelector(':scope > .dk-copy')) {
      const b = el('button', 'dk-copy', esc(L.copy));
      b.type = 'button';
      b.setAttribute('data-dk-gen', '');
      guard(b);
      b.addEventListener('click', () => {
        const t = srcTextOfCode(pre);
        copyText(t).then((ok) => { if (!ok) return; b.textContent = L.copiedCode; setTimeout(() => { b.textContent = L.copy; }, 1500); });
      });
      pre.appendChild(b);
    }
  }
  function srcTextOfCode(pre) {
    const c = sourceClone(pre);
    const code = c.querySelector(':scope > code') || c;
    return code.textContent.replace(/^\n/, '').replace(/\n$/, '');
  }

  // ── Quiz ─────────────────────────────────────────────────────────────────
  // <div data-quiz> … <button data-option data-correct>…</button> … <p data-explain>
  // Without [data-option], the <li>s of the quiz are the options. Clicking an
  // option marks it right/wrong, shows the right answer and the explanation;
  // clicking it again clears the answer. Leaving the slide clears it too.
  const quizOptions = (q) => {
    const o = Array.from(q.querySelectorAll('[data-option]'));
    return o.length ? o : Array.from(q.querySelectorAll('li')).filter((li) => !li.closest('[data-explain]'));
  };
  function setQuiz(q, i) {
    q._dkChoice = i;
    q.classList.toggle('dk-answered', i >= 0);
    quizOptions(q).forEach((o, k) => {
      const right = o.hasAttribute('data-correct');
      o.classList.toggle('dk-right', i >= 0 && right);
      o.classList.toggle('dk-wrong', k === i && !right);
      o.classList.toggle('dk-chosen', k === i);
    });
  }
  function setupQuiz(q) {
    quizOptions(q).forEach((o) => {
      o.classList.add('dk-opt');
      if (!/^(BUTTON|A|INPUT|SELECT|TEXTAREA)$/.test(o.tagName)) { addAttr(o, 'tabindex', '0'); addAttr(o, 'role', 'button'); }
    });
    q.classList.add('dk-quiz');
    if (q._dkQuiz) { setQuiz(q, q._dkChoice == null ? -1 : q._dkChoice); return; }
    q._dkQuiz = true;
    q._dkChoice = -1;
    const actor = addActor('quiz', q, () => q._dkChoice, (v) => setQuiz(q, v));
    const choose = (o) => {
      const i = quizOptions(q).indexOf(o);
      if (i < 0) return;
      setQuiz(q, i === q._dkChoice ? -1 : i);
      sendAct(actor);
    };
    guard(q, '.dk-opt');
    q.addEventListener('click', (e) => { const o = liftGen(e.target).closest && liftGen(e.target).closest('.dk-opt'); if (o && q.contains(o) && !ed.on) choose(o); });
    q.addEventListener('keydown', (e) => {
      const o = e.target.closest && e.target.closest('.dk-opt');
      if (o && e.key === 'Enter' && o.getAttribute('role') === 'button') { e.preventDefault(); e.stopPropagation(); choose(o); }
    });
    api.widget(q, { leave: () => { if (q._dkChoice >= 0) setQuiz(q, -1); } });
  }

  // ── Charts: DeckKit.chart / canvas[data-chart] / table[data-chart] ───────
  // Bar, line, pie and donut charts in the template's colours, drawn from
  // data, growing in when the slide opens.
  const numFmt = (v, spec) => {
    const d = spec.decimals != null ? spec.decimals : Math.abs(v) >= 100 ? 0 : 2;
    let s;
    try { s = new Intl.NumberFormat(document.documentElement.lang || undefined, { maximumFractionDigits: d }).format(v); } catch (e) { s = String(+v.toFixed(d)); }
    return (spec.prefix || '') + s + (spec.unit || '');
  };
  function niceScale(lo, hi, n = 5) {
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (lo > hi) [lo, hi] = [hi, lo];
    if (hi - lo <= Math.abs(hi) * 1e-9 || hi === lo) { const d = Math.abs(hi) * 0.1 || 1; lo -= lo === 0 ? 0 : d; hi += d; }
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), m = raw / mag;
    const st = (m > 5 ? 10 : m > 2 ? 5 : m > 1 ? 2 : 1) * mag;
    const a = Math.floor(lo / st + 1e-9) * st, b = Math.ceil(hi / st - 1e-9) * st;
    const ticks = [];
    for (let k = 0, v = a; v <= b + st / 2 && k < 60; k++, v = a + k * st) ticks.push(+v.toPrecision(12));
    return { lo: a, hi: b, ticks };
  }
  function normChart(spec) {
    const s = Object.assign({ type: 'bar' }, spec);
    s.type = String(s.type).toLowerCase();
    let series = s.series || (s.data ? [{ name: s.name || '', data: s.data }] : []);
    if (!Array.isArray(series)) series = [series];
    series = series.map((x) => (Array.isArray(x) ? { name: '', data: x } : x)).map((x) => ({ name: x.name || '', color: x.color, data: (x.data || []).map((v) => (v == null || v === '' ? NaN : +v)) }));
    const n = Math.max(0, ...series.map((x) => x.data.length));
    s.labels = (s.labels || []).slice(0, n);
    while (s.labels.length < n) s.labels.push(String(s.labels.length + 1));
    s.series = series;
    return s;
  }
  function drawChart(cv, spec, k) {
    const { ctx, w, h } = fitCanvas(cv);
    if (!w || !h) return;
    const s = spec, pal = palette(cv);
    ctx.clearRect(0, 0, w, h);
    const font = getComputedStyle(cv).fontFamily || 'sans-serif';
    const fs = s.fontSize || Math.round(Math.max(12, Math.min(24, h / 20)));
    ctx.font = `${fs}px ${font}`;
    ctx.textBaseline = 'middle';
    const colorOf = (x, j) => x.color || pal.accents[j % pal.accents.length];
    const series = s.series, labels = s.labels;
    if (!series.length || !labels.length) return;
    if (s.type === 'pie' || s.type === 'donut' || s.type === 'doughnut') {
      const vals = series[0].data.map((v) => (isFinite(v) && v > 0 ? v : 0));
      const total = vals.reduce((a, b) => a + b, 0);
      if (!total) return;
      const legendW = Math.min(w * 0.42, Math.max(...labels.map((l) => ctx.measureText(l).width)) + fs * 5);
      const r = Math.max(4, Math.min((w - legendW) / 2, h / 2) - fs * 0.6);
      const cx = (w - legendW) / 2, cy = h / 2;
      let a0 = -Math.PI / 2;
      const sweep = Math.PI * 2 * k;
      vals.forEach((v, j) => {
        const a = (v / total) * Math.PI * 2;
        const end = Math.min(a0 + a, -Math.PI / 2 + sweep);
        if (end > a0) {
          ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, a0, end); ctx.closePath();
          ctx.fillStyle = pal.accents[j % pal.accents.length]; ctx.fill();
          ctx.strokeStyle = pal.bg; ctx.lineWidth = 2; ctx.stroke();
        }
        if (k >= 1 && s.values !== false && v / total >= 0.06) {
          const mid = a0 + a / 2, rr = s.type === 'pie' ? r * 0.62 : r * 0.79;
          ctx.fillStyle = pal.bg; ctx.textAlign = 'center';
          ctx.font = `600 ${fs}px ${font}`;
          ctx.fillText(Math.round(v / total * 100) + '%', cx + Math.cos(mid) * rr, cy + Math.sin(mid) * rr);
          ctx.font = `${fs}px ${font}`;
        }
        a0 += a;
      });
      if (s.type !== 'pie') { ctx.globalCompositeOperation = 'destination-out'; ctx.beginPath(); ctx.arc(cx, cy, r * 0.58, 0, Math.PI * 2); ctx.fill(); ctx.globalCompositeOperation = 'source-over'; }
      const ly = cy - (labels.length * fs * 1.6) / 2 + fs * 0.8;
      labels.forEach((l, j) => {
        const y = ly + j * fs * 1.6, x = w - legendW + fs;
        ctx.fillStyle = pal.accents[j % pal.accents.length]; ctx.fillRect(x, y - fs * 0.4, fs * 0.8, fs * 0.8);
        ctx.fillStyle = pal.ink; ctx.textAlign = 'left';
        ctx.fillText(l + (s.values === false ? '' : '  ' + numFmt(vals[j], s)), x + fs * 1.3, y);
      });
      return;
    }
    // Bar / line: legend on top when there is more than one series.
    let top = fs * 0.6;
    if (series.length > 1 || s.legend) {
      let x = 0;
      ctx.textAlign = 'left';
      series.forEach((ser, j) => {
        ctx.fillStyle = colorOf(ser, j); ctx.fillRect(x, top, fs * 0.8, fs * 0.8);
        ctx.fillStyle = pal.ink; ctx.fillText(ser.name, x + fs * 1.2, top + fs * 0.4);
        x += fs * 2.4 + ctx.measureText(ser.name).width;
      });
      top += fs * 1.8;
    }
    const stacked = !!s.stacked && s.type === 'bar';
    let vals = [];
    if (stacked) labels.forEach((_, i) => {
      let pos = 0, neg = 0;
      series.forEach((ser) => { const v = ser.data[i]; if (isFinite(v)) { if (v >= 0) pos += v; else neg += v; } });
      vals.push(pos, neg);
    });
    else vals = series.flatMap((ser) => ser.data.filter(isFinite));
    if (!vals.length) return;
    const sc = niceScale(s.min != null ? +s.min : Math.min(0, ...vals), s.max != null ? +s.max : Math.max(0, ...vals));
    let lo = s.min != null ? +s.min : sc.lo, hi = s.max != null ? +s.max : sc.hi;
    if (!(hi > lo)) { lo = sc.lo; hi = sc.hi; }
    const ticks = sc.ticks.filter((t) => t >= lo - 1e-9 && t <= hi + 1e-9);
    const left = Math.max(fs, ...ticks.map((t) => ctx.measureText(numFmt(t, s)).width)) + fs * 0.8;
    const right = w - fs * 0.4, bottom = h - fs * 1.9, topY = top + (s.values ? fs * 1.2 : fs * 0.4);
    const Y = (v) => bottom - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo || 1) * (bottom - topY);
    ctx.lineWidth = 1;
    ticks.forEach((t) => {
      ctx.strokeStyle = pal.grid; ctx.beginPath(); ctx.moveTo(left, Y(t)); ctx.lineTo(right, Y(t)); ctx.stroke();
      ctx.fillStyle = pal.mute; ctx.textAlign = 'right'; ctx.fillText(numFmt(t, s), left - fs * 0.4, Y(t));
    });
    const slot = (right - left) / labels.length;
    const X = (i) => left + slot * (i + 0.5);
    // Category labels; thin them out when they don't fit.
    const widest = Math.max(...labels.map((l) => ctx.measureText(l).width));
    const every = Math.max(1, Math.ceil((widest + fs) / slot));
    ctx.fillStyle = pal.mute; ctx.textAlign = 'center';
    labels.forEach((l, i) => { if (i % every === 0) ctx.fillText(l, X(i), bottom + fs * 0.95); });
    const base = Y(Math.max(lo, Math.min(hi, 0)));
    const ease = 1 - (1 - k) ** 3;
    if (s.type === 'line' || s.type === 'area') {
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, left + (right - left) * ease, h); ctx.clip();
      series.forEach((ser, j) => {
        const col = colorOf(ser, j);
        const pts = ser.data.map((v, i) => (isFinite(v) ? [X(i), Y(v)] : null));
        const runs = [[]];
        pts.forEach((p) => (p ? runs[runs.length - 1].push(p) : runs.push([])));
        runs.filter((r) => r.length).forEach((run) => {
          if (s.type === 'area' || s.fill) {
            ctx.beginPath(); run.forEach(([x, y], q) => (q ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
            ctx.lineTo(run[run.length - 1][0], base); ctx.lineTo(run[0][0], base); ctx.closePath();
            ctx.globalAlpha = 0.18; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1;
          }
          ctx.beginPath(); run.forEach(([x, y], q) => (q ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, fs / 6); ctx.lineJoin = 'round'; ctx.stroke();
          run.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, Math.max(3, fs / 5), 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill(); });
        });
        if (s.values && k >= 1) ser.data.forEach((v, i) => { if (isFinite(v)) { ctx.fillStyle = pal.ink; ctx.textAlign = 'center'; ctx.fillText(numFmt(v, s), X(i), Y(v) - fs * 0.9); } });
      });
      ctx.restore();
    } else {
      const group = slot * 0.72, bw = stacked ? group : group / series.length;
      labels.forEach((_, i) => {
        let pos = 0, neg = 0;
        series.forEach((ser, j) => {
          const v = ser.data[i];
          if (!isFinite(v)) return;
          let y0, y1;
          if (stacked) { const from = v >= 0 ? pos : neg; y0 = Y(from); y1 = Y(from + v * ease); if (v >= 0) pos += v; else neg += v; }
          else { y0 = base; y1 = Y(v * ease); }
          const x = X(i) - group / 2 + (stacked ? 0 : j * bw);
          ctx.fillStyle = colorOf(ser, j);
          ctx.fillRect(x + (stacked ? 0 : 1), Math.min(y0, y1), bw - (stacked ? 0 : 2), Math.abs(y1 - y0));
          if (s.values && k >= 1 && !stacked) { ctx.fillStyle = pal.ink; ctx.textAlign = 'center'; ctx.fillText(numFmt(v, s), x + bw / 2, v >= 0 ? y1 - fs * 0.6 : y1 + fs * 0.6); }
        });
        if (s.values && k >= 1 && stacked) { ctx.fillStyle = pal.ink; ctx.textAlign = 'center'; ctx.fillText(numFmt(pos + neg, s), X(i), Y(pos) - fs * 0.6); }
      });
    }
    ctx.strokeStyle = pal.mute; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(left, base); ctx.lineTo(right, base); ctx.stroke();
  }
  function chart(target, spec) {
    let cv = typeof target === 'string' ? (document.getElementById(target.replace(/^#/, '')) || document.querySelector(target)) : target;
    if (!cv) { console.warn('[deck-kit] chart target not found:', target); return null; }
    if (cv.tagName !== 'CANVAS') {
      const c = el('canvas', 'dk-chart');
      c.setAttribute('data-dk-gen', '');
      cv.appendChild(c);
      cv = c;
    }
    let s = normChart(spec || {}), k = 0, raf = 0, release = null;
    const stop = () => { cancelAnimationFrame(raf); if (release) { release(); release = null; } };
    const grow = () => {
      stop();
      if (reducedMotion() || s.animate === false || stepsAll || printUndo) { k = 1; drawChart(cv, s, 1); return; }
      k = 0;
      track(new Promise((res) => { release = res; setTimeout(res, 2000); }));
      const t0 = performance.now();
      const frame = (now) => {
        k = Math.min(1, (now - t0) / (s.duration || 700));
        drawChart(cv, s, k);
        if (k < 1) raf = requestAnimationFrame(frame); else stop();
      };
      raf = requestAnimationFrame(frame);
    };
    api.widget(cv, {
      enter: grow,
      draw: () => drawChart(cv, s, k),
      leave: () => { stop(); k = 0; },
      print: () => { stop(); k = 1; drawChart(cv, s, 1); },
    });
    return {
      canvas: cv,
      update(next) { s = normChart(Object.assign({}, s, next)); k = 1; drawChart(cv, s, 1); return this; },
      redraw() { drawChart(cv, s, k); return this; },
      get spec() { return s; },
    };
  }
  /** table[data-chart="bar|line|pie…"]: first column = labels, header row = series names. */
  function tableChart(t) {
    const rows = Array.from(t.querySelectorAll('tr')).map((r) => Array.from(r.children).map((c) => srcText(c).trim()));
    if (rows.length < 2) return;
    // Header: <thead>, a first row of <th>s, or a first row whose value cells aren't numbers.
    const first = t.querySelector('tr');
    const numeric = (c) => /^[^\p{L}]*\d[^\p{L}]*$/u.test(c);
    const head = t.querySelector('thead tr') || (first && Array.from(first.children).every((c) => c.tagName === 'TH') ? first : null) ||
      (rows[0].slice(1).some((c) => c && !numeric(c)) ? first : null);
    const headRow = head ? Array.from(head.children).map((c) => srcText(c).trim()) : null;
    const body = rows.filter((_, i) => !(head && i === 0 && rows[0].join('|') === headRow.join('|')));
    const num = (c) => { const v = parseFloat(String(c).replace(/[\u2212\u2013]/g, '-').replace(/[\s,]/g, '').replace(/[^\d.eE+-]/g, '')); return isFinite(v) ? v : NaN; };
    const cols = Math.max(...body.map((r) => r.length));
    const series = [];
    for (let j = 1; j < cols; j++) series.push({ name: headRow ? headRow[j] || '' : '', data: body.map((r) => num(r[j])) });
    const spec = { type: t.getAttribute('data-chart') || 'bar', labels: body.map((r) => r[0]), series };
    try { Object.assign(spec, JSON.parse(t.getAttribute('data-chart-options') || '{}')); } catch (e) { console.warn('[deck-kit] bad data-chart-options', e); }
    const box = el('div', 'dk-chart-box');
    box.setAttribute('data-dk-gen', '');
    const h = t.getAttribute('data-chart-height');
    if (h) box.style.height = /^\d+$/.test(h) ? h + 'px' : h;
    const cv = el('canvas', 'dk-chart');
    box.appendChild(cv);
    t.after(box);
    t.classList.add('dk-chart-src');
    chart(cv, spec);
  }

  // ── Stepper: play / pause / step / reset for simulations ─────────────────
  //   DeckKit.stepper('#sim', { frames: 60, interval: 100, reset(), step(k) })
  // Declarative: <div data-stepper data-interval="900"> with [data-frame]
  // children shows one frame at a time.
  function stepper(target, opts = {}) {
    const box = typeof target === 'string' ? (document.getElementById(target.replace(/^#/, '')) || document.querySelector(target)) : target;
    if (!box) { console.warn('[deck-kit] stepper target not found:', target); return null; }
    const o = Object.assign({ frames: Infinity, interval: 700, loop: false }, opts);
    let k = 0, timer = null;
    const bar = el('div', 'dk-stepper', `<button type="button" data-s="reset" title="${esc(L.resetT)}" aria-label="${esc(L.resetT)}">⏮</button><button type="button" data-s="play" title="${esc(L.play)}" aria-label="${esc(L.play)}">▶</button><button type="button" data-s="next" title="${esc(L.stepNext)}" aria-label="${esc(L.stepNext)}">⏭</button><span class="dk-stepper-n"></span>`);
    bar.setAttribute('data-dk-gen', '');
    guard(bar);
    if (/^(CANVAS|IMG|SVG|VIDEO|svg)$/.test(box.tagName) || o.after) box.after(bar); else box.appendChild(bar);
    const [bReset, bPlay, bNext] = bar.querySelectorAll('button');
    const nEl = bar.querySelector('.dk-stepper-n');
    const run = (f, ...a) => { try { if (f) f(...a); } catch (e) { console.error('[deck-kit] stepper failed:', e); } };
    const render = () => {
      nEl.textContent = isFinite(o.frames) ? `${k} / ${o.frames}` : String(k);
      bPlay.textContent = timer ? '⏸' : '▶';
      bPlay.title = timer ? L.pause : L.play;
      bPlay.setAttribute('aria-label', bPlay.title);
      bReset.disabled = k === 0 && !timer;
      bNext.disabled = !o.loop && k >= o.frames;
    };
    const ctl = {
      next() {
        if (k >= o.frames) {
          if (!o.loop) { ctl.pause(); return false; }
          k = 0; run(o.reset, ctx); render();   // frame 0 gets its turn too
          return true;
        }
        k++;
        run(o.step, k, ctx);
        render();
        return true;
      },
      play() {
        if (timer) return;
        if (k >= o.frames && !o.loop) { k = 0; run(o.reset, ctx); }
        timer = setInterval(() => { if (!ctl.next()) ctl.pause(); }, Math.max(16, o.interval));
        render();
      },
      pause() { clearInterval(timer); timer = null; render(); },
      reset() { ctl.pause(); k = 0; run(o.reset, ctx); render(); },
      /** Jump to frame n by replaying from the start (simulations are deterministic). */
      seek(n) { ctl.pause(); k = 0; run(o.reset, ctx); while (k < Math.min(n, isFinite(o.frames) ? o.frames : n)) { k++; run(o.step, k, ctx); } render(); },
      get frame() { return k; },
      get playing() { return !!timer; },
      bar,
    };
    const ctx = { el: box, get frame() { return k; }, kit: api, stepper: ctl };
    const actor = addActor('stepper', box, () => ({ k, p: !!timer }), (v) => { if (v.k !== k) ctl.seek(v.k); if (v.p) ctl.play(); else ctl.pause(); });
    bar.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const a = b.dataset.s;
      if (a === 'reset') ctl.reset();
      if (a === 'next') { ctl.pause(); ctl.next(); }
      if (a === 'play') { if (timer) ctl.pause(); else ctl.play(); }
      sendAct(actor);
    });
    run(o.reset, ctx);
    render();
    api.widget(box, { leave: () => ctl.pause() });
    return ctl;
  }
  function setupFrames(box) {
    if (box._dkStepper && box.querySelector(':scope > .dk-stepper')) return;
    const frames = Array.from(box.querySelectorAll(':scope > [data-frame]'));
    const list = frames.length ? frames : Array.from(box.querySelectorAll('[data-frame]'));
    if (!list.length) return;
    const show = (i) => list.forEach((f, j) => f.classList.toggle('dk-frame-off', j !== i));
    box._dkStepper = stepper(box, {
      frames: list.length - 1,
      interval: parseFloat(box.getAttribute('data-interval')) || 900,
      loop: box.hasAttribute('data-loop'),
      reset: () => show(0),
      step: (k) => show(k),
    });
  }

  // ── Before / after slider ────────────────────────────────────────────────
  // <div data-compare="50"><img src="before.jpg" data-label="Trước"><img src="after.jpg"></div>
  function setupCompare(box) {
    if (box.querySelector(':scope > .dk-cmp-range')) return;
    const kids = realChildren(box);
    if (kids.length < 2) { console.warn('[deck-kit] data-compare needs two children (before, after)', box); return; }
    const [a, b] = kids;
    box.classList.add('dk-cmp');
    a.classList.add('dk-cmp-before');
    b.classList.add('dk-cmp-after');
    const start = parseFloat(box.getAttribute('data-compare'));
    const inp = el('input', 'dk-cmp-range');
    inp.type = 'range'; inp.min = '0'; inp.max = '100'; inp.step = '0.1';
    inp.value = String(isFinite(start) ? Math.max(0, Math.min(100, start)) : 50);
    inp.setAttribute('aria-label', L.compare);
    inp.setAttribute('data-dk-gen', '');
    const handle = el('div', 'dk-cmp-handle');
    handle.setAttribute('data-dk-gen', '');
    box.append(inp, handle);
    [[a, 'dk-cmp-l'], [b, 'dk-cmp-r']].forEach(([x, cls]) => {
      const t = x.getAttribute('data-label');
      if (!t) return;
      const lb = el('span', 'dk-cmp-label ' + cls);
      lb.textContent = t;
      lb.setAttribute('data-dk-gen', '');
      box.appendChild(lb);
    });
    const set = () => box.style.setProperty('--dk-cmp', inp.value + '%');
    inp.addEventListener('input', set);
    set();
    if (!box._dkCmp) { box._dkCmp = true; guard(box, '.dk-cmp-range'); }
  }

  // ── Hotspots: numbered dots on an image ──────────────────────────────────
  // <div data-hotspots><img src="map.png">
  //   <p data-hotspot="30%,40%">Explanation shown when dot 1 is clicked</p></div>
  function setupHotspots(box) {
    const tips = Array.from(box.querySelectorAll('[data-hotspot]')).filter((t) => (t.parentElement.closest('[data-hotspots]') || t.parentElement) === box);
    if (!tips.length || box.querySelector(':scope > .dk-hs-dot')) return;
    box.classList.add('dk-hs');
    const pal = palette(box);
    box.style.setProperty('--dk-hs-bg', pal.accents[0]);
    box.style.setProperty('--dk-hs-fg', pal.bg);
    box.style.setProperty('--dk-tip-bg', pal.bg);
    box.style.setProperty('--dk-tip-fg', pal.ink);
    const dots = [];
    const setOpen = (list) => tips.forEach((t, i) => {
      const on = list.includes(i);
      t.classList.toggle('dk-open', on);
      if (dots[i]) dots[i].setAttribute('aria-expanded', String(on));
    });
    const openList = () => tips.map((t, i) => (t.classList.contains('dk-open') ? i : -1)).filter((i) => i >= 0);
    // Re-run on the same box (after an edit) reuses its sync slot.
    const actor = box._dkActor || (box._dkActor = addActor('hotspot', box, openList, setOpen));
    actor.state = openList;
    actor.apply = setOpen;
    tips.forEach((t, i) => {
      const [x, y] = (t.getAttribute('data-hotspot') || '').split(/[\s,;]+/).map(parseFloat);
      if (!isFinite(x) || !isFinite(y)) { console.warn('[deck-kit] data-hotspot needs "x%,y%":', t); return; }
      t.classList.add('dk-hs-tip');
      if (x > 60) t.classList.add('dk-hs-left');
      t.style.setProperty('--dk-x', x + '%');
      t.style.setProperty('--dk-y', y + '%');
      const d = el('button', 'dk-hs-dot');
      d.type = 'button';
      d.textContent = t.getAttribute('data-label') || String(i + 1);
      d.setAttribute('data-dk-gen', '');
      d.setAttribute('aria-expanded', 'false');
      d.setAttribute('aria-label', srcText(t).trim().slice(0, 80));
      d.style.setProperty('--dk-x', x + '%');
      d.style.setProperty('--dk-y', y + '%');
      d.addEventListener('click', () => {
        const now = openList();
        // One bubble at a time unless the box says data-hotspots="many".
        const many = box.getAttribute('data-hotspots') === 'many';
        setOpen(now.includes(i) ? now.filter((j) => j !== i) : many ? now.concat(i) : [i]);
        sendAct(actor);
      });
      box.appendChild(d);
      dots[i] = d;
    });
    if (!box._dkHs) {
      box._dkHs = true;
      guard(box, '.dk-hs-dot, .dk-hs-tip');
      api.widget(box, { leave: () => box._dkActor.apply([]) });
    }
  }

  // ── QR codes ─────────────────────────────────────────────────────────────
  // <div data-qr="https://example.com"></div>; an empty value means this
  // deck's own address (when it is served over http/https).
  function setupQr(box) {
    if (box.querySelector(':scope > .dk-qr')) return;
    let url = (box.getAttribute('data-qr') || '').trim();
    if (!url) {
      if (!/^https?:$/.test(location.protocol)) { console.warn('[deck-kit] data-qr is empty and the deck is a local file: give it a URL'); return; }
      url = location.href.replace(/[?#].*$/, '');
    } else if (!/^[a-z][\w+.-]*:/i.test(url) && /^https?:$/.test(location.protocol)) {
      try { url = new URL(url, location.href).href; } catch (e) {}
    }
    track(loadLib('qr').then(() => {
      if (box.querySelector(':scope > .dk-qr')) return;
      const qrcode = window.qrcode;
      if (qrcode.stringToBytesFuncs && qrcode.stringToBytesFuncs['UTF-8']) qrcode.stringToBytes = qrcode.stringToBytesFuncs['UTF-8'];
      const q = qrcode(0, box.getAttribute('data-qr-level') || 'M');
      q.addData(url, 'Byte');
      q.make();
      const n = q.getModuleCount();
      const pal = palette(box);
      // Scanners need dark modules on a light ground: dark templates get a white card.
      const dark = pal.dark, pad = dark ? 3 : 1, size = n + pad * 2;
      let d = '';
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (q.isDark(r, c)) d += `M${c + pad} ${r + pad}h1v1h-1z`;
      const svg = `<svg class="dk-qr" data-dk-gen="" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" shape-rendering="crispEdges" role="img" aria-label="${esc(url)}">` +
        (dark ? `<rect width="${size}" height="${size}" rx="1.5" fill="#fff"/>` : '') +
        `<path d="${d}" fill="${dark ? '#000' : pal.ink}"/></svg>`;
      box.insertAdjacentHTML('beforeend', svg);
    })).catch(() => {
      if (box.querySelector(':scope > .dk-qr')) return;
      const f = el('span', 'dk-qr dk-qr-fallback');
      f.textContent = url;
      f.setAttribute('data-dk-gen', '');
      box.appendChild(f);
    });
  }

  // ── Zoom (Z): magnify part of the slide while presenting ─────────────────
  const zoom = { on: false, picking: false, s: null, prev: null, timer: null, restore: null, x0: 0, y0: 0, dragging: false };
  function startZoom() {
    if (zoom.on || zoom.picking) { unzoom(); return; }
    if (cur < 0) return;
    zoom.picking = true;
    show('zoomLayer', true);
    flash(L.zoomHint, 2400);
  }
  function stopZoomPick() {
    zoom.picking = false;
    zoom.dragging = false;
    show('zoomLayer', false);
    ui.zoomBox.style.display = 'none';
  }
  /** Zoom to the box (client px) picked on the current slide, and tell the other window. */
  function zoomTo(x, y, w, h) {
    const s = slides[cur];
    if (!s || zoom.on) return;
    const r = s.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const point = w < 12 && h < 12;
    const f = point ? 2.2 : Math.min(r.width / w, r.height / h);
    // Centre as a fraction of the slide, so both screens zoom the same spot whatever their size.
    const fx = (x + w / 2 - r.left) / r.width, fy = (y + h / 2 - r.top) / r.height;
    zoomCore(s, fx, fy, f);
    route({ dk: 'zoom', slide: cur, fx, fy, f });
  }
  /** Scale slide s by f so the point (fx, fy) — fractions of the slide — lands in its middle. */
  function zoomCore(s, fx, fy, f) {
    f = Math.max(1.25, Math.min(8, f));
    fx = Math.max(0, Math.min(1, fx)); fy = Math.max(0, Math.min(1, fy));
    const cx = fx * s.offsetWidth, cy = fy * s.offsetHeight;
    const mx = s.offsetWidth / 2, my = s.offsetHeight / 2;
    // Scaling by f about origin o moves c to o + f(c − o); choose o so c lands mid-slide.
    const ox = (mx - f * cx) / (1 - f), oy = (my - f * cy) / (1 - f);
    finishUnzoom();
    if (zoom.s && zoom.s !== s) unzoom(true, true);
    if (!zoom.s) {
      zoom.s = s;
      zoom.prev = { scale: s.style.scale, origin: s.style.transformOrigin, transition: s.style.transition, overflow: document.documentElement.style.overflow };
    }
    document.documentElement.style.overflow = 'hidden';
    s.style.transition = reducedMotion() ? 'none' : 'scale .45s cubic-bezier(.2, .7, .2, 1)';
    s.style.transformOrigin = `${ox}px ${oy}px`;
    s.style.scale = String(f);
    zoom.on = true;
  }
  /** Run a zoom-out that is still animating to its end now. */
  function finishUnzoom() {
    clearTimeout(zoom.timer);
    const r = zoom.restore;
    zoom.restore = null;
    if (r) r();
  }
  function unzoom(instant, quiet) {
    stopZoomPick();
    if (!zoom.s) { zoom.on = false; if (instant) finishUnzoom(); return; }
    const s = zoom.s, prev = zoom.prev;
    if (!quiet) route({ dk: 'zoom', slide: slides.indexOf(s), off: true });
    zoom.s = null;
    zoom.on = false;
    zoom.restore = () => {
      s.style.scale = prev.scale; s.style.transformOrigin = prev.origin; s.style.transition = prev.transition;
      document.documentElement.style.overflow = prev.overflow;
      if (!s.getAttribute('style')) s.removeAttribute('style');
    };
    if (instant || reducedMotion()) { finishUnzoom(); return; }
    s.style.scale = '1';
    zoom.timer = setTimeout(finishUnzoom, 470);
  }
  /** A zoom message from the other window. */
  function applyZoom(m) {
    applyingRemote = true;
    try {
      if (m.off) { if (zoom.s && zoom.s === slides[m.slide]) unzoom(); }
      else if (m.slide === cur && slides[cur] && isFinite(m.fx + m.fy + m.f)) zoomCore(slides[cur], m.fx, m.fy, m.f);
    } finally { applyingRemote = false; }
  }
  /** Mirror: a box picked over this preview in the presenter window (viewport fractions). */
  function zoomPick(m) {
    if (!isFinite(m.x + m.y + m.w + m.h)) return;
    zoomTo(m.x * innerWidth, m.y * innerHeight, m.w * innerWidth, m.h * innerHeight);
  }
  function setupZoom() {
    const layer = ui.zoomLayer, box = ui.zoomBox;
    layer.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      layer.setPointerCapture(e.pointerId);
      zoom.dragging = true; zoom.x0 = e.clientX; zoom.y0 = e.clientY;
    });
    layer.addEventListener('pointermove', (e) => {
      if (!zoom.dragging) return;
      const x = Math.min(zoom.x0, e.clientX), y = Math.min(zoom.y0, e.clientY);
      Object.assign(box.style, { display: 'block', left: x + 'px', top: y + 'px', width: Math.abs(e.clientX - zoom.x0) + 'px', height: Math.abs(e.clientY - zoom.y0) + 'px' });
    });
    const end = (e) => {
      if (!zoom.dragging) return;
      const x = Math.min(zoom.x0, e.clientX), y = Math.min(zoom.y0, e.clientY), w = Math.abs(e.clientX - zoom.x0), h = Math.abs(e.clientY - zoom.y0);
      stopZoomPick();
      if (e.type === 'pointerup') zoomTo(x, y, w, h);
    };
    layer.addEventListener('pointerup', end);
    layer.addEventListener('pointercancel', end);
  }

  /** Turn the author's markup into live features, inside `root` (a slide,
   *  an element, or the whole document). Safe to call again on re-rendered
   *  content (after typing or undo in edit mode). */
  function enhance(root = document) {
    const all = (sel) => {
      const list = root.matches && root.matches(sel) ? [root] : [];
      root.querySelectorAll(sel).forEach((e) => list.push(e));
      return list.filter((e) => !e.closest('aside.notes') && !isGen(e) && slideOf(e));
    };
    slides.filter((s) => root === document || root.contains(s) || s.contains(root)).forEach((s) => renderMath(scanMath(s.contains(root) && root !== document ? root : s)));
    all('pre').filter((p) => p.querySelector(':scope > code') || p.hasAttribute('data-lines') || p.hasAttribute('data-copy')).forEach(setupCode);
    all('[data-quiz]').forEach(setupQuiz);
    all('[data-compare]').forEach(setupCompare);
    const hsBoxes = new Set();
    all('[data-hotspot]').forEach((t) => hsBoxes.add(t.parentElement.closest('[data-hotspots]') || t.parentElement));
    if (root.matches && root.matches('[data-hotspots]')) hsBoxes.add(root);
    hsBoxes.forEach(setupHotspots);
    all('[data-qr]').forEach(setupQr);
    all('canvas[data-chart]').forEach((cv) => {
      if (cv._dkChart) return;
      let spec;
      try { spec = JSON.parse(cv.getAttribute('data-chart')); } catch (e) { console.warn('[deck-kit] data-chart is not valid JSON:', cv, e.message); return; }
      cv._dkChart = chart(cv, spec);
    });
    all('table[data-chart]').forEach((t) => { if (!t.nextElementSibling || !t.nextElementSibling.matches('.dk-chart-box')) tableChart(t); });
    all('[data-stepper]').forEach(setupFrames);
  }

  // ── Public API ───────────────────────────────────────────────────────────
  const api = {
    version: '2.0.0',
    role: ROLE,
    get index() { return cur; },
    get length() { return slides.length; },
    get mode() { return adapter ? adapter.mode : null; },
    get slides() { return slides.slice(); },
    goTo: (i) => adapter && goto(i),
    next: () => adapter && advance(1),
    prev: () => adapter && advance(-1),
    /** Steps on the current slide; all(true) shows every step everywhere (export, check). */
    steps: {
      get n() { return stepInfo().n; },
      get total() { return stepInfo().total; },
      set: (n) => { if (cur >= 0) setSteps(cur, n); },
      all: (on = true) => showAllSteps(on),
      get allOn() { return stepsAll; },
    },
    chart: (target, spec) => chart(target, spec),
    stepper: (target, opts) => stepper(target, opts),
    zoom: { out: () => unzoom(), get on() { return zoom.on; } },
    sourceHtml: (node) => sourceHtml(node),
    enhance: (root) => enhance(root || document),
    get busy() { return busy; },
    /** Register an interactive widget for a slide (element, id, or any node inside it). */
    widget(target, hooks) {
      const w = { target, hooks: hooks || {}, slide: null };
      widgets.push(w);
      if (adapter) {
        w.slide = resolveSlide(target);
        if (!w.slide) console.warn('[deck-kit] widget target not inside a slide:', target);
        else if (w.slide === slides[cur]) { call(w, 'enter'); call(w, 'draw'); }
      }
      return w;
    },
    canvas: fitCanvas,
    bind,
    plot,
    palette,
    titleOf: (i) => slides[i] ? titleOf(slides[i], i) : '',
    notesOf: (i) => slides[i] ? notesOf(slides[i], i) : '',
    get plan() { return { start: plan.start.slice(), mins: plan.mins.slice(), total: plan.total }; },
    toggle: (name, on) => ui && show({ notes: 'notes', menu: 'menu', timer: 'tmr', help: 'help' }[name], on),
    presenter: () => openPresenter(),
    timer: {
      start: startTimer, reset: () => timerCmd('reset'), pause: () => timerCmd('pause'),
      get elapsed() { return elapsed(); },
    },
    visitAll,
    settle: settleSlide,
    preparePrint,
    cleanupPrint,
    patchNotes,
    patchEdits,
    patchEditsReport,
    stopMedia: (i) => stopMedia(slides[i == null ? cur : i]),
    edit: { enter: () => enterEdit(), exit: () => exitEdit(), get on() { return ed.on; }, pending: () => pendingEdits() },
    describe: (node) => describe(node),
    /** Visit every slide (so on-entry content renders), then open the print dialog. */
    async print() {
      if (document.body.getAttribute('data-print-visit') !== 'off') {
        await visitAll({ progress: (i, n) => flash(fmt(L.printing, { i, n }), 1500) });
      }
      show('toast', false);
      preparePrint();
      window.print();
    },
  };
  window.DeckKit = api;

  // ── Boot ─────────────────────────────────────────────────────────────────
  function init() {
    adapter = findAdapter();
    slides = adapter.slides();
    if (!slides.length) { console.warn('[deck-kit] no slides found'); return; }
    buildPlan();
    buildUI();
    widgets.forEach((w) => { w.slide = resolveSlide(w.target); });
    window.addEventListener('message', onMessage);

    if (ROLE === 'presenter') {
      const m = (location.hash || '').match(/^#(\d+)$/);
      cur = m ? Math.max(0, Math.min(slides.length - 1, +m[1] - 1)) : 0;
      loadDraft();
      buildPresenter();
      window.addEventListener('keydown', onKey, true);
      requestAnimationFrame(loop);
      return;
    }

    enhance(document);
    slides.forEach((s) => applySteps(s, 0));

    if (ROLE === 'mirror') {
      // Preview inside the presenter window: slides only, no chrome, no input.
      ui.host.style.display = 'none';
      prepareMedia();
      // Previews must stay silent: only the audience screen plays sound.
      const mute = () => document.querySelectorAll('video, audio').forEach((m) => { m.muted = true; });
      mute();
      new MutationObserver(mute).observe(document.body, { childList: true, subtree: true });
      const m = (location.hash || '').match(/^#(\d+)$/);
      pinned = m ? Math.max(0, Math.min(slides.length - 1, +m[1] - 1)) : 0;
      if (!adapter.ownsHash && pinned > 0) goto(pinned);
      document.addEventListener('input', onControl, true);
      document.addEventListener('change', onControl, true);
      document.addEventListener('click', onDocClick);
      window.addEventListener('keydown', onMirrorKey, true);
      requestAnimationFrame(loop);
      return;
    }

    prepareMedia();
    setupTouch();
    setupInk();
    setupZoom();
    window.addEventListener('click', onStepClick, true);
    window.addEventListener('pointerdown', (e) => { lastPress = { el: e.composedPath()[0], at: performance.now() }; }, true);
    window.addEventListener('wheel', onStepWheel, { capture: true, passive: false });
    document.querySelectorAll('[data-countdown]').forEach(cdRender);
    document.addEventListener('click', onDocClick);
    document.addEventListener('click', onLinkClick, true);
    ['pointerdown', 'pointermove', 'click', 'mousedown', 'mouseup', 'touchstart', 'touchend'].forEach((t) => window.addEventListener(t, onEditPointer, { capture: true, passive: false }));
    window.addEventListener('dblclick', onEditDblClick, true);
    window.addEventListener('hashchange', () => { const k = slideForHash(location.hash); if (k >= 0 && k < slides.length && k !== cur) goto(k); });
    window.addEventListener('beforeunload', (e) => { if (ed.on && pendingEdits().length) { e.preventDefault(); e.returnValue = ''; } });
    document.addEventListener('input', onControl, true);
    document.addEventListener('change', onControl, true);
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('resize', onResize);
    window.addEventListener('beforeprint', preparePrint);
    window.addEventListener('afterprint', cleanupPrint);
    window.addEventListener('beforeunload', () => { if (presenterWin && !presenterWin.closed) presenterWin.close(); });
    if (window.matchMedia) {
      const mq = matchMedia('print');
      const h = (m) => (m.matches ? preparePrint() : cleanupPrint());
      if (mq.addEventListener) mq.addEventListener('change', h); else if (mq.addListener) mq.addListener(h);
    }

    // Deep link (#N) for decks that don't restore it themselves.
    // Deep link: #N for decks that don't restore it themselves, #id for any deck.
    {
      const k = slideForHash(location.hash);
      const numeric = /^#\d+$/.test(location.hash);
      if (k > 0 && k < slides.length && (!numeric || !adapter.ownsHash)) goto(k);
    }

    try {
      if (!localStorage.getItem('deck-kit-hint')) { flash(L.helpHint, 3500); localStorage.setItem('deck-kit-hint', '1'); }
      const restored = loadDraft();
      if (restored) { renderNotes(); flash(fmt(L.restored, { n: restored }), 5000); }
      const restoredEdits = PUBLISHED ? 0 : loadEditDraft();
      if (restoredEdits) flash(fmt(L.editRestored, { n: restoredEdits }), 6000);
    } catch (e) {}

    requestAnimationFrame(loop);
    document.dispatchEvent(new CustomEvent('deckkit:ready', { detail: api }));
  }

  function boot() {
    // <deck-stage> collects its slides on slotchange; wait for the upgrade
    // and one more task so index/children are settled.
    if (document.querySelector('deck-stage') && window.customElements) {
      customElements.whenDefined('deck-stage').then(() => setTimeout(init, 0));
    } else {
      setTimeout(init, 0);
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
