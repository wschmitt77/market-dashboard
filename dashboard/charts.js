/* Chart helpers for the Market Signals Dashboard (d3 v7 global). Shared on window.MS. */
(function () {
  const MS = (window.MS = window.MS || {});
  const colors = () => dash.colors || ['#4c6ef5', '#f59f00', '#2f9e44', '#e8590c', '#7048e8', '#0ca678', '#d6336c', '#868e96'];
  MS.colors = colors;
  const parseDate = (s) => (s instanceof Date ? s : new Date(String(s).slice(0, 10) + 'T00:00:00Z'));
  MS.parseDate = parseDate;
  const fmtDate = d3.utcFormat('%b %-d, %Y');
  const fmtDateShort = d3.utcFormat('%b %-d');
  const fmtMonth = d3.utcFormat("%b '%y");
  const num = (v) => (v === null || v === undefined || v === '' ? null : (typeof v === 'number' ? v : Number(v)));
  MS.num = num;
  MS.fmt = {
    pct1: (v) => (v == null || isNaN(v) ? '—' : d3.format('+.1f')(v) + '%'),
    pct1u: (v) => (v == null || isNaN(v) ? '—' : d3.format('.1f')(v) + '%'),
    pct0u: (v) => (v == null || isNaN(v) ? '—' : d3.format('.0f')(v) + '%'),
    f1: (v) => (v == null || isNaN(v) ? '—' : d3.format('.1f')(v)),
    f2: (v) => (v == null || isNaN(v) ? '—' : d3.format('.2f')(v)),
    f3: (v) => (v == null || isNaN(v) ? '—' : d3.format('.3f')(v)),
    f0: (v) => (v == null || isNaN(v) ? '—' : d3.format(',.0f')(v)),
    usd: (v) => (v == null || isNaN(v) ? '—' : '$' + d3.format(',.2f')(v)),
    usd0: (v) => (v == null || isNaN(v) ? '—' : '$' + d3.format(',.0f')(v)),
    pp: (v) => (v == null || isNaN(v) ? '—' : d3.format('+.1f')(v) + ' pp'),
    pp2: (v) => (v == null || isNaN(v) ? '—' : d3.format('+.2f')(v) + ' pp'),
  };

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  MS.clear = clear;

  function failed(el, text) {
    clear(el);
    const d = document.createElement('div');
    d.className = 'ms-fail';
    d.textContent = text || 'Failed to load';
    el.appendChild(d);
  }
  MS.failed = failed;

  /** Measure the widest tick label of a formatter over a scale's ticks. */
  function measure(svg, labels, cls) {
    const g = svg.append('g').attr('class', cls || 'ms-measure');
    let w = 0;
    labels.forEach((t) => { const tx = g.append('text').attr('font-size', 11).text(t); w = Math.max(w, tx.node().getComputedTextLength()); });
    g.remove();
    return w;
  }

  function tooltip(container) {
    let tip = container.querySelector('.ms-tip');
    if (!tip) { tip = document.createElement('div'); tip.className = 'ms-tip'; tip.hidden = true; container.appendChild(tip); }
    return tip;
  }

  function showTip(tip, container, x, y, lines) {
    clear(tip);
    lines.forEach((ln, i) => {
      const row = document.createElement('div');
      if (i === 0) row.className = 'ms-tip-date';
      if (ln.color) { const sw = document.createElement('span'); sw.className = 'ms-sw'; sw.style.background = ln.color; row.appendChild(sw); }
      row.appendChild(document.createTextNode(ln.text));
      tip.appendChild(row);
    });
    tip.hidden = false;
    const cw = container.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 12; if (left + tw > cw - 4) left = Math.max(4, x - tw - 12);
    let top = y - th - 8; if (top < 0) top = y + 12;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }

  /**
   * Multi-series line chart over dated rows.
   * opts: {series:[{key,label,color,dash}], yFormat, refs:[{y,label}], band:[lo,hi], log, height, xKey, zeroLine, areaKey, yDomain}
   */
  MS.lineChart = function (el, rows, opts) {
    clear(el);
    const W = el.clientWidth || 320; if (W < 40) return;
    const H = opts.height || (el.classList.contains('ms-tall') ? 300 : el.classList.contains('ms-short') ? 160 : 220);
    const xKey = opts.xKey || 'date';
    const data = rows.map((r) => ({ d: parseDate(r[xKey]), r })).filter((p) => !isNaN(p.d));
    const series = opts.series.filter((s) => data.some((p) => num(p.r[s.key]) != null));
    if (!series.length) { failed(el, 'No data'); return; }
    const yF = opts.yFormat || MS.fmt.f1;
    const allY = [];
    series.forEach((s) => data.forEach((p) => { const v = num(p.r[s.key]); if (v != null && !isNaN(v) && (!opts.log || v > 0)) allY.push(v); }));
    (opts.refs || []).forEach((r) => allY.push(r.y));
    if (opts.band) allY.push(opts.band[0], opts.band[1]);
    if (opts.zeroLine) allY.push(0);
    let [y0, y1] = opts.yDomain || d3.extent(allY);
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    const pad = (y1 - y0) * 0.06;
    const yDom = opts.log ? [y0 * 0.9, y1 * 1.1] : [y0 - pad, y1 + pad];
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H);
    const yScale0 = (opts.log ? d3.scaleLog() : d3.scaleLinear()).domain(yDom).range([H - 28, 10]);
    const ticks = yScale0.ticks(5);
    const lw = measure(svg, ticks.map(yF)) + 10;
    const endLabels = series.map((s) => { const last = [...data].reverse().find((p) => num(p.r[s.key]) != null); return last ? yF(num(last.r[s.key])) : ''; });
    const rw = series.length > 1 ? 0 : measure(svg, endLabels) + 8;
    const m = { top: 10, right: 8 + rw, bottom: 28, left: lw };
    const x = d3.scaleUtc().domain(d3.extent(data, (p) => p.d)).range([m.left, W - m.right]);
    const y = yScale0.copy().range([H - m.bottom, m.top]);
    const g = svg.append('g');
    // grid + axes
    g.append('g').selectAll('line').data(ticks).join('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', (t) => y(t)).attr('y2', (t) => y(t)).attr('stroke', 'var(--cds-chart-grid)');
    g.append('g').attr('transform', `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(Math.max(2, Math.floor((W - m.left - m.right) / 90))).tickFormat((d) => (x.domain()[1] - x.domain()[0] > 400 * 864e5 ? fmtMonth(d) : fmtDateShort(d))).tickSize(0).tickPadding(8))
      .call((ax) => { ax.select('.domain').attr('stroke', 'var(--cds-chart-axis)'); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 11); ax.attr('font-size', null).attr('font-family', null); });
    g.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(yF).tickSize(0).tickPadding(6))
      .call((ax) => { ax.select('.domain').remove(); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 11); ax.attr('font-size', null).attr('font-family', null); });
    if (opts.band) {
      g.append('rect').attr('x', m.left).attr('width', W - m.left - m.right).attr('y', y(opts.band[1])).attr('height', Math.max(0, y(opts.band[0]) - y(opts.band[1]))).attr('fill', 'var(--cds-chart-grid)').attr('opacity', 0.5);
    }
    (opts.refs || []).forEach((r) => {
      g.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(r.y)).attr('y2', y(r.y)).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
      if (r.label) g.append('text').attr('x', W - m.right).attr('y', y(r.y) - 3).attr('text-anchor', 'end').attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(r.label);
    });
    if (opts.zeroLine) g.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--cds-chart-axis)');
    const line = (key) => d3.line().defined((p) => { const v = num(p.r[key]); return v != null && !isNaN(v) && (!opts.log || v > 0); }).x((p) => x(p.d)).y((p) => y(num(p.r[key])));
    const palette = colors();
    series.forEach((s, i) => {
      const col = s.color || palette[i % palette.length];
      if (s.area) {
        const gid = 'g' + Math.random().toString(36).slice(2, 8);
        const grad = svg.append('defs').append('linearGradient').attr('id', gid).attr('x1', 0).attr('x2', 0).attr('y1', 0).attr('y2', 1);
        [[0, .26], [.25, .16], [.5, .08], [.75, .03], [1, 0]].forEach(([o, a]) => grad.append('stop').attr('offset', o).attr('stop-opacity', a).attr('style', `stop-color:${col}`));
        const area = d3.area().defined((p) => num(p.r[s.key]) != null).x((p) => x(p.d)).y0(y(opts.zeroLine ? 0 : yDom[0])).y1((p) => y(num(p.r[s.key])));
        g.append('path').datum(data).attr('d', area).attr('fill', `url(#${gid})`);
      }
      g.append('path').datum(data).attr('d', line(s.key)).attr('fill', 'none').attr('stroke', col).attr('stroke-width', s.width || 2).attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round').attr('stroke-dasharray', s.dash ? '5 4' : null).attr('opacity', s.muted ? 0.55 : 1);
      if (series.length === 1 && endLabels[i]) {
        const last = [...data].reverse().find((p) => num(p.r[s.key]) != null);
        if (last) g.append('text').attr('x', x(last.d) + 5).attr('y', y(num(last.r[s.key])) + 4).attr('font-size', 11).attr('fill', 'var(--color-fg)').text(endLabels[i]);
      }
    });
    // legend for 2+ series
    if (series.length > 1) {
      const lg = document.createElement('div'); lg.className = 'ms-legend';
      series.forEach((s, i) => { const it = document.createElement('span'); const sw = document.createElement('span'); sw.className = 'ms-sw'; sw.style.background = s.color || palette[i % palette.length]; it.appendChild(sw); it.appendChild(document.createTextNode(s.label || s.key)); lg.appendChild(it); });
      el.insertBefore(lg, el.firstChild);
    }
    // hover
    const tip = tooltip(el);
    const rule = g.append('line').attr('y1', m.top).attr('y2', H - m.bottom).attr('stroke', 'var(--cds-chart-axis)').attr('opacity', 0);
    const dots = series.map((s, i) => g.append('circle').attr('r', 4).attr('fill', s.color || palette[i % palette.length]).attr('stroke', 'var(--color-panel)').attr('stroke-width', 2).attr('opacity', 0));
    const bis = d3.bisector((p) => p.d).center;
    const svgNode = svg.node();
    function move(ev) {
      const [mx, my] = d3.pointer(ev, svgNode);
      const sx = mx * (W / (svgNode.clientWidth || W));
      const i = Math.max(0, Math.min(data.length - 1, bis(data, x.invert(sx))));
      const p = data[i];
      rule.attr('x1', x(p.d)).attr('x2', x(p.d)).attr('opacity', 1);
      const lines = [{ text: fmtDate(p.d) }];
      series.forEach((s, k) => { const v = num(p.r[s.key]); dots[k].attr('cx', x(p.d)).attr('cy', v == null ? -99 : y(v)).attr('opacity', v == null ? 0 : 1); lines.push({ color: series.length > 1 ? (s.color || palette[k % palette.length]) : null, text: (series.length > 1 ? (s.label || s.key) + ' ' : '') + yF(v) }); });
      const rect = el.getBoundingClientRect(); const sr = svgNode.getBoundingClientRect();
      showTip(tip, el, (sr.left - rect.left) + x(p.d) * (svgNode.clientWidth / W), (sr.top - rect.top) + my, lines);
    }
    svg.on('pointermove', move).on('pointerleave', () => { tip.hidden = true; rule.attr('opacity', 0); dots.forEach((d) => d.attr('opacity', 0)); });
  };

  /** Tiny sparkline (last n rows) with the last point marked. */
  MS.spark = function (el, rows, key, n) {
    clear(el);
    const W = el.clientWidth || 160, H = 36;
    const data = rows.slice(-(n || 126)).map((r) => num(r[key])).filter((v) => v != null && !isNaN(v));
    if (data.length < 2) return;
    const x = d3.scaleLinear().domain([0, data.length - 1]).range([2, W - 6]);
    const y = d3.scaleLinear().domain(d3.extent(data)).range([H - 3, 3]);
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H);
    const up = data[data.length - 1] >= data[0];
    const col = up ? 'var(--color-ok)' : 'var(--color-bad)';
    svg.append('path').datum(data).attr('d', d3.line().x((v, i) => x(i)).y((v) => y(v))).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 1.5).attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round');
    svg.append('circle').attr('cx', x(data.length - 1)).attr('cy', y(data[data.length - 1])).attr('r', 2.5).attr('fill', col);
  };

  /** Horizontal 0–100 gauge with marker levels. */
  MS.gauge = function (el, value, markers, cls) {
    clear(el);
    const fill = document.createElement('div'); fill.className = 'ms-fill ' + (cls || '');
    fill.style.width = (value == null ? 0 : Math.max(0, Math.min(100, value))) + '%';
    el.appendChild(fill);
    (markers || []).forEach((mk, i) => { const m = document.createElement('div'); m.className = 'ms-mark'; m.style.left = mk + '%'; const s = document.createElement('span'); s.textContent = String(mk); m.appendChild(s); el.appendChild(m); });
  };

  /**
   * Grouped bars by category with optional per-bar labels and reference lines.
   * items: [{cat, groups:[{name, value, label, color}]}], opts: {yFormat, refs:[{y,label,color}], height}
   */
  MS.groupedBars = function (el, items, opts) {
    clear(el);
    const W = el.clientWidth || 320; if (W < 40) return;
    const H = opts.height || (el.classList.contains('ms-tall') ? 300 : 220);
    const yF = opts.yFormat || MS.fmt.f1;
    const groupNames = opts.groupNames || [...new Set(items.flatMap((it) => it.groups.map((g) => g.name)))];
    const vals = items.flatMap((it) => it.groups.map((g) => g.value)).filter((v) => v != null).concat((opts.refs || []).map((r) => r.y), [0]);
    let [y0, y1] = d3.extent(vals); if (y0 === y1) { y0 -= 1; y1 += 1; }
    const pad = (y1 - y0) * 0.15;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H);
    const yTmp = d3.scaleLinear().domain([Math.min(0, y0 - pad), y1 + pad]).range([H - 30, 14]);
    const ticks = yTmp.ticks(5);
    const m = { top: 14, right: 8, bottom: 30, left: measure(svg, ticks.map(yF)) + 10 };
    const x0 = d3.scaleBand().domain(items.map((it) => it.cat)).range([m.left, W - m.right]).paddingInner(0.25).paddingOuter(0.1);
    const x1 = d3.scaleBand().domain(groupNames).range([0, x0.bandwidth()]).padding(0.12);
    const y = yTmp.copy().range([H - m.bottom, m.top]);
    const g = svg.append('g');
    g.append('g').selectAll('line').data(ticks).join('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', (t) => y(t)).attr('y2', (t) => y(t)).attr('stroke', 'var(--cds-chart-grid)');
    g.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(yF).tickSize(0).tickPadding(6)).call((ax) => { ax.select('.domain').remove(); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 11); ax.attr('font-size', null).attr('font-family', null); });
    g.append('g').attr('transform', `translate(0,${H - m.bottom})`).call(d3.axisBottom(x0).tickSize(0).tickPadding(8)).call((ax) => { ax.select('.domain').attr('stroke', 'var(--cds-chart-axis)'); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 11); ax.attr('font-size', null).attr('font-family', null); });
    g.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--cds-chart-axis)');
    const palette = colors();
    const tip = tooltip(el);
    items.forEach((it) => it.groups.forEach((gr) => {
      if (gr.value == null) return;
      const gi = groupNames.indexOf(gr.name);
      const col = gr.color || palette[gi % palette.length];
      const bx = x0(it.cat) + x1(gr.name), bw = Math.min(x1.bandwidth(), 32), off = (x1.bandwidth() - bw) / 2;
      const top = Math.min(y(0), y(gr.value)), hgt = Math.max(1, Math.abs(y(0) - y(gr.value)));
      const rect = g.append('rect').attr('x', bx + off).attr('width', bw).attr('y', top).attr('height', hgt).attr('rx', 3).attr('fill', col);
      if (gr.label) g.append('text').attr('x', bx + off + bw / 2).attr('y', gr.value >= 0 ? top - 3 : top + hgt + 11).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(gr.label);
      rect.on('pointermove', (ev) => { const [mx, my] = d3.pointer(ev, el); showTip(tip, el, mx, my, [{ text: `${it.cat} · ${gr.name}` }, { color: col, text: yF(gr.value) + (gr.extra ? ' · ' + gr.extra : '') }]); }).on('pointerleave', () => { tip.hidden = true; });
    }));
    (opts.refs || []).forEach((r, i) => {
      g.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(r.y)).attr('y2', y(r.y)).attr('stroke', r.color || 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
      if (r.label) g.append('text').attr('x', W - m.right).attr('y', y(r.y) - 3).attr('text-anchor', 'end').attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(r.label);
    });
    if (groupNames.length > 1) {
      const lg = document.createElement('div'); lg.className = 'ms-legend';
      groupNames.forEach((n, i) => { const it = document.createElement('span'); const sw = document.createElement('span'); sw.className = 'ms-sw'; sw.style.background = palette[i % palette.length]; it.appendChild(sw); it.appendChild(document.createTextNode(n)); lg.appendChild(it); });
      el.insertBefore(lg, el.firstChild);
    }
  };

  /** Horizontal bars for one value per category (sorted), diverging around 0. */
  MS.hBars = function (el, items, opts) {
    clear(el);
    const W = el.clientWidth || 320; if (W < 40) return;
    const rowH = 24, m = { top: 6, right: 48, bottom: 24, left: 8 };
    const labelW = Math.max(...items.map((it) => it.label.length)) * 6.5 + 10;
    const H = m.top + m.bottom + items.length * rowH;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H);
    const vals = items.map((it) => it.value).filter((v) => v != null).concat([0]);
    const ext = d3.extent(vals); const span = Math.max(Math.abs(ext[0]), Math.abs(ext[1])) || 1;
    const x = d3.scaleLinear().domain([-span, span]).range([m.left + labelW, W - m.right]);
    const y = d3.scaleBand().domain(items.map((it) => it.label)).range([m.top, H - m.bottom]).padding(0.25);
    const g = svg.append('g');
    g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', m.top).attr('y2', H - m.bottom).attr('stroke', 'var(--cds-chart-axis)');
    const yF = opts.yFormat || MS.fmt.pp;
    items.forEach((it) => {
      g.append('text').attr('x', m.left).attr('y', y(it.label) + y.bandwidth() / 2 + 4).attr('font-size', 11).attr('fill', 'var(--color-fg)').text(it.label);
      if (it.value == null) return;
      const w = Math.abs(x(it.value) - x(0));
      g.append('rect').attr('x', Math.min(x(0), x(it.value))).attr('y', y(it.label)).attr('width', Math.max(1, w)).attr('height', y.bandwidth()).attr('rx', 3).attr('fill', it.value >= 0 ? 'var(--color-ok)' : 'var(--color-bad)').attr('opacity', 0.85);
      g.append('text').attr('x', it.value >= 0 ? x(it.value) + 4 : x(it.value) - 4).attr('text-anchor', it.value >= 0 ? 'start' : 'end').attr('y', y(it.label) + y.bandwidth() / 2 + 4).attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text(yF(it.value));
    });
    g.append('g').attr('transform', `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(yF).tickSize(0).tickPadding(6)).call((ax) => { ax.select('.domain').remove(); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 10); ax.attr('font-size', null).attr('font-family', null); });
  };

  /** Labeled scatter with quadrants (rotation map). points: [{label, x, y, sub}] */
  MS.scatter = function (el, points, opts) {
    clear(el);
    const W = el.clientWidth || 320; if (W < 40) return;
    const H = opts.height || 300;
    const m = { top: 16, right: 16, bottom: 34, left: 44 };
    const xs = points.map((p) => p.x).filter((v) => v != null), ys = points.map((p) => p.y).filter((v) => v != null);
    const xr = Math.max(Math.abs(d3.min(xs)), Math.abs(d3.max(xs)), 1) * 1.25, yr = Math.max(Math.abs(d3.min(ys)), Math.abs(d3.max(ys)), 1) * 1.25;
    const x = d3.scaleLinear().domain([-xr, xr]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([-yr, yr]).range([H - m.bottom, m.top]);
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H);
    const g = svg.append('g');
    g.append('rect').attr('x', x(0)).attr('y', m.top).attr('width', W - m.right - x(0)).attr('height', y(0) - m.top).attr('fill', 'var(--color-ok)').attr('opacity', 0.06);
    g.append('rect').attr('x', m.left).attr('y', y(0)).attr('width', x(0) - m.left).attr('height', H - m.bottom - y(0)).attr('fill', 'var(--color-bad)').attr('opacity', 0.06);
    g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', m.top).attr('y2', H - m.bottom).attr('stroke', 'var(--cds-chart-axis)');
    g.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--cds-chart-axis)');
    [['leading', W - m.right - 4, m.top + 12, 'end'], ['improving', m.left + 4, m.top + 12, 'start'], ['lagging', m.left + 4, H - m.bottom - 6, 'start'], ['weakening', W - m.right - 4, H - m.bottom - 6, 'end']].forEach(([t, px, py, a]) => g.append('text').attr('x', px).attr('y', py).attr('text-anchor', a).attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(t));
    g.append('g').attr('transform', `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(MS.fmt.f1).tickSize(0).tickPadding(6)).call((ax) => { ax.select('.domain').remove(); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 10); ax.attr('font-size', null).attr('font-family', null); });
    g.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(MS.fmt.f1).tickSize(0).tickPadding(6)).call((ax) => { ax.select('.domain').remove(); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 10); ax.attr('font-size', null).attr('font-family', null); });
    g.append('text').attr('x', (m.left + W - m.right) / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(opts.xLabel || '');
    g.append('text').attr('transform', `translate(12,${(m.top + H - m.bottom) / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(opts.yLabel || '');
    const tip = tooltip(el);
    const palette = colors();
    points.forEach((p, i) => {
      if (p.x == null || p.y == null) return;
      const col = palette[i % palette.length];
      g.append('circle').attr('cx', x(p.x)).attr('cy', y(p.y)).attr('r', 7).attr('fill', col).attr('stroke', 'var(--color-panel)').attr('stroke-width', 2)
        .on('pointermove', (ev) => { const [mx, my] = d3.pointer(ev, el); showTip(tip, el, mx, my, [{ text: p.label + (p.sub ? ' · ' + p.sub : '') }, { color: col, text: `ratio ${MS.fmt.f1(p.x)} · momentum ${MS.fmt.f1(p.y)}` }]); }).on('pointerleave', () => { tip.hidden = true; });
      g.append('text').attr('x', x(p.x) + 9).attr('y', y(p.y) + 4).attr('font-size', 11).attr('fill', 'var(--color-fg)').text(p.label);
    });
  };

  /** Dot plot: x numeric vs y numeric, with labels on hover. */
  MS.dots = function (el, points, opts) {
    clear(el);
    const W = el.clientWidth || 320; if (W < 40) return;
    const H = opts.height || 220;
    const m = { top: 12, right: 12, bottom: 32, left: 44 };
    const pts = points.filter((p) => p.x != null && p.y != null);
    if (!pts.length) { const d = document.createElement('div'); d.className = 'ms-fail'; d.textContent = opts.empty || 'Not enough resolved days yet'; el.appendChild(d); return; }
    const x = d3.scaleLinear().domain(opts.xDomain || d3.extent(pts, (p) => p.x)).nice().range([m.left, W - m.right]);
    const ye = d3.extent(pts, (p) => p.y); const y = d3.scaleLinear().domain([Math.min(0, ye[0]), Math.max(0, ye[1])]).nice().range([H - m.bottom, m.top]);
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H);
    const g = svg.append('g');
    g.append('g').attr('transform', `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(5).tickSize(0).tickPadding(6)).call((ax) => { ax.select('.domain').attr('stroke', 'var(--cds-chart-axis)'); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 10); ax.attr('font-size', null).attr('font-family', null); });
    g.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(MS.fmt.pct1).tickSize(0).tickPadding(6)).call((ax) => { ax.select('.domain').remove(); ax.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 10); ax.attr('font-size', null).attr('font-family', null); });
    g.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--cds-chart-axis)');
    g.append('text').attr('x', (m.left + W - m.right) / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', 'var(--color-fg-muted)').text(opts.xLabel || '');
    const tip = tooltip(el);
    const palette = colors();
    pts.forEach((p) => {
      const col = p.series === 1 ? palette[1] : palette[0];
      g.append('circle').attr('cx', x(p.x)).attr('cy', y(p.y)).attr('r', 5).attr('fill', col).attr('opacity', 0.8).attr('stroke', 'var(--color-panel)').attr('stroke-width', 1.5)
        .on('pointermove', (ev) => { const [mx, my] = d3.pointer(ev, el); showTip(tip, el, mx, my, [{ text: p.label }, { color: col, text: `${opts.xLabel || 'x'} ${MS.fmt.f0(p.x)} · ${MS.fmt.pct1(p.y)}` }]); }).on('pointerleave', () => { tip.hidden = true; });
    });
    if (opts.legend) {
      const lg = document.createElement('div'); lg.className = 'ms-legend';
      opts.legend.forEach((n, i) => { const it = document.createElement('span'); const sw = document.createElement('span'); sw.className = 'ms-sw'; sw.style.background = palette[i]; it.appendChild(sw); it.appendChild(document.createTextNode(n)); lg.appendChild(it); });
      el.insertBefore(lg, el.firstChild);
    }
  };
})();
