/* Market Signals Dashboard — drawings and interactions. Data comes only from dash.data(); text lives in the HTML. */
(function () {
  const MS = window.MS;
  const state = (window.MS_STATE = window.MS_STATE || { asset: 'QQQ', horizon: 20, sort: { key: 'date', dir: 'desc' } });

  /** Returns rows when a dataset is ok; otherwise renders the right placeholder state and returns null. */
  function rowsOf(id, el, kind) {
    const g = dash.data(id);
    if (g.status === 'ok') { el.classList.remove('dash-skeleton'); return Array.isArray(g.data) ? g.data : (g.data && (g.data.rows || g.data.data || g.data.results)) || []; }
    if (g.status === 'loading') { el.classList.add('dash-skeleton'); if (kind === 'text') el.textContent = '—'; else MS.clear(el); return null; }
    el.classList.remove('dash-skeleton');
    if (kind === 'text') el.textContent = '—';
    else if (kind === 'chart') MS.failed(el, g.status === 'error' || g.status === 'connect' ? 'Failed to load' : 'No data');
    else MS.clear(el);
    return null;
  }

  function textDraw(name, id, field, opt) {
    dash.draw(name, (el) => {
      const rows = rowsOf(id, el, 'text'); if (!rows) return;
      let row = rows[(opt && opt.row) || 0];
      if (opt && opt.where) row = rows.find((r) => Object.entries(opt.where).every(([k, v]) => String(r[k]) === String(v)));
      const v = row ? row[field] : null;
      el.textContent = v == null || v === '' ? '—' : String(v);
      if (opt && opt.sign) { const n = MS.num(row && row[opt.sign]); el.classList.toggle('ms-up', n != null && n > 0); el.classList.toggle('ms-down', n != null && n < 0); }
    });
  }

  // ---------------------------------------------------------------- header + action panel
  ['data_through', 'generated_at', 'feed_run', 'takeaway', 'action', 'regime_label', 'regime_score', 'regime_cap', 'dip_score', 'dip_tranches', 'heat_score', 'trim_tranches',
   'qqq_close', 'ivv_close', 'tqqq_close', 'vix', 'hy'].forEach((n) => {
    const field = { data_through: 'data_through', generated_at: 'generated_at', feed_run: 'feed_run_utc', takeaway: 'takeaway', action: 'action', regime_label: 'regime_label', regime_score: 'regime_score_fmt',
      regime_cap: 'max_tqqq_alloc_pct_fmt', dip_score: 'dip_score_fmt', dip_tranches: 'dip_tranches_fmt', heat_score: 'heat_score_fmt', trim_tranches: 'trim_tranches_fmt',
      qqq_close: 'qqq_close_fmt', ivv_close: 'ivv_close_fmt', tqqq_close: 'tqqq_close_fmt', vix: 'vix_fmt', hy: 'hy_oas_fmt' }[n];
    textDraw(n, 'summary', field);
  });
  textDraw('metrics_live', 'summary', 'metrics_live_fmt');
  [['qqq_1d', 'qqq_chg_1d'], ['qqq_ytd', 'qqq_ytd'], ['ivv_1d', 'ivv_chg_1d'], ['ivv_ytd', 'ivv_ytd'], ['tqqq_1d', 'tqqq_chg_1d'], ['tqqq_ytd', 'tqqq_ytd'], ['vix_1d', 'vix_chg_1d']].forEach(([n, f]) => textDraw(n, 'summary', f + '_fmt', { sign: f }));

  dash.draw('regime_dot', (el) => {
    const rows = rowsOf('summary', el, 'text'); el.textContent = '';
    const light = rows && rows[0] ? String(rows[0].regime_light) : '';
    ['ms-green', 'ms-yellow', 'ms-red'].forEach((c) => el.classList.remove(c));
    if (light) el.classList.add('ms-' + light);
  });
  dash.draw('dip_gauge', (el) => { const rows = rowsOf('summary', el, 'chart'); if (!rows) return; MS.gauge(el, MS.num(rows[0].dip_score), [55, 70, 85], 'ms-dip'); });
  dash.draw('heat_gauge', (el) => { const rows = rowsOf('summary', el, 'chart'); if (!rows) return; MS.gauge(el, MS.num(rows[0].heat_score), [60, 75, 90], 'ms-heat'); });

  function partsTable(name, which, cols) {
    dash.draw(name, (tb) => {
      const rows = rowsOf('composite_parts', tb, 'table'); if (!rows) return;
      MS.clear(tb);
      rows.filter((r) => r.composite === which).forEach((r) => {
        const tr = document.createElement('tr'); tr.dataset.key = r.key;
        cols.forEach(([f, num]) => { const td = document.createElement('td'); if (num) td.className = 'num'; td.textContent = r[f] == null || r[f] === '' ? '—' : String(r[f]); tr.appendChild(td); });
        tb.appendChild(tr);
      });
    });
  }
  partsTable('regime_parts', 'regime', [['component'], ['value'], ['contribution_fmt', 1]]);
  partsTable('dip_parts', 'dip', [['component'], ['value', 1], ['weight', 1], ['contribution_fmt', 1]]);
  partsTable('heat_parts', 'heat', [['component'], ['value', 1], ['weight', 1], ['contribution_fmt', 1]]);

  ['qqq', 'ivv', 'tqqq', 'vix'].forEach((k) => dash.draw('spark_' + k, (el) => { const rows = rowsOf('series', el, 'chart'); if (!rows) return; MS.spark(el, rows, k, 126); }));

  // ---------------------------------------------------------------- oversold / overbought gauges (QQQ, IVV, Dow)
  function gaugeTile(name) {
    dash.draw(name, (el) => {
      const rows = rowsOf('gauges', el, 'chart'); if (!rows) return;
      MS.clear(el);
      const r = rows.find((x) => String(x.key) === el.dataset.key);
      const head = document.createElement('div'); head.className = 'ms-ghead';
      const lab = document.createElement('div'); lab.className = 'ms-label'; lab.textContent = r ? `${r.ticker} · ${r.name || ''}` : el.dataset.key;
      const lvl = document.createElement('div'); lvl.className = 'ms-glevel'; lvl.textContent = r ? `${r.close_fmt || '—'} · ${r.chg_1d_fmt || '—'}` : '—';
      head.appendChild(lab); head.appendChild(lvl); el.appendChild(head);
      const status = document.createElement('div'); status.className = 'ms-gstatus ' + (r && r.status_class ? r.status_class : ''); status.textContent = r ? (r.status || '—') : 'No data';
      el.appendChild(status);
      const score = r ? MS.num(r.score) : null;
      const bar = document.createElement('div'); bar.className = 'ms-gbar'; bar.setAttribute('role', 'img'); bar.setAttribute('aria-label', score == null ? 'no score' : `stretch score ${Math.round(score)} of 100`);
      if (score != null) { const needle = document.createElement('div'); needle.className = 'ms-gneedle'; needle.style.left = Math.max(0, Math.min(100, score)) + '%'; bar.appendChild(needle); }
      const scale = document.createElement('div'); scale.className = 'ms-gscale'; ['oversold', 'neutral', 'overbought'].forEach((t) => { const sp = document.createElement('span'); sp.textContent = t; scale.appendChild(sp); }); bar.appendChild(scale);
      el.appendChild(bar);
      const inputs = document.createElement('div'); inputs.className = 'ms-ginputs'; inputs.textContent = r ? ((score == null ? '' : `Score ${Math.round(score)} of 100 · `) + (r.inputs_fmt || '')) : '';
      el.appendChild(inputs);
    });
  }
  ['gauge_qqq', 'gauge_ivv', 'gauge_dow'].forEach(gaugeTile);

  // ---------------------------------------------------------------- recommended allocation (plan panel)
  textDraw('plan_headline', 'plan', 'value', { where: { key: 'headline' } });
  textDraw('plan_headline_detail', 'plan', 'detail', { where: { key: 'headline' } });
  textDraw('plan_asof', 'plan', 'as_of', { where: { key: 'headline' } });
  const tradeCls = (s, role) => (role === 'reserve' ? '' : /^buy/.test(s || '') ? 'ok' : /^sell/.test(s || '') ? 'bad' : '');
  dash.draw('alloc_table', (tb) => {
    const rows = rowsOf('allocation', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    rows.forEach((r) => {
      const tr = document.createElement('tr'); tr.dataset.key = r.ticker;
      [['ticker', 0, 'Fund', 'ms-nolabel'], ['role', 0, 'Role'], ['target_fmt', 1, 'Target'], ['target_pct_fmt', 1, 'Share'], ['held_fmt', 1, 'Held'], ['trade_fmt', 0, 'Trade'], ['note', 0, 'Note', 'ms-nolabel']].forEach(([f, num, label, cls]) => {
        const td = document.createElement('td'); td.dataset.label = label; if (num) td.className = 'num'; if (cls) td.classList.add(cls);
        if (f === 'ticker') { td.appendChild(document.createTextNode(String(r.ticker))); const nm = document.createElement('span'); nm.className = 'ms-fund-name'; nm.textContent = r.name || ''; td.appendChild(nm); }
        else if (f === 'trade_fmt') { const chip = document.createElement('span'); chip.className = 'ms-chip ' + tradeCls(r.trade_fmt, r.role); chip.textContent = r.trade_fmt == null || r.trade_fmt === '' ? '—' : String(r.trade_fmt); td.appendChild(chip); }
        else td.textContent = r[f] == null || r[f] === '' ? '—' : String(r[f]);
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  });
  dash.draw('alloc_bars', (el) => {
    const rows = rowsOf('allocation', el, 'chart'); if (!rows) return;
    MS.clear(el);
    const palette = MS.colors();
    const W = Math.max(240, el.clientWidth || 300), rowH = 22, gap = 10, labelW = 44, legendH = 18;
    const bars = [['Held', 'held_usd'], ['Target', 'target_usd']];
    const H = bars.length * (rowH + gap) + legendH + 4;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H).attr('role', 'img').attr('aria-label', 'Held versus target allocation by fund');
    bars.forEach(([label, field], bi) => {
      const total = d3.sum(rows, (r) => Math.max(0, MS.num(r[field]) || 0)) || 1;
      const y = legendH + bi * (rowH + gap);
      svg.append('text').attr('x', 0).attr('y', y + rowH / 2).attr('dy', '0.35em').attr('fill', 'var(--color-fg-muted)').attr('font-size', 11).text(label);
      let x = labelW;
      rows.forEach((r, i) => {
        const v = Math.max(0, MS.num(r[field]) || 0); if (!v) return;
        const w = (v / total) * (W - labelW);
        svg.append('rect').attr('x', x).attr('y', y).attr('width', Math.max(0, w - 1)).attr('height', rowH).attr('fill', palette[i % palette.length]).attr('rx', 2)
          .append('title').text(`${r.ticker} ${label.toLowerCase()}: ${r[field === 'held_usd' ? 'held_fmt' : 'target_fmt']} (${Math.round((v / total) * 100)}%)`);
        if (w > 34) svg.append('text').attr('x', x + w / 2).attr('y', y + rowH / 2).attr('dy', '0.35em').attr('text-anchor', 'middle').attr('fill', '#fff').attr('font-size', 11).text(r.ticker);
        x += w;
      });
    });
    let lx = labelW;
    rows.forEach((r, i) => {
      const g = svg.append('g').attr('transform', `translate(${lx},0)`);
      g.append('rect').attr('x', 0).attr('y', 3).attr('width', 9).attr('height', 9).attr('rx', 2).attr('fill', palette[i % palette.length]);
      g.append('text').attr('x', 12).attr('y', 8).attr('dy', '0.35em').attr('fill', 'var(--color-fg-muted)').attr('font-size', 10).text(r.ticker);
      lx += 14 + 7 * String(r.ticker).length + 10;
    });
  });
  dash.draw('plan_rules', (ul) => {
    const rows = rowsOf('plan', ul, 'table'); if (!rows) return;
    MS.clear(ul);
    rows.filter((r) => !['headline', 'backtest', 'holdings'].includes(r.key)).forEach((r) => {
      const li = document.createElement('li');
      const dot = document.createElement('span'); dot.className = 'ms-rdot ' + (r.state || '');
      const body = document.createElement('span'); const b = document.createElement('b'); b.textContent = (r.label || '') + ': '; body.appendChild(b); body.appendChild(document.createTextNode(r.value || ''));
      li.appendChild(dot); li.appendChild(body); ul.appendChild(li);
    });
  });
  dash.draw('plan_table', (tb) => {
    const rows = rowsOf('plan', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    rows.forEach((r) => {
      const tr = document.createElement('tr'); tr.dataset.key = r.key;
      [['label', 'Rule', 'ms-nolabel'], ['value', 'Where it stands'], ['detail', 'Detail']].forEach(([f, label, cls]) => { const td = document.createElement('td'); td.dataset.label = label; if (cls) td.className = cls; td.textContent = r[f] == null || r[f] === '' ? '—' : String(r[f]); tr.appendChild(td); });
      tb.appendChild(tr);
    });
  });

  // ---------------------------------------------------------------- insights + scorecard strip
  ['sc_hit:hit_rate_fmt', 'sc_resolved:n_resolved', 'sc_pending:n_pending', 'sc_brier:brier_fmt'].forEach((s) => { const [n, f] = s.split(':'); textDraw(n, 'scorecard', f, { where: { group: 'All calls' } }); });

  const CAT_LABEL = { regime: 'Regime', dip_buy: 'Dip buy', heat: 'Heat', sector: 'Sectors', volatility: 'Volatility', breadth: 'Breadth', credit_rates: 'Credit & rates', macro: 'Macro', calendar: 'Calendar', risk: 'Risk' };
  dash.draw('insights_list', (ol) => {
    const rows = rowsOf('insights', ol, 'table'); if (!rows) return;
    MS.clear(ol);
    [...rows].sort((a, b) => MS.num(a.rank) - MS.num(b.rank)).forEach((r) => {
      const li = document.createElement('li');
      const rank = document.createElement('div'); rank.className = 'ms-rank'; rank.textContent = String(r.rank);
      const body = document.createElement('div');
      const head = document.createElement('div'); head.className = 'ms-head';
      const title = document.createElement('span'); title.className = 'ms-title'; title.textContent = r.title || '';
      const cat = document.createElement('span'); cat.className = 'ms-chip'; cat.textContent = r.category_label || CAT_LABEL[r.category] || r.category || '';
      head.appendChild(title); head.appendChild(cat);
      if (r.testable === 'yes') { const pr = document.createElement('span'); pr.className = 'ms-chip ok'; pr.textContent = (r.prediction_fmt || '') + (r.confidence_fmt ? ' · ' + r.confidence_fmt : ''); head.appendChild(pr); }
      const text = document.createElement('p'); text.className = 'ms-text'; text.textContent = r.text || '';
      const act = document.createElement('div'); act.className = 'ms-act'; const b = document.createElement('b'); b.textContent = 'Action: '; act.appendChild(b); act.appendChild(document.createTextNode(r.action || ''));
      body.appendChild(head); body.appendChild(text); body.appendChild(act);
      li.appendChild(rank); li.appendChild(body); ol.appendChild(li);
    });
  });

  // ---------------------------------------------------------------- metric tables
  const stateCls = (s) => (s === 'ok' ? 'ok' : s === 'warn' ? 'warn' : s === 'bad' ? 'bad' : '');
  function metricsTable(name) {
    dash.draw(name, (tb) => {
      const rows = rowsOf('metrics', tb, 'table'); if (!rows) return;
      const layer = tb.dataset.layer;
      MS.clear(tb);
      rows.filter((r) => r.layer_key === layer).sort((a, b) => MS.num(a.n) - MS.num(b.n)).forEach((r) => {
        const tr = document.createElement('tr'); tr.dataset.key = r.id;
        const cells = [['n', String(r.n), '#', 'ms-nolabel'], ['name', r.name, 'Metric', 'ms-nolabel'], ['display', r.display || '—', 'Value'], ['signal', null, 'Signal'], ['pctile_1y', null, '1-yr pctile'], ['reading', r.reading || '', 'Reading', 'ms-nolabel']];
        cells.forEach(([f, txt, label, cls]) => {
          const td = document.createElement('td'); td.dataset.label = label; if (cls) td.className = cls;
          if (f === 'signal') { const chip = document.createElement('span'); chip.className = 'ms-chip ' + stateCls(r.state); chip.textContent = r.signal || ''; td.appendChild(chip); }
          else if (f === 'pctile_1y') {
            const v = MS.num(r.pctile_1y);
            if (v == null) td.textContent = '—';
            else { const bar = document.createElement('span'); bar.className = 'ms-bar'; bar.style.width = Math.max(2, Math.round(v * 0.6)) + 'px'; td.appendChild(bar); td.appendChild(document.createTextNode(String(r.pctile_1y))); td.classList.add('ms-pct'); }
          } else td.textContent = txt;
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
    });
  }
  ['trend', 'meanrev', 'vol', 'breadth', 'sectors', 'sentiment', 'credit', 'macro'].forEach((l) => metricsTable('metrics_' + l));

  // ---------------------------------------------------------------- charts over the daily series
  function seriesChart(name, build) {
    dash.draw(name, (el) => { const rows = rowsOf('series', el, 'chart'); if (!rows) return; build(el, rows); });
  }
  seriesChart('price_chart', (el, rows) => {
    const a = state.asset.toLowerCase();
    MS.lineChart(el, rows, { series: [{ key: a, label: state.asset }, { key: a + '_sma50', label: '50-day', dash: true, width: 1.5 }, { key: a + '_sma200', label: '200-day', width: 1.5 }], yFormat: MS.fmt.f0 });
  });
  seriesChart('dd_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'qqq_dd', label: 'QQQ' }, { key: 'ivv_dd', label: 'IVV' }, { key: 'tqqq_dd', label: 'TQQQ', muted: true }], yFormat: MS.fmt.pct1u, refs: [{ y: -5 }, { y: -10 }, { y: -15 }, { y: -20, label: 'tranche levels' }], zeroLine: true }));
  seriesChart('rsi_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'qqq_rsi14', label: 'QQQ' }, { key: 'ivv_rsi14', label: 'IVV' }], yFormat: MS.fmt.f0, band: [30, 70], yDomain: [0, 100] }));
  seriesChart('pctb_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'qqq_pctb', label: '%B (0–1 = inside bands)' }], yFormat: MS.fmt.f2, band: [0, 1] }));
  seriesChart('vix_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'vix', label: 'VIX' }, { key: 'vxn', label: 'VXN' }], yFormat: MS.fmt.f0, refs: [{ y: 14, label: 'complacent' }, { y: 30, label: 'dip zone' }] }));
  seriesChart('vixts_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'vix_ratio', label: 'VIX/VIX3M', area: true }], yFormat: MS.fmt.f2, refs: [{ y: 1, label: 'backwardation above' }] }));
  seriesChart('rv_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'qqq_rv20', label: 'Realized vol 20d (%)' }, { key: 'vol_drag', label: '3× drag (%/yr)' }], yFormat: MS.fmt.f0 }));
  seriesChart('gap_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'tqqq_gap63', label: 'TQQQ − 3×QQQ, 63d', area: true }], yFormat: MS.fmt.pp, zeroLine: true }));
  seriesChart('breadth_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'spx_pct200', label: 'S&P 500' }, { key: 'ndx_pct200', label: 'Nasdaq-100' }], yFormat: MS.fmt.pct0u, refs: [{ y: 20, label: 'washout' }, { y: 80, label: 'frothy' }], yDomain: [0, 100] }));
  seriesChart('mcc_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'mcclellan', label: 'McClellan', area: true }], yFormat: MS.fmt.f0, band: [-70, 70], zeroLine: true }));
  seriesChart('nhnl_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'spx_nhnl', label: 'S&P 500' }, { key: 'ndx_nhnl', label: 'Nasdaq-100' }], yFormat: MS.fmt.f0, zeroLine: true }));
  const indexed = (rows, keys) => { const base = {}; keys.forEach((k) => { const f = rows.find((r) => MS.num(r[k]) != null); base[k] = f ? MS.num(f[k]) : null; }); return rows.map((r) => { const o = { date: r.date }; keys.forEach((k) => { const v = MS.num(r[k]); o[k] = v == null || base[k] == null ? null : (v / base[k]) * 100; }); return o; }); };
  seriesChart('eq_chart', (el, rows) => MS.lineChart(el, indexed(rows, ['rsp_ivv', 'qqqe_qqq']), { series: [{ key: 'rsp_ivv', label: 'RSP / IVV' }, { key: 'qqqe_qqq', label: 'QQQE / QQQ' }], yFormat: MS.fmt.f0, refs: [{ y: 100 }] }));
  seriesChart('ratio_chart', (el, rows) => MS.lineChart(el, indexed(rows, ['xly_xlp', 'sphb_splv', 'smh_qqq']), { series: [{ key: 'xly_xlp', label: 'XLY / XLP' }, { key: 'sphb_splv', label: 'SPHB / SPLV' }, { key: 'smh_qqq', label: 'SMH / QQQ' }], yFormat: MS.fmt.f0, refs: [{ y: 100 }] }));
  seriesChart('qqqivv_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'qqq_ivv', label: 'QQQ / IVV' }], yFormat: MS.fmt.f3 }));
  seriesChart('pc_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'put_call10', label: 'Equity put/call, 10d' }], yFormat: MS.fmt.f2, refs: [{ y: 0.55, label: 'complacent' }, { y: 0.9, label: 'fear' }] }));
  seriesChart('credit_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'hy_oas', label: 'High yield' }, { key: 'ig_oas', label: 'Investment grade' }], yFormat: MS.fmt.f2, refs: [{ y: 5, label: 'stress' }] }));
  seriesChart('curve_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 't10y3m', label: '10y − 3m' }, { key: 't10y2y', label: '10y − 2y' }], yFormat: MS.fmt.f2, zeroLine: true }));
  seriesChart('real_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'real10', label: '10y real yield', area: true }], yFormat: MS.fmt.f2 }));
  seriesChart('liq_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'net_liquidity', label: 'Net liquidity ($T)', area: true }], yFormat: MS.fmt.f2 }));
  seriesChart('gas_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'gas_weekly', label: 'Regular gas' }], yFormat: MS.fmt.usd }));
  seriesChart('wti_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'wti', label: 'WTI' }], yFormat: MS.fmt.usd0 }));
  seriesChart('claims_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'claims4', label: 'Claims 4-wk avg (k)' }], yFormat: MS.fmt.f0 }));
  seriesChart('infl_chart', (el, rows) => MS.lineChart(el, rows, { series: [{ key: 'breakeven10', label: '10y breakeven' }, { key: 'fwd5y5y', label: '5y5y forward' }], yFormat: MS.fmt.f2 }));

  // ---------------------------------------------------------------- sectors
  dash.draw('rrg_chart', (el) => {
    const rows = rowsOf('sectors', el, 'chart'); if (!rows) return;
    MS.scatter(el, rows.map((r) => ({ label: r.ticker, sub: r.sector, x: MS.num(r.rs_ratio), y: MS.num(r.rs_momentum) })), { xLabel: 'relative strength vs IVV (ratio vs 6-month average, %)', yLabel: '10-day momentum of the ratio (%)', height: el.clientWidth < 420 ? 280 : 320 });
  });
  dash.draw('sector_bars', (el) => {
    const rows = rowsOf('sectors', el, 'chart'); if (!rows) return;
    MS.hBars(el, [...rows].sort((a, b) => MS.num(b.rel_3m) - MS.num(a.rel_3m)).map((r) => ({ label: `${r.ticker} ${r.sector}`, value: MS.num(r.rel_3m) })), { yFormat: MS.fmt.pp });
  });
  dash.draw('sectors_table', (tb) => {
    const rows = rowsOf('sectors', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    [...rows].sort((a, b) => MS.num(b.rel_3m) - MS.num(a.rel_3m)).forEach((r) => {
      const tr = document.createElement('tr'); tr.dataset.key = r.ticker;
      [['ticker'], ['sector'], ['rel_1m_fmt', 1], ['rel_3m_fmt', 1], ['rel_6m_fmt', 1], ['z_1y_fmt', 1], ['quadrant'], ['dist_200d_fmt', 1]].forEach(([f, num]) => {
        const td = document.createElement('td'); if (num) td.className = 'num';
        if (f === 'quadrant') { const chip = document.createElement('span'); chip.className = 'ms-chip ' + ({ leading: 'ok', improving: 'ok', weakening: 'warn', lagging: 'bad' }[r.quadrant] || ''); chip.textContent = r.quadrant || ''; td.appendChild(chip); }
        else { td.textContent = r[f] == null ? '—' : String(r[f]); if (num && /^[+-]/.test(String(r[f]))) td.classList.add(String(r[f]).startsWith('+') ? 'ms-up' : 'ms-down'); }
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  });

  // ---------------------------------------------------------------- events
  dash.draw('events_table', (tb) => {
    const rows = rowsOf('events', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    if (!rows.length) return;
    rows.forEach((r) => { const tr = document.createElement('tr'); tr.dataset.key = encodeURIComponent(r.event) + ',' + encodeURIComponent(r.date); ['event', 'date', 'days_to_fmt'].forEach((f) => { const td = document.createElement('td'); td.textContent = r[f] == null ? '—' : String(r[f]); tr.appendChild(td); }); tb.appendChild(tr); });
  });

  // ---------------------------------------------------------------- backtests
  function btChart(name, score) {
    dash.draw(name, (el) => {
      const rows = rowsOf('backtest', el, 'chart'); if (!rows) return;
      const h = state.horizon;
      const sub = rows.filter((r) => r.score === score && MS.num(r.horizon) === h);
      const buckets = [...new Set(sub.filter((r) => r.bucket !== 'all').map((r) => r.bucket))];
      const assets = [...new Set(sub.map((r) => r.asset))];
      const items = buckets.map((b) => ({ cat: b, groups: assets.map((a) => { const r = sub.find((x) => x.asset === a && x.bucket === b); return { name: a, value: r ? MS.num(r.mean_ret) : null, label: r && r.hit_rate != null ? r.hit_rate_fmt : '', extra: r ? `n=${r.n}, ${r.hit_rate_fmt} positive` : '' }; }) }));
      const refs = assets.map((a, i) => { const r = sub.find((x) => x.asset === a && x.bucket === 'all'); return r ? { y: MS.num(r.mean_ret), label: `${a} all days`, color: MS.colors()[i] } : null; }).filter(Boolean);
      MS.groupedBars(el, items, { yFormat: (v) => MS.fmt.pct1(v), refs, groupNames: assets });
    });
  }
  btChart('bt_dip_chart', 'dip');
  btChart('bt_heat_chart', 'heat');
  dash.draw('bt_curve_chart', (el) => {
    const rows = rowsOf('backtest_curve', el, 'chart'); if (!rows) return;
    const keys = Object.keys(rows[0] || {}).filter((k) => k !== 'date');
    MS.lineChart(el, rows, { series: keys.map((k) => ({ key: k, label: k })), yFormat: MS.fmt.f0, log: true });
  });
  dash.draw('bt_stats_table', (tb) => {
    const rows = rowsOf('backtest_stats', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    rows.forEach((r) => { const tr = document.createElement('tr'); tr.dataset.key = r.strategy; [['strategy'], ['cagr_fmt', 1], ['max_drawdown_fmt', 1], ['volatility_fmt', 1], ['sharpe_fmt', 1]].forEach(([f, num]) => { const td = document.createElement('td'); if (num) td.className = 'num'; td.textContent = r[f] == null ? '—' : String(r[f]); tr.appendChild(td); }); tb.appendChild(tr); });
  });
  dash.draw('bt_table', (tb) => {
    const rows = rowsOf('backtest', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    rows.forEach((r) => { const tr = document.createElement('tr'); tr.dataset.key = [r.score, r.asset, r.horizon, r.bucket].map((x) => encodeURIComponent(String(x))).join(','); [['score'], ['asset'], ['horizon', 1], ['bucket'], ['n', 1], ['mean_ret_fmt', 1], ['hit_rate_fmt', 1], ['edge_fmt', 1]].forEach(([f, num]) => { const td = document.createElement('td'); if (num) td.className = 'num'; td.textContent = r[f] == null ? '—' : String(r[f]); tr.appendChild(td); }); tb.appendChild(tr); });
  });

  // ---------------------------------------------------------------- scorecard
  dash.draw('sc_table', (tb) => {
    const rows = rowsOf('scorecard', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    rows.forEach((r) => { const tr = document.createElement('tr'); tr.dataset.key = encodeURIComponent(r.group_type) + ',' + encodeURIComponent(r.group); [['group_type', 0, 'Cut'], ['group', 0, 'Group'], ['n_resolved', 1, 'Resolved'], ['n_pending', 1, 'Pending'], ['hit_rate_fmt', 1, 'Hit rate'], ['avg_signed_fmt', 1, 'Avg move'], ['brier_fmt', 1, 'Brier']].forEach(([f, num, label]) => { const td = document.createElement('td'); td.dataset.label = label; if (num) td.className = 'num'; td.textContent = r[f] == null ? '—' : String(r[f]); tr.appendChild(td); }); tb.appendChild(tr); });
  });
  dash.draw('calib_chart', (el) => {
    const rows = rowsOf('score_history', el, 'chart'); if (!rows) return;
    const pts = [];
    rows.forEach((r) => { const y = MS.num(r.qqq_fwd_20d); if (y == null) return; if (MS.num(r.dip_score) != null) pts.push({ x: MS.num(r.dip_score), y, label: `${r.date} · dip ${MS.fmt.f0(MS.num(r.dip_score))}`, series: 0 }); if (MS.num(r.heat_score) != null) pts.push({ x: MS.num(r.heat_score), y, label: `${r.date} · heat ${MS.fmt.f0(MS.num(r.heat_score))}`, series: 1 }); });
    MS.dots(el, pts, { xLabel: 'score', xDomain: [0, 100], legend: ['dip score', 'heat score'], empty: 'No 20-day outcomes yet — the first points appear about four weeks after launch.' });
  });
  dash.draw('log_table', (tb) => {
    const rows = rowsOf('insight_log', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    const { key, dir } = state.sort;
    const sorted = [...rows].sort((a, b) => { let va = a[key], vb = b[key]; const na = MS.num(va), nb = MS.num(vb); let c = (na != null && nb != null && key !== 'date') ? na - nb : String(va ?? '').localeCompare(String(vb ?? '')); if (c === 0) c = MS.num(a.rank) - MS.num(b.rank); return dir === 'asc' ? c : -c; });
    sorted.forEach((r) => {
      const tr = document.createElement('tr'); tr.dataset.key = r.date + ',' + r.rank;
      [['date', 0, 'Date'], ['rank', 1, '#'], ['category_label', 0, 'Category'], ['title', 0, 'Call'], ['prediction_fmt', 0, 'Prediction'], ['confidence_fmt', 1, 'Confidence'], ['status', 0, 'Result'], ['realized_fmt', 1, 'Move'], ['resolves_on', 0, 'Resolves']].forEach(([f, num, label]) => {
        const td = document.createElement('td'); td.dataset.label = label; if (num) td.className = 'num';
        if (f === 'status') { const chip = document.createElement('span'); chip.className = 'ms-chip ' + ({ hit: 'ok', miss: 'bad', pending: 'warn' }[r.status] || ''); chip.textContent = r.status == null ? '' : String(r.status); td.appendChild(chip); }
        else if (f === 'category_label') td.textContent = String(r.category_label ?? r.category ?? '');
        else td.textContent = r[f] == null || r[f] === '' ? '—' : String(r[f]);
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  });

  // ---------------------------------------------------------------- sources
  dash.draw('sources_table', (tb) => {
    const rows = rowsOf('sources', tb, 'table'); if (!rows) return;
    MS.clear(tb);
    rows.forEach((r) => { const tr = document.createElement('tr'); tr.dataset.key = r.source; [['source', 'Source'], ['status', 'Status'], ['last_date', 'Last data'], ['fetched_at', 'Fetched'], ['note', 'Note']].forEach(([f, label]) => { const td = document.createElement('td'); td.dataset.label = label; if (f === 'status') { const chip = document.createElement('span'); chip.className = 'ms-chip ' + (r.status === 'ok' ? 'ok' : r.status === 'kept_previous' || r.status === 'partial' ? 'warn' : 'bad'); chip.textContent = String(r.status ?? ''); td.appendChild(chip); } else td.textContent = r[f] == null || r[f] === '' ? '—' : String(r[f]); tr.appendChild(td); }); tb.appendChild(tr); });
  });

  // ---------------------------------------------------------------- controls (listeners only set attributes; drawings redraw)
  function redrawIn(section) { section.querySelectorAll('[data-draw]').forEach((el) => dash.draw(el.dataset.draw)); }
  function showTab(name, push) {
    document.querySelectorAll('[data-tab-panel]').forEach((sec) => { sec.hidden = sec.dataset.tabPanel !== name; });
    document.querySelectorAll('.ms-tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === name ? 'true' : 'false'));
    const sec = document.querySelector(`[data-tab-panel="${name}"]`);
    if (sec) { redrawIn(sec); if (push) { dash.setLink(name); } }
  }
  document.querySelectorAll('.ms-tabs button').forEach((b) => b.addEventListener('click', () => { showTab(b.dataset.tab, true); const sec = document.querySelector(`[data-tab-panel="${b.dataset.tab}"]`); if (sec && window.innerWidth < 700) sec.scrollIntoView({ block: 'start' }); }));
  document.querySelectorAll('[data-asset]').forEach((b) => b.addEventListener('click', () => { state.asset = b.dataset.asset; document.querySelectorAll('[data-asset]').forEach((x) => x.setAttribute('aria-pressed', x.dataset.asset === state.asset ? 'true' : 'false')); dash.draw('price_chart'); }));
  document.querySelectorAll('[data-horizon]').forEach((b) => b.addEventListener('click', () => { state.horizon = Number(b.dataset.horizon); document.querySelectorAll('[data-horizon]').forEach((x) => x.setAttribute('aria-pressed', Number(x.dataset.horizon) === state.horizon ? 'true' : 'false')); dash.draw('bt_dip_chart'); dash.draw('bt_heat_chart'); }));
  document.querySelectorAll('th button[data-sort]').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.sort; state.sort = { key: k, dir: state.sort.key === k && state.sort.dir === 'desc' ? 'asc' : 'desc' };
    document.querySelectorAll('th button[data-sort]').forEach((x) => x.parentElement.setAttribute('aria-sort', x.dataset.sort === k ? (state.sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'));
    dash.draw('log_table');
  }));
  const initial = (window.location.hash || '').slice(1);
  if (initial && document.querySelector(`[data-tab-panel="${initial}"]`)) showTab(initial, false);
})();
