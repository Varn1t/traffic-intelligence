DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>TraffiQ</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root{
      --bg:#08060e;--surface:#110e1a;--surface2:#181326;--border:#2a1f4a;
      --accent:#a855f7;--accent2:#7c3aed;--green:#22c55e;--red:#ef4444;
      --yellow:#f59e0b;--cyan:#c084fc;--text:#e2e8f0;--muted:#8b7fb0;
      --glow:rgba(168,85,247,0.15);
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

    /* ── ANIMATED BACKGROUND ── */
    body::before{content:'';position:fixed;inset:0;
      background:radial-gradient(ellipse 80% 50% at 50% -20%,#2d1b6955,transparent),
                 radial-gradient(ellipse 50% 40% at 80% 110%,#4c1d9544,transparent),
                 radial-gradient(ellipse 30% 25% at 10% 60%,#7c3aed18,transparent);
      pointer-events:none;z-index:0}

    .wrap{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:28px 24px}

    /* ── HEADER ── */
    header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;flex-wrap:wrap;gap:12px}
    .brand{display:flex;align-items:center;gap:14px}
    .brand-icon{width:46px;height:46px;border-radius:14px;
      background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;font-size:1.4rem;
      box-shadow:0 0 30px #a855f740}
    .brand h1{font-size:1.45rem;font-weight:800;letter-spacing:-.02em;color:#f1f5f9}
    .brand p{font-size:.78rem;color:var(--muted);margin-top:2px}

    .header-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
    .live-badge{display:flex;align-items:center;gap:7px;
      background:#0d1225;border:1px solid var(--border);border-radius:100px;
      padding:6px 14px;font-size:.75rem;font-weight:600;color:var(--green)}
    .live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse-dot 1.4s ease-in-out infinite}
    @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.7)}}
    .badge-paused{border-color:#f59e0b40!important;color:#fcd34d!important}
    .badge-offline{border-color:#ef444440!important;color:#fca5a5!important;animation:pulse-offline 2s ease-in-out infinite}
    @keyframes pulse-offline{0%,100%{box-shadow:0 0 8px #ef444420}50%{box-shadow:0 0 16px #ef444440}}

    .btn{display:flex;align-items:center;gap:6px;padding:8px 16px;border-radius:10px;
      font-family:'Inter',sans-serif;font-size:.78rem;font-weight:600;cursor:pointer;
      border:1px solid var(--border);background:var(--surface2);color:var(--text);
      transition:all .18s;white-space:nowrap}
    .btn:hover{border-color:var(--accent);color:var(--accent);background:#1a0f30}
    .btn.active{border-color:var(--accent);background:#2d1b69;color:var(--accent)}
    .btn.danger{border-color:#ef444440;color:var(--red)}
    .btn.danger:hover{background:#2d0c0c;border-color:var(--red)}
    .btn-export{border-color:#22c55e40;color:var(--green)}
    .btn-export:hover{background:#0d2d1a;border-color:var(--green)}

    /* ── KPI GRID ── */
    .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:24px}
    .kpi{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;
      padding:20px;position:relative;overflow:hidden;transition:all .3s ease;
      backdrop-filter:blur(8px)}
    .kpi:hover{border-color:var(--accent);transform:translateY(-4px);box-shadow:0 8px 30px rgba(168,85,247,0.15)}
    .kpi::before{content:'';position:absolute;inset:0;border-radius:18px;opacity:0;
      background:linear-gradient(135deg,rgba(168,85,247,0.08),rgba(124,58,237,0.05));transition:opacity .3s}
    .kpi:hover::before{opacity:1}
    .kpi-icon{font-size:1.3rem;margin-bottom:10px;opacity:.8}
    .kpi-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin-bottom:6px}
    .kpi-val{font-size:2rem;font-weight:800;line-height:1;letter-spacing:-.04em;transition:color .3s}
    .kpi-val.blue{color:var(--accent)}
    .kpi-val.green{color:var(--green)}
    .kpi-val.red{color:var(--red)}
    .kpi-val.yellow{color:var(--yellow)}
    .kpi-val.cyan{color:var(--cyan)}
    .kpi-sub{font-size:.7rem;color:var(--muted);margin-top:5px}
    .kpi-bar{position:absolute;bottom:0;left:0;height:3px;border-radius:0 0 18px 18px;
      background:linear-gradient(90deg,#a855f7,#7c3aed);width:60%;transition:width .6s}

    /* ── ALERTS ── */
    .alert-row{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}
    .alert-card{flex:1;min-width:200px;border-radius:14px;padding:14px 18px;
      display:flex;align-items:center;gap:12px;font-size:.83rem;font-weight:500;
      border:1px solid transparent;animation:slide-in .3s ease}
    @keyframes slide-in{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
    .alert-em{background:#1a0a00;border-color:#f59e0b80;color:#fcd34d;animation:glow-yellow 1.5s ease-in-out infinite}
    @keyframes glow-yellow{0%,100%{box-shadow:0 0 10px #f59e0b20}50%{box-shadow:0 0 22px #f59e0b50}}
    .alert-icon{font-size:1.5rem;flex-shrink:0}
    .alert-title{font-weight:700;font-size:.88rem}
    .alert-sub{font-size:.75rem;opacity:.75;margin-top:2px}

    /* ── CHART ── */
    .chart-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;
      padding:24px;margin-bottom:20px;backdrop-filter:blur(8px);transition:border-color .3s}
    .chart-card:hover{border-color:#a855f740}
    .chart-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px}
    .chart-title{font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
    .chart-controls{display:flex;gap:8px}
    .chart-btn{padding:5px 12px;border-radius:8px;font-size:.72rem;font-weight:600;
      font-family:'Inter',sans-serif;cursor:pointer;border:1px solid var(--border);
      background:var(--surface2);color:var(--muted);transition:all .18s}
    .chart-btn.active{background:#2d1b69;border-color:var(--accent);color:var(--accent)}

    /* ── TABLE ── */
    .table-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;overflow:hidden;margin-bottom:20px;backdrop-filter:blur(8px)}
    .table-header{display:flex;align-items:center;justify-content:space-between;padding:18px 22px 0;flex-wrap:wrap;gap:10px;margin-bottom:14px}
    .section-title{font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
    .table-hint{font-size:.72rem;color:var(--muted);font-style:italic}
    table{width:100%;border-collapse:collapse}
    th{padding:10px 16px;text-align:left;font-size:.68rem;color:var(--muted);
      text-transform:uppercase;letter-spacing:.08em;font-weight:600;border-bottom:1px solid var(--border)}
    td{padding:12px 16px;border-bottom:1px solid #110e1a60;font-size:.84rem;
      transition:background .2s}
    tr:last-child td{border-bottom:none}
    tbody tr{cursor:pointer;transition:all .2s}
    tbody tr:hover td{background:#1f1535}
    tbody tr.lane-selected td{background:#2d1b69!important}
    tbody tr.lane-selected{outline:none}

    .los-badge{display:inline-flex;align-items:center;justify-content:center;
      width:28px;height:28px;border-radius:8px;font-size:.75rem;font-weight:800}
    .badge-green{background:#14532d40;color:#4ade80;border:1px solid #4ade8030}
    .badge-yellow{background:#713f1240;color:#fbbf24;border:1px solid #fbbf2430}
    .badge-orange{background:#7c2d1240;color:#fb923c;border:1px solid #fb923c30}
    .badge-red{background:#45090940;color:#f87171;border:1px solid #f8717130}

    .status-pill{display:inline-block;padding:3px 10px;border-radius:100px;font-size:.68rem;font-weight:700}
    .pill-clear{background:#14532d30;color:#4ade80;border:1px solid #4ade8025}
    .pill-moderate{background:#713f1230;color:#fbbf24;border:1px solid #fbbf2425}
    .pill-congested{background:#45090930;color:#f87171;border:1px solid #f8717125}

    .queue-val{color:var(--red);font-weight:700}
    .trend-up{color:var(--red);font-size:1rem;font-weight:800}
    .trend-down{color:var(--green);font-size:1rem;font-weight:800}
    .trend-flat{color:var(--muted);font-size:1rem;font-weight:800}

    /* ── SPEED TABLE ── */
    .speed-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:18px;overflow:hidden;margin-bottom:20px;backdrop-filter:blur(8px)}
    .speed-row-highlight{background:#2d0c0c}

    /* ── LEGEND ── */
    .legend{display:flex;gap:20px;flex-wrap:wrap;align-items:center;
      background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);border-radius:14px;
      padding:14px 20px;margin-bottom:20px;font-size:.75rem;backdrop-filter:blur(8px)}
    .legend-group{display:flex;align-items:center;gap:7px;color:var(--muted)}
    .legend-dot{width:8px;height:8px;border-radius:2px}
    .legend-sep{width:1px;height:16px;background:var(--border)}

    /* ── KEYS ── */
    .keys-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:8px}
    .key-item{background:var(--surface);border:1px solid var(--border);border-radius:12px;
      padding:12px 14px;display:flex;align-items:center;gap:10px}
    .key-cap{font-family:monospace;font-size:.95rem;font-weight:800;min-width:28px;
      text-align:center;background:#0d1530;border:1.5px solid var(--border);
      border-radius:6px;padding:2px 8px;color:var(--accent)}
    .key-desc{font-size:.75rem;color:var(--muted)}
    .key-name{font-weight:600;color:var(--text);font-size:.8rem;margin-bottom:2px}

    /* ── FOOTER ── */
    .footer{text-align:center;font-size:.72rem;color:var(--muted);margin-top:8px;padding-bottom:12px}
    .footer a{color:var(--accent);text-decoration:none}

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar{width:6px;height:6px}
    ::-webkit-scrollbar-track{background:var(--bg)}
    ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    ::-webkit-scrollbar-thumb:hover{background:#a855f780}

    /* ── SESSION SUMMARY MODAL ── */
    .sum-overlay{display:none;position:fixed;inset:0;z-index:999;
      background:rgba(8,6,14,.88);backdrop-filter:blur(14px);
      align-items:center;justify-content:center;padding:20px}
    .sum-overlay.open{display:flex}
    .sum-modal{background:linear-gradient(145deg,#110e1a,#181326);
      border:1px solid var(--border);border-radius:26px;padding:44px 40px;
      width:100%;max-width:540px;position:relative;
      box-shadow:0 0 100px rgba(168,85,247,.25);animation:sum-in .3s ease}
    @keyframes sum-in{from{opacity:0;transform:scale(.94) translateY(20px)}to{opacity:1;transform:scale(1) translateY(0)}}
    .sum-close{position:absolute;top:18px;right:20px;background:none;border:none;
      color:var(--muted);font-size:1.5rem;cursor:pointer;transition:color .15s;line-height:1}
    .sum-close:hover{color:var(--text)}
    .sum-header{text-align:center;margin-bottom:32px}
    .sum-icon{width:64px;height:64px;border-radius:20px;
      background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;
      font-size:1.8rem;margin:0 auto 16px;
      box-shadow:0 0 40px #a855f740}
    .sum-title{font-size:1.5rem;font-weight:800;letter-spacing:-.03em;margin-bottom:6px}
    .sum-sub{font-size:.82rem;color:var(--muted)}
    .sum-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
    .sum-card{background:linear-gradient(145deg,var(--surface),var(--surface2));
      border:1px solid var(--border);border-radius:16px;padding:18px 20px;
      transition:border-color .25s}
    .sum-card:hover{border-color:#a855f740}
    .sum-card-icon{font-size:1.2rem;margin-bottom:8px;opacity:.85}
    .sum-card-label{font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;
      color:var(--muted);font-weight:600;margin-bottom:4px}
    .sum-card-val{font-size:1.6rem;font-weight:800;letter-spacing:-.04em}
    .sum-card-val.purple{color:var(--accent)}
    .sum-card-val.green{color:var(--green)}
    .sum-card-val.red{color:var(--red)}
    .sum-card-val.yellow{color:var(--yellow)}
    .sum-card-val.cyan{color:var(--cyan)}
    .sum-peak{background:linear-gradient(135deg,#2d1b6940,#1a1030);
      border:1px solid #a855f730;border-radius:14px;padding:14px 18px;
      text-align:center;font-size:.82rem;color:var(--muted);margin-bottom:20px}
    .sum-peak b{color:var(--accent)}
    .sum-actions{display:flex;gap:10px}
    .sum-btn{flex:1;padding:12px;border-radius:12px;font-family:'Inter',sans-serif;
      font-size:.85rem;font-weight:700;cursor:pointer;border:1px solid var(--border);
      background:var(--surface2);color:var(--text);transition:all .18s}
    .sum-btn:hover{border-color:var(--accent);color:var(--accent);background:#1a0f30}
    .sum-btn.primary{background:linear-gradient(135deg,#a855f7,#7c3aed);
      border-color:transparent;color:#fff;box-shadow:0 0 20px #a855f730}
    .sum-btn.primary:hover{transform:translateY(-2px);box-shadow:0 6px 30px #a855f750}
  </style>
</head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="brand-icon">&#x1F6A6;</div>
      <div>
        <h1>TraffiQ</h1>
        <p>Real-time analysis &middot; localhost:5050</p>
      </div>
    </div>
    <div class="header-controls">
      <div class="live-badge" id="live-badge">
        <span class="live-dot" id="live-dot"></span>
        <span id="live-txt">LIVE</span>
      </div>
      <button class="btn" id="btn-pause" onclick="togglePause()">&#x23F8;&#xFE0F; Pause</button>
      <button class="btn btn-export" onclick="exportData()">&#x2B07;&#xFE0F; Export JSON</button>
      <button class="btn" onclick="openSummary()" style="border-color:#a855f740;color:var(--accent)">&#x1F4CB; Session Summary</button>
      <a href="/logout" class="btn danger" style="text-decoration:none">&#x1F6AA; Logout</a>
    </div>
  </header>

  <!-- KPI CARDS -->
  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-icon">&#x1F697;</div><div class="kpi-label">Total Vehicles</div><div class="kpi-val blue" id="k-veh">—</div><div class="kpi-sub" id="k-veh-sub">this session</div><div class="kpi-bar" style="background:linear-gradient(90deg,#3b82f6,#06b6d4)"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x1F4CD;</div><div class="kpi-label">Active Lanes</div><div class="kpi-val green" id="k-lanes">—</div><div class="kpi-sub">monitored</div><div class="kpi-bar" style="background:linear-gradient(90deg,#22c55e,#06b6d4);width:80%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x26A0;&#xFE0F;</div><div class="kpi-label">Incidents</div><div class="kpi-val green" id="k-inc">—</div><div class="kpi-sub">stopped vehicles</div><div class="kpi-bar" style="background:linear-gradient(90deg,#ef4444,#f59e0b);width:30%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x26A1;</div><div class="kpi-label">Mode</div><div class="kpi-val yellow" id="k-mode" style="font-size:1.1rem;letter-spacing:0">—</div><div class="kpi-sub">press keys to switch</div><div class="kpi-bar" style="background:linear-gradient(90deg,#f59e0b,#f97316);width:55%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x1F4F9;</div><div class="kpi-label">FPS</div><div class="kpi-val cyan" id="k-fps">—</div><div class="kpi-sub" id="k-frame-sub">frame —</div><div class="kpi-bar" style="background:linear-gradient(90deg,#06b6d4,#8b5cf6);width:70%"></div></div>
    <div class="kpi"><div class="kpi-icon">&#x1F6A8;</div><div class="kpi-label">Speeders</div><div class="kpi-val red" id="k-speeders">—</div><div class="kpi-sub" id="k-speed-limit">limit: — km/h</div><div class="kpi-bar" style="background:linear-gradient(90deg,#ef4444,#ec4899);width:40%"></div></div>
  </div>

  <!-- ALERT BANNERS -->
  <div id="em-banner" style="display:none;margin-bottom:16px">
    <div class="alert-card alert-em">
      <span class="alert-icon">&#x1F6A8;</span>
      <div><div class="alert-title">EMERGENCY VEHICLE</div><div class="alert-sub" id="em-txt">Lane — given priority</div></div>
    </div>
  </div>

  <!-- INCIDENTS -->
  <div id="inc-section" style="display:none;margin-bottom:16px">
    <div id="inc-list"></div>
  </div>

  <!-- CHART -->
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">&#x1F4CA; Lane Traffic &mdash; Last 2&nbsp;min</span>
      <div class="chart-controls">
        <button class="chart-btn active" id="btn-bar" onclick="setChartType('bar')">Bar</button>
        <button class="chart-btn" id="btn-line" onclick="setChartType('line')">Line</button>
      </div>
    </div>
    <canvas id="lane-chart" height="72"></canvas>
  </div>

  <!-- LEGEND ROW -->
  <div class="legend">
    <span style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:var(--muted)">Level of Service (Congestion)</span>
    <div class="legend-group"><div class="legend-dot" style="background:#4ade80"></div>A–B: Smooth / No delays</div>
    <div class="legend-group"><div class="legend-dot" style="background:#fbbf24"></div>C: Stable traffic</div>
    <div class="legend-group"><div class="legend-dot" style="background:#fb923c"></div>D: Slowing down</div>
    <div class="legend-group"><div class="legend-dot" style="background:#f87171"></div>E–F: Stop & Go / Jam</div>
    <div class="legend-sep"></div>
    <span style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:var(--muted)">Trend</span>
    <div class="legend-group"><span style="color:var(--red);font-weight:700">&#x2191;</span> Increasing</div>
    <div class="legend-group"><span style="color:var(--green);font-weight:700">&#x2193;</span> Decreasing</div>
    <div class="legend-group"><span style="color:var(--muted);font-weight:700">&#x2192;</span> Stable</div>
    <div class="legend-sep"></div>
    <div class="legend-group" style="margin-left:auto;font-style:italic">&#x1F4A1; Click a lane row to highlight it</div>
  </div>

  <!-- LANE TABLE -->
  <div class="table-card">
    <div class="table-header">
      <span class="section-title">&#x1F6E3;&#xFE0F; Lane Summary</span>
      <span class="table-hint" id="filter-hint"></span>
    </div>
    <table>
      <thead><tr>
        <th>Lane</th><th>Cars</th><th>Buses</th><th>Trucks</th><th>Bikes</th>
        <th>Total</th><th>LOS</th><th>Flow/min</th><th>Queue@Red</th><th>Status</th><th>Trend</th><th style="color:var(--accent)">Pred&#160;15s</th>
      </tr></thead>
      <tbody id="lane-tbody"></tbody>
    </table>
  </div>

  <!-- SPEED CAMERA TABLE -->
  <div id="speed-section" style="display:none">
    <div class="speed-card">
      <div class="table-header">
        <span class="section-title">&#x1F4F7; Speed Camera &mdash; Recent Violations</span>
      </div>
      <table>
        <thead><tr><th>Time</th><th>Track ID</th><th>Lane</th><th>Speed</th><th>Class</th></tr></thead>
        <tbody id="speed-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- KEYBOARD CONTROLS -->
  <div style="margin-bottom:20px">
    <div style="font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:12px">&#x2328;&#xFE0F; Keyboard Controls <span style="font-weight:400;text-transform:none;letter-spacing:0">(press in video window)</span></div>
    <div class="keys-grid">
      <div class="key-item"><span class="key-cap">L</span><div><div class="key-name">Lanes</div><div class="key-desc">Show lane regions &amp; counts</div></div></div>
      <div class="key-item"><span class="key-cap">H</span><div><div class="key-name">Heatmap</div><div class="key-desc">Vehicle density overlay</div></div></div>
      <div class="key-item"><span class="key-cap">S</span><div><div class="key-name">Speed</div><div class="key-desc">Labels per vehicle</div></div></div>
      <div class="key-item"><span class="key-cap">T</span><div><div class="key-name">Timer</div><div class="key-desc">Adaptive signal timing</div></div></div>
      <div class="key-item"><span class="key-cap" style="color:var(--red);border-color:var(--red)">Q</span><div><div class="key-name" style="color:var(--red)">Quit</div><div class="key-desc">Stop &amp; save session</div></div></div>
    </div>
  </div>

  <div class="footer">TraffiQ Dashboard &mdash; <a href="/api/stats">JSON API</a> &middot; <a href="/api/history">History API</a> &middot; <a href="/api/summary">Summary API</a></div>
</div>

<!-- SESSION SUMMARY MODAL -->
<div class="sum-overlay" id="sum-overlay" onclick="if(event.target===this)closeSummary()">
  <div class="sum-modal">
    <button class="sum-close" onclick="closeSummary()">&#x00D7;</button>
    <div class="sum-header">
      <div class="sum-icon">&#x1F4CB;</div>
      <div class="sum-title">Session Summary</div>
      <div class="sum-sub" id="sum-sub">Loading session data&hellip;</div>
    </div>
    <div class="sum-grid">
      <div class="sum-card">
        <div class="sum-card-icon">&#x23F1;&#xFE0F;</div>
        <div class="sum-card-label">Duration</div>
        <div class="sum-card-val purple" id="sum-duration">—</div>
      </div>
      <div class="sum-card">
        <div class="sum-card-icon">&#x1F697;</div>
        <div class="sum-card-label">Total Vehicles</div>
        <div class="sum-card-val cyan" id="sum-vehicles">—</div>
      </div>
      <div class="sum-card">
        <div class="sum-card-icon">&#x1F6A8;</div>
        <div class="sum-card-label">Speeders Caught</div>
        <div class="sum-card-val red" id="sum-speeders">—</div>
      </div>
      <div class="sum-card">
        <div class="sum-card-icon">&#x26A0;&#xFE0F;</div>
        <div class="sum-card-label">Incidents Flagged</div>
        <div class="sum-card-val yellow" id="sum-incidents">—</div>
      </div>
      <div class="sum-card">
        <div class="sum-card-icon">&#x1F504;</div>
        <div class="sum-card-label">Wrong-Way Vehicles</div>
        <div class="sum-card-val red" id="sum-wrongway">—</div>
      </div>
      <div class="sum-card">
        <div class="sum-card-icon">&#x1F697;&#x1F4A8;</div>
        <div class="sum-card-label">Tailgate Events</div>
        <div class="sum-card-val yellow" id="sum-tailgate">—</div>
      </div>
    </div>
    <div class="sum-peak" id="sum-peak-row">
      &#x1F3C6; Peak Traffic: <b id="sum-peak-count">—</b> vehicles at <b id="sum-peak-time">—</b>
    </div>
    <div class="sum-actions">
      <button class="sum-btn" onclick="exportSummary()">&#x2B07;&#xFE0F; Export JSON</button>
      <button class="sum-btn primary" onclick="closeSummary()">&#x2714; Got it</button>
    </div>
  </div>
</div>

<script>
// ── CONSTANTS ──
const COLORS=['#a855f7','#22c55e','#f59e0b','#ef4444','#7c3aed','#c084fc','#f97316','#ec4899'];
const LOS_C={A:'badge-green',B:'badge-green',C:'badge-yellow',D:'badge-orange',E:'badge-red',F:'badge-red'};
const LOS_COL={A:'#4ade80',B:'#a3e635',C:'#fbbf24',D:'#fb923c',E:'#f87171',F:'#dc2626'};

// ── STATE ──
let paused=false, selectedLane=null, chartType='bar', lastData={};

// ── CHART INIT ──
const ctx=document.getElementById('lane-chart');
let chart=new Chart(ctx,{
  type:'bar',
  data:{labels:[],datasets:[]},
  options:{
    responsive:true,animation:{duration:300},
    plugins:{legend:{labels:{color:'#64748b',font:{family:'Inter',size:11},boxWidth:12,boxHeight:12,padding:16}}},
    scales:{
      x:{ticks:{color:'#475569',font:{family:'Inter',size:10}},grid:{color:'#0d122580'}},
      y:{ticks:{color:'#475569',font:{family:'Inter',size:10}},grid:{color:'#1e2d5080'},beginAtZero:true}
    }
  }
});

function setChartType(t){
  chartType=t;
  document.getElementById('btn-bar').className='chart-btn'+(t==='bar'?' active':'');
  document.getElementById('btn-line').className='chart-btn'+(t==='line'?' active':'');
  // Mutate type in-place — Chart.js v4 supports this without destroy/recreate
  chart.config.type = t;
  chart.data.datasets.forEach(ds=>{
    ds.backgroundColor = ds.borderColor+(t==='bar'?'66':'22');
    ds.borderWidth     = t==='bar'?0:2;
    ds.borderRadius    = t==='bar'?6:0;
    ds.tension         = 0.4;
    ds.fill            = t==='line';
    ds.pointRadius     = t==='line'?2:0;
    ds.pointHoverRadius= 5;
  });
  chart.update();
}

function togglePause(){
  paused=!paused;
  const btn=document.getElementById('btn-pause');
  const badge=document.getElementById('live-badge');
  const dot=document.getElementById('live-dot');
  const txt=document.getElementById('live-txt');
  btn.innerHTML=paused?'&#x25B6;&#xFE0F; Resume':'&#x23F8;&#xFE0F; Pause';
  btn.className='btn'+(paused?' active':'');
  if(paused){
    dot.style.animation='none';
    dot.style.background='#f59e0b';
    txt.style.color='#fcd34d';
    txt.textContent='PAUSED';
    badge.classList.add('badge-paused');
    badge.classList.remove('badge-offline');
  } else {
    dot.style.animation='pulse-dot 1.4s ease-in-out infinite';
    dot.style.background='#22c55e';
    txt.style.color='#86efac';
    txt.textContent='LIVE';
    badge.classList.remove('badge-paused','badge-offline');
  }
}

function exportData(){
  const blob=new Blob([JSON.stringify(lastData,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='traffic_snapshot_'+new Date().toISOString().slice(0,19).replace(/:/g,'-')+'.json';
  a.click();
}

function losClass(l){return LOS_C[l]||'badge-green';}
function statusPill(t){
  if(t<5)return '<span class="status-pill pill-clear">CLEAR</span>';
  if(t<15)return '<span class="status-pill pill-moderate">MODERATE</span>';
  return '<span class="status-pill pill-congested">CONGESTED</span>';
}
function trendEl(a){
  if(a==='\u2191')return '<span class="trend-up">\u2191</span>';
  if(a==='\u2193')return '<span class="trend-down">\u2193</span>';
  return '<span class="trend-flat">\u2192</span>';
}

function animateVal(el,newVal){
  const cur=parseFloat(el.textContent)||0;
  if(cur===newVal||el.textContent==='—')return el.textContent=newVal;
  el.style.transition='color .3s';
  el.style.color=newVal>cur?'var(--red)':'var(--green)';
  el.textContent=newVal;
  setTimeout(()=>el.style.color='',500);
}

function updateChart(hist){
  if(!hist.length)return;
  const labels=hist.map(h=>h.t);
  let lids=[...new Set(hist.flatMap(h=>Object.keys(h.lanes||{})))].sort();
  if (selectedLane) {
    lids = lids.filter(l => String(l) === String(selectedLane));
  }
  const newDatasets=lids.map((lid,i)=>({
    label:'Lane '+lid,
    data:hist.map(h=>(h.lanes||{})[lid]??0),
    backgroundColor:COLORS[i%COLORS.length]+(chartType==='bar'?'66':'22'),
    borderColor:COLORS[i%COLORS.length],
    borderWidth:chartType==='bar'?0:2,
    borderRadius:chartType==='bar'?6:0,
    tension:0.4,fill:chartType==='line',
    pointRadius:chartType==='line'?2:0,
    pointHoverRadius:5,
  }));
  chart.data.labels=labels;
  chart.data.datasets=newDatasets;
  chart.update('none');
}

function renderData(s, hist) {
  // KPIs
  animateVal(document.getElementById('k-veh'),s.vehicle_count??'—');
  document.getElementById('k-lanes').textContent=Object.keys(s.lane_counts||{}).length;
  const incEl=document.getElementById('k-inc');
  incEl.textContent=(s.incidents||[]).length;
  incEl.className='kpi-val '+(s.incidents?.length?'red':'green');
  document.getElementById('k-mode').textContent=(s.mode||'').toUpperCase()||'—';
  document.getElementById('k-fps').textContent=s.fps??'—';
  document.getElementById('k-frame-sub').textContent='frame '+(s.frame_id??'—')+(s.ml_ready?' | ML✓':' | ML⋯');
  document.getElementById('k-speeders').textContent=(s.speeders||[]).length;
  if(s.speed_limit)document.getElementById('k-speed-limit').textContent='limit: '+s.speed_limit+' km/h';

  // Emergency
  const emb=document.getElementById('em-banner');
  if(s.emergency_active){
    emb.style.display='block';
    document.getElementById('em-txt').textContent='Lane '+s.emergency_lane+' given priority — timer extended';
  } else emb.style.display='none';

  // Incidents
  const is=document.getElementById('inc-section');
  if(s.incidents?.length){
    is.style.display='block';
    document.getElementById('inc-list').innerHTML=s.incidents.map(inc=>`
      <div class="alert-card" style="background:#1a0808;border-color:#ef444450;color:#fca5a5;margin-bottom:8px">
        <span class="alert-icon">&#x1F6A8;</span>
        <div>
          <div class="alert-title">Incident &mdash; Lane ${inc.lane}</div>
          <div class="alert-sub">Vehicle #${inc.track_id} stopped ${inc.duration}s at (${inc.cx}, ${inc.cy})</div>
        </div>
      </div>`).join('');
  } else is.style.display='none';

  // Lane table
  const tb=document.getElementById('lane-tbody');
  tb.innerHTML='';
  for(const[lid,counts] of Object.entries(s.lane_counts||{})){
    const total=Object.values(counts).reduce((a,b)=>a+b,0);
    const los=(s.lane_los||{})[lid]||'?';
    const flow=(s.lane_flow||{})[lid]??'—';
    const q=(s.lane_queue||{})[lid]??0;
    const arrow=(s.lane_trends||{})[lid]||'\u2192';
    const pred=(s.lane_predictions||{})[lid];
    const mlReady=s.ml_ready||false;
    const isSelected=String(selectedLane)===String(lid);
    const lc=LOS_COL[los]||'#64748b';
    const flowDisp=typeof flow==='number'?flow.toFixed(1):flow;
    const predDisp=mlReady&&pred!=null?`<span style="color:var(--accent);font-weight:700">${pred}</span>`:`<span style="color:var(--muted);font-size:.72rem">warmup</span>`;
    tb.innerHTML+=`
      <tr onclick="selectLane('${lid}')" ${isSelected?'class="lane-selected"':''} id="lane-row-${lid}">
        <td><b style="font-size:.9rem">Lane ${lid}</b></td>
        <td>${counts.car||0}</td><td>${counts.bus||0}</td>
        <td>${counts.truck||0}</td><td>${counts.motorbike||0}</td>
        <td><b>${total}</b></td>
        <td><span class="los-badge ${losClass(los)}">${los}</span></td>
        <td style="font-variant-numeric:tabular-nums">${flowDisp}</td>
        <td>${q>0?`<span class="queue-val">${q}</span>`:q}</td>
        <td>${statusPill(total)}</td>
        <td>${trendEl(arrow)}</td>
        <td>${predDisp}</td>
      </tr>`;
  }
  if(selectedLane)document.getElementById('filter-hint').textContent='Lane '+selectedLane+' selected — click again to deselect';
  else document.getElementById('filter-hint').textContent='';

  // Speed
  const ss=document.getElementById('speed-section');
  if(s.speeders?.length){
    ss.style.display='block';
    document.getElementById('speed-tbody').innerHTML=[...s.speeders].reverse().slice(0,12).map(sp=>`
      <tr>
        <td style="color:var(--muted);font-size:.76rem">${sp.timestamp}</td>
        <td><b>#${sp.track_id}</b></td>
        <td>Lane ${sp.lane||'—'}</td>
        <td><span style="color:var(--red);font-weight:700">${sp.speed_kmh} km/h</span></td>
        <td style="color:var(--muted)">${sp.class||'—'}</td>
      </tr>`).join('');
  } else ss.style.display='none';

  // Chart
  updateChart(hist);
}

async function poll(){
  if(paused)return;
  try{
    const[sr,hr]=await Promise.all([fetch('/api/stats'),fetch('/api/history')]);
    if(!sr.ok || !hr.ok) throw new Error('Network response was not ok');
    lastData = {s: await sr.json(), hist: await hr.json()};
    
    // Restore LIVE status if we were offline
    const badge=document.getElementById('live-badge');
    const dot=document.getElementById('live-dot');
    const txt=document.getElementById('live-txt');
    if(badge.classList.contains('badge-offline')){
      badge.classList.remove('badge-offline');
      dot.style.animation='pulse-dot 1.4s ease-in-out infinite';
      dot.style.background='#22c55e';
      txt.style.color='#86efac';
      txt.textContent='LIVE';
    }
    
    renderData(lastData.s, lastData.hist);
  }catch(e){
    console.warn('poll err',e);
    const badge=document.getElementById('live-badge');
    const dot=document.getElementById('live-dot');
    const txt=document.getElementById('live-txt');
    badge.classList.add('badge-offline');
    dot.style.animation='none';
    dot.style.background='#ef4444';
    txt.style.color='#fca5a5';
    txt.textContent='OFFLINE';
  }
}

function selectLane(lid){
  if(String(selectedLane)===String(lid)){selectedLane=null;}
  else{selectedLane=lid;}
  if(lastData.s) renderData(lastData.s, lastData.hist);
}

// ── SESSION SUMMARY MODAL ──
let lastSummary = {};
let sessionEndedNotified = false;

async function fetchSummary(){
  try{
    const r = await fetch('/api/summary');
    if(!r.ok) return;
    const d = await r.json();
    lastSummary = d;
    // Auto-open when session ends (Q pressed in OpenCV window)
    if(d.session_ended && !sessionEndedNotified){
      sessionEndedNotified = true;
      openSummary();
    }
  } catch(e){}
}

function openSummary(){
  fetchSummary().then(()=>{
    const d = lastSummary;
    document.getElementById('sum-duration').textContent  = d.duration   ?? '—';
    document.getElementById('sum-vehicles').textContent  = d.total_vehicles ?? '—';
    document.getElementById('sum-speeders').textContent  = d.speeders   ?? '—';
    document.getElementById('sum-incidents').textContent = d.total_incidents ?? '—';
    document.getElementById('sum-wrongway').textContent  = d.wrong_way  ?? '—';
    document.getElementById('sum-tailgate').textContent  = d.tailgate_events ?? '—';
    document.getElementById('sum-peak-count').textContent = d.peak_count ?? '—';
    document.getElementById('sum-peak-time').textContent  = d.peak_time  || 'N/A';
    document.getElementById('sum-sub').textContent = 'Started at ' + (d.session_start ?? '—');
    document.getElementById('sum-overlay').classList.add('open');
  });
}

function closeSummary(){
  document.getElementById('sum-overlay').classList.remove('open');
}

function exportSummary(){
  const blob = new Blob([JSON.stringify(lastSummary, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'session_summary_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.json';
  a.click();
}

document.addEventListener('keydown', e => { if(e.key==='Escape') closeSummary(); });

poll(); setInterval(poll, 2000);
setInterval(fetchSummary, 3000);
</script>
</body></html>
"""

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>TraffiQ &mdash; AI-Powered Traffic Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
  <style>
    :root{--bg:#08060e;--surface:#110e1a;--surface2:#181326;--border:#2a1f4a;
      --accent:#a855f7;--accent2:#7c3aed;--green:#22c55e;--muted:#8b7fb0;--cyan:#c084fc;--text:#e2e8f0;}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden}
    body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
      background:radial-gradient(ellipse 80% 50% at 50% -20%,#2d1b6955,transparent),
                 radial-gradient(ellipse 60% 40% at 80% 110%,#4c1d9544,transparent),
                 radial-gradient(ellipse 40% 30% at 10% 60%,#7c3aed18,transparent);}
    nav{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
      padding:16px 48px;background:rgba(8,6,14,.85);backdrop-filter:blur(20px);border-bottom:1px solid var(--border)}
    .nav-brand{display:flex;align-items:center;gap:12px;text-decoration:none}
    .nav-icon{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;font-size:1.2rem;box-shadow:0 0 20px #a855f740}
    .nav-title{font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em}
    .nav-btn{padding:10px 22px;border-radius:10px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      color:#fff;font-family:'Inter',sans-serif;font-size:.85rem;font-weight:700;border:none;cursor:pointer;
      box-shadow:0 0 20px #a855f740;transition:all .2s}
    .nav-btn:hover{transform:translateY(-2px);box-shadow:0 4px 30px #a855f760}
    .hero{position:relative;z-index:1;text-align:center;padding:120px 24px 80px}
    .hero-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:100px;
      background:#2d1b6940;border:1px solid #a855f740;font-size:.75rem;font-weight:600;color:var(--cyan);
      margin-bottom:28px;letter-spacing:.08em;text-transform:uppercase}
    .hero-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pdot 1.4s ease-in-out infinite}
    @keyframes pdot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.7)}}
    .hero h1{font-size:clamp(2.5rem,6vw,4.5rem);font-weight:900;letter-spacing:-.04em;line-height:1.05;margin-bottom:24px}
    .hero h1 span{background:linear-gradient(135deg,#a855f7,#c084fc,#e879f9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .hero-sub{font-size:1.1rem;color:var(--muted);max-width:580px;margin:0 auto 40px;line-height:1.7}
    .hero-btns{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
    .btn-primary{padding:14px 32px;border-radius:12px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      color:#fff;font-family:'Inter',sans-serif;font-size:.95rem;font-weight:700;border:none;cursor:pointer;
      box-shadow:0 0 30px #a855f740;transition:all .25s}
    .btn-primary:hover{transform:translateY(-3px);box-shadow:0 8px 40px #a855f760}
    .btn-ghost{padding:14px 32px;border-radius:12px;background:transparent;color:var(--text);
      font-family:'Inter',sans-serif;font-size:.95rem;font-weight:600;border:1px solid var(--border);
      cursor:pointer;transition:all .25s}
    .btn-ghost:hover{border-color:var(--accent);color:var(--accent);background:#1a0f30;transform:translateY(-2px)}
    .stats-bar{position:relative;z-index:1;display:flex;max-width:900px;margin:0 auto 100px;
      border:1px solid var(--border);border-radius:20px;
      background:linear-gradient(145deg,var(--surface),var(--surface2));overflow:hidden;backdrop-filter:blur(8px)}
    .stat-item{flex:1;padding:28px 20px;text-align:center;position:relative}
    .stat-item+.stat-item::before{content:'';position:absolute;left:0;top:20%;height:60%;width:1px;background:var(--border)}
    .stat-num{font-size:1.8rem;font-weight:900;background:linear-gradient(135deg,#a855f7,#c084fc);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .stat-label{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:4px}
    .section{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:0 24px 100px}
    .section-tag{display:inline-block;padding:4px 14px;border-radius:100px;background:#2d1b6940;
      border:1px solid #a855f740;font-size:.72rem;font-weight:700;color:var(--cyan);
      text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px}
    .section-title{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;letter-spacing:-.03em;margin-bottom:14px}
    .section-sub{color:var(--muted);font-size:1rem;line-height:1.7;max-width:520px;margin-bottom:56px}
    .feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
    .feat-card{background:linear-gradient(145deg,var(--surface),var(--surface2));border:1px solid var(--border);
      border-radius:20px;padding:28px;transition:all .3s;position:relative;overflow:hidden}
    .feat-card:hover{border-color:var(--accent);transform:translateY(-5px);box-shadow:0 12px 40px rgba(168,85,247,.15)}
    .feat-card::before{content:'';position:absolute;inset:0;border-radius:20px;opacity:0;
      background:linear-gradient(135deg,rgba(168,85,247,.07),rgba(124,58,237,.04));transition:opacity .3s}
    .feat-card:hover::before{opacity:1}
    .feat-icon{font-size:2rem;margin-bottom:16px}.feat-title{font-size:1.05rem;font-weight:700;margin-bottom:8px}
    .feat-desc{font-size:.85rem;color:var(--muted);line-height:1.65}
    .feat-tag{display:inline-block;margin-top:14px;padding:3px 10px;border-radius:6px;
      font-size:.68rem;font-weight:700;background:#2d1b6940;color:var(--cyan);border:1px solid #a855f730}
    .metrics-preview{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-top:56px}
    .metric-card{background:linear-gradient(145deg,var(--surface),var(--surface2));
      border:1px solid var(--border);border-radius:16px;padding:20px;text-align:center;transition:all .25s}
    .metric-card:hover{border-color:var(--accent);transform:translateY(-3px);box-shadow:0 8px 24px rgba(168,85,247,.12)}
    .metric-card-icon{font-size:1.5rem;margin-bottom:10px}
    .metric-card-val{font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#a855f7,#c084fc);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .metric-card-label{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:4px}
    footer{position:relative;z-index:1;text-align:center;padding:36px 24px;
      border-top:1px solid var(--border);font-size:.78rem;color:var(--muted)}
    .modal-overlay{display:none;position:fixed;inset:0;z-index:1000;background:rgba(8,6,14,.88);
      backdrop-filter:blur(12px);align-items:center;justify-content:center}
    .modal-overlay.open{display:flex}
    .modal{background:linear-gradient(145deg,#110e1a,#181326);border:1px solid var(--border);
      border-radius:24px;padding:40px;width:100%;max-width:420px;
      box-shadow:0 0 80px rgba(168,85,247,.2);position:relative;animation:modal-in .25s ease}
    @keyframes modal-in{from{opacity:0;transform:scale(.95) translateY(16px)}to{opacity:1;transform:scale(1) translateY(0)}}
    .modal-close{position:absolute;top:16px;right:18px;background:none;border:none;
      color:var(--muted);font-size:1.4rem;cursor:pointer;transition:color .15s}
    .modal-close:hover{color:var(--text)}
    .modal-icon{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#a855f7,#7c3aed);
      display:flex;align-items:center;justify-content:center;font-size:1.6rem;
      margin:0 auto 20px;box-shadow:0 0 30px #a855f740}
    .modal h2{text-align:center;font-size:1.4rem;font-weight:800;margin-bottom:6px}
    .modal-sub{text-align:center;font-size:.82rem;color:var(--muted);margin-bottom:28px}
    .form-group{margin-bottom:18px}
    .form-label{display:block;font-size:.75rem;font-weight:600;color:var(--muted);
      text-transform:uppercase;letter-spacing:.08em;margin-bottom:7px}
    .form-input{width:100%;padding:12px 16px;border-radius:10px;background:#0d0a14;
      border:1px solid var(--border);color:var(--text);font-family:'Inter',sans-serif;
      font-size:.9rem;outline:none;transition:border-color .2s}
    .form-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(168,85,247,.12)}
    .login-btn{width:100%;padding:13px;border-radius:10px;
      background:linear-gradient(135deg,#a855f7,#7c3aed);color:#fff;
      font-family:'Inter',sans-serif;font-size:.95rem;font-weight:700;
      border:none;cursor:pointer;box-shadow:0 0 24px #a855f730;transition:all .2s;margin-top:4px}
    .login-btn:hover{transform:translateY(-2px);box-shadow:0 6px 32px #a855f750}
    .login-error{display:none;background:#2d0c0c;border:1px solid #ef444450;border-radius:8px;
      padding:10px 14px;font-size:.82rem;color:#fca5a5;margin-bottom:16px}
    .login-error.show{display:block}
    ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}
    ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    ::-webkit-scrollbar-thumb:hover{background:#a855f780}
  </style>
</head>
<body>
<nav>
  <a class="nav-brand" href="/"><div class="nav-icon">&#x1F6A6;</div><span class="nav-title">TraffiQ</span></a>
  <button class="nav-btn" onclick="openLogin()">&#x1F511; Login to Dashboard</button>
</nav>
<div class="hero">
  <div class="hero-badge"><span class="hero-badge-dot"></span>AI-Powered &middot; Real-Time &middot; YOLOv8</div>
  <h1>Next-Gen <span>TraffiQ</span><br>at Your Fingertips</h1>
  <p class="hero-sub">Professional-grade traffic analysis powered by YOLOv8 + ByteTrack. Real-time detection, speed analysis, incident alerts and adaptive signals in a live dashboard.</p>
  <div class="hero-btns">
    <button class="btn-primary" onclick="openLogin()">&#x1F680; Open Dashboard</button>
    <button class="btn-ghost" onclick="document.getElementById('features').scrollIntoView({behavior:'smooth'})">&#x1F50E; Explore Features</button>
  </div>
</div>
<div class="stats-bar" style="max-width:900px;margin:0 auto 100px">
  <div class="stat-item"><div class="stat-num">YOLOv8</div><div class="stat-label">Detection Model</div></div>
  <div class="stat-item"><div class="stat-num">ByteTrack</div><div class="stat-label">Object Tracking</div></div>
  <div class="stat-item"><div class="stat-num">4</div><div class="stat-label">Vehicle Classes</div></div>
  <div class="stat-item"><div class="stat-num">2s</div><div class="stat-label">Update Interval</div></div>
</div>
<div class="section" id="features">
  <div class="section-tag">&#x2728; Features</div>
  <div class="section-title">Everything you need to<br>monitor traffic</div>
  <p class="section-sub">Industry-standard algorithms combined in one sleek dashboard.</p>
  <div class="feat-grid">
    <div class="feat-card"><div class="feat-icon">&#x1F3AF;</div><div class="feat-title">Multi-Object Tracking</div><div class="feat-desc">ByteTrack tracks every vehicle with unique IDs across frames, even through occlusions.</div><span class="feat-tag">ByteTrack</span></div>
    <div class="feat-card"><div class="feat-icon">&#x26A1;</div><div class="feat-title">Lane-Level Analytics</div><div class="feat-desc">Per-lane vehicle counts, LOS A&ndash;F grading, flow rate per minute, and queue length at red.</div><span class="feat-tag">HCM Standard</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4F7;</div><div class="feat-title">Speed Camera</div><div class="feat-desc">Pixel-to-meter calibrated speed per vehicle. Violations logged to CSV in real time.</div><span class="feat-tag">Configurable Limit</span></div>
    <div class="feat-card"><div class="feat-icon">&#x26A0;&#xFE0F;</div><div class="feat-title">Incident Detection</div><div class="feat-desc">Flags vehicles stopped for &gt;5s with instant dashboard alerts showing location and duration.</div><span class="feat-tag">Auto Alert</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F6A8;</div><div class="feat-title">Emergency Priority</div><div class="feat-desc">Fast-moving large vehicles trigger automatic signal timer extension on their lane.</div><span class="feat-tag">Smart Signals</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4CA;</div><div class="feat-title">Live Chart History</div><div class="feat-desc">2-minute time-series chart of lane traffic with instant bar / line toggle.</div><span class="feat-tag">Chart.js v4</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F525;</div><div class="feat-title">Heatmap Overlay</div><div class="feat-desc">Accumulative density heatmap of vehicle dwell time. PNG exportable for offline analysis.</div><span class="feat-tag">OpenCV</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4C8;</div><div class="feat-title">Trend Prediction</div><div class="feat-desc">Rolling linear regression predicts traffic direction (rising / falling / stable) per lane.</div><span class="feat-tag">No ML Required</span></div>
    <div class="feat-card"><div class="feat-icon">&#x1F4BE;</div><div class="feat-title">CSV Data Logging</div><div class="feat-desc">Every detection and speed event logged to CSV. JSON snapshot export from the dashboard.</div><span class="feat-tag">Analytics-Ready</span></div>
  </div>
</div>
<div class="section">
  <div class="section-tag">&#x1F4DD; Dashboard Metrics</div>
  <div class="section-title">What the dashboard shows you</div>
  <p class="section-sub">All key metrics in real-time, updating every 2 seconds.</p>
  <div class="metrics-preview">
    <div class="metric-card"><div class="metric-card-icon">&#x1F697;</div><div class="metric-card-val">Live</div><div class="metric-card-label">Total Vehicles</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F4CD;</div><div class="metric-card-val">A&ndash;F</div><div class="metric-card-label">Level of Service</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F4F9;</div><div class="metric-card-val">FPS</div><div class="metric-card-label">Processing Speed</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F6A8;</div><div class="metric-card-val">Auto</div><div class="metric-card-label">Speeder Alerts</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x26A0;&#xFE0F;</div><div class="metric-card-val">Real-Time</div><div class="metric-card-label">Incidents</div></div>
    <div class="metric-card"><div class="metric-card-icon">&#x1F4CA;</div><div class="metric-card-val">2 Min</div><div class="metric-card-label">History Chart</div></div>
  </div>
</div>
<footer>TraffiQ &mdash; AI-Powered Traffic Analysis &middot; Built with YOLOv8 + ByteTrack + Flask
  <span style="color:#a855f7;margin-left:8px">&#x2665; By Varnit</span>
</footer>
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeLogin()">
  <div class="modal">
    <button class="modal-close" onclick="closeLogin()">&times;</button>
    <div class="modal-icon">&#x1F6A6;</div>
    <h2>Welcome Back</h2>
    <p class="modal-sub">Enter your credentials to access the live dashboard</p>
    <div class="login-error" id="login-error">&#x26A0; Incorrect username or password</div>
    <form method="POST" action="/login">
      <div class="form-group"><label class="form-label">Username</label>
        <input class="form-input" type="text" name="username" id="inp-user" placeholder="Enter username" required/></div>
      <div class="form-group"><label class="form-label">Password</label>
        <input class="form-input" type="password" name="password" placeholder="Enter password" required/></div>
      <button class="login-btn" type="submit">&#x1F680; Access Dashboard</button>
    </form>
  </div>
</div>
<script>
function openLogin(){document.getElementById('modal-overlay').classList.add('open');setTimeout(()=>document.getElementById('inp-user').focus(),100);}
function closeLogin(){document.getElementById('modal-overlay').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLogin();});
if(new URLSearchParams(window.location.search).get('error')==='1'){openLogin();document.getElementById('login-error').classList.add('show');}
</script>
</body></html>"""
