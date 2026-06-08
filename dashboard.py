"""
job_pipeline/dashboard.py
=========================
Interactive web dashboard for the job-application tracker.

What it does
------------
- Imports applications from `tracker.json` into a local SQLite database
  (`tracker.db`). Re-importing is safe: new jobs are added and scraped fields
  are refreshed, but your own edits (Applied status, notes) are preserved.
- Serves a single-page dashboard at http://127.0.0.1:8000 showing, per job:
  posted date, fit-match score, pipeline status, application link, an editable
  Applied / Pending status, and editable notes.
- Shows analytics: totals, applied vs pending, score distribution, top
  companies, and pipeline-status breakdown.
- Your edits (Applied/Pending, notes) are saved straight to the database.

Run it
------
    python dashboard.py                 # imports tracker.json, opens browser
    python dashboard.py --port 9000     # custom port
    python dashboard.py --no-browser    # don't auto-open the browser

Zero external dependencies — standard library only.
"""

import argparse
import json
import sqlite3
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).parent
TRACKER_JSON = BASE_DIR / "tracker.json"
DB_PATH = BASE_DIR / "tracker.db"
OUTPUT_DIR = (BASE_DIR / "applications").resolve()  # generated docs live here

# Fields the user is allowed to edit from the UI (everything else is pipeline-owned).
EDITABLE_FIELDS = {"applied_status", "notes"}


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def get_conn() -> sqlite3.Connection:
    """Open a fresh connection (one per request — safe across server threads)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the applications table if it doesn't exist, and migrate older DBs."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key       TEXT UNIQUE,          -- dedup key (url or company|title)
                company          TEXT,
                role_title       TEXT,
                role_url         TEXT,
                date_posted      TEXT,                 -- when the portal posted/updated it
                fit_score        REAL,
                pipeline_status  TEXT,                 -- status set by run_pipeline.py
                applied_status   TEXT DEFAULT 'pending', -- user-owned: pending | applied
                applied_date     TEXT,                 -- when the pipeline processed it
                cover_letter_path TEXT,
                resume_path      TEXT,                 -- tailored resume (Tier A/B, score>=70%)
                spoc_email       TEXT,                 -- contact email if found in posting
                notes            TEXT DEFAULT ''        -- user-owned free text
            )
            """
        )
        # Migrate databases created before these columns existed.
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(applications)")}
        for col in ("resume_path", "spoc_email"):
            if col not in existing:
                conn.execute(f"ALTER TABLE applications ADD COLUMN {col} TEXT")


def _source_key(app: dict) -> str:
    """Stable identity for a job: prefer URL, fall back to company|title."""
    url = (app.get("role_url") or "").strip()
    if url:
        return url
    return f"{app.get('company', '')}|{app.get('role_title', '')}".lower()


def sync_from_json() -> dict:
    """Import/refresh rows from tracker.json. Returns {added, updated, total}."""
    if not TRACKER_JSON.exists():
        return {"added": 0, "updated": 0, "total": 0, "error": "tracker.json not found"}

    with open(TRACKER_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    apps = data.get("applications", [])

    added = updated = 0
    with get_conn() as conn:
        for app in apps:
            key = _source_key(app)
            row = conn.execute(
                "SELECT id FROM applications WHERE source_key = ?", (key,)
            ).fetchone()

            if row is None:
                # New job — applied_status defaults to 'pending'.
                conn.execute(
                    """
                    INSERT INTO applications
                        (source_key, company, role_title, role_url, date_posted,
                         fit_score, pipeline_status, applied_status, applied_date,
                         cover_letter_path, resume_path, spoc_email, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, '')
                    """,
                    (
                        key, app.get("company", ""), app.get("role_title", ""),
                        app.get("role_url", ""), app.get("date_posted", ""),
                        app.get("fit_score"), app.get("status", ""),
                        app.get("applied_date", ""), app.get("cover_letter_path", ""),
                        app.get("resume_path", ""), app.get("spoc_email", ""),
                    ),
                )
                added += 1
            else:
                # Existing job — refresh pipeline-owned fields only.
                # applied_status and notes are NOT touched (they're the user's).
                conn.execute(
                    """
                    UPDATE applications
                       SET company = ?, role_title = ?, role_url = ?, date_posted = ?,
                           fit_score = ?, pipeline_status = ?, applied_date = ?,
                           cover_letter_path = ?, resume_path = ?, spoc_email = ?
                     WHERE id = ?
                    """,
                    (
                        app.get("company", ""), app.get("role_title", ""),
                        app.get("role_url", ""), app.get("date_posted", ""),
                        app.get("fit_score"), app.get("status", ""),
                        app.get("applied_date", ""), app.get("cover_letter_path", ""),
                        app.get("resume_path", ""), app.get("spoc_email", ""),
                        row["id"],
                    ),
                )
                updated += 1

        total = conn.execute("SELECT COUNT(*) AS c FROM applications").fetchone()["c"]

    return {"added": added, "updated": updated, "total": total}


def fetch_all() -> list[dict]:
    """Return all applications as a list of dicts."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY fit_score DESC, id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_field(row_id: int, field: str, value) -> bool:
    """Update a single editable field on one row."""
    if field not in EDITABLE_FIELDS:
        return False
    if field == "applied_status" and value not in ("applied", "pending"):
        return False
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE applications SET {field} = ? WHERE id = ?", (value, row_id)
        )
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# HTML (single page; analytics computed client-side, no external libraries)
# --------------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Application Tracker</title>
<style>
  :root{
    --bg:#0f1419; --panel:#1a2029; --panel2:#222b36; --border:#2c3744;
    --text:#e6edf3; --muted:#8b98a5; --accent:#4f9cf9;
    --green:#3fb950; --amber:#d29922; --red:#f85149;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:var(--bg);color:var(--text);font-size:14px}
  header{padding:20px 28px;border-bottom:1px solid var(--border);
         display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  h1{font-size:20px;margin:0;font-weight:600}
  .sub{color:var(--muted);font-size:13px}
  button{background:var(--panel2);color:var(--text);border:1px solid var(--border);
         border-radius:6px;padding:7px 14px;cursor:pointer;font-size:13px}
  button:hover{border-color:var(--accent)}
  .primary{background:var(--accent);border-color:var(--accent);color:#04111f;font-weight:600}
  main{padding:24px 28px;max-width:1400px;margin:0 auto}

  /* stat cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:22px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
  .card .num{font-size:26px;font-weight:700}
  .card .lbl{color:var(--muted);font-size:12px;margin-top:4px;text-transform:uppercase;letter-spacing:.04em}

  /* charts */
  .charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin-bottom:26px}
  .chart{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
  .chart h3{margin:0 0 14px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  .bar-row{display:flex;align-items:center;gap:10px;margin-bottom:9px;font-size:13px}
  .bar-row .name{width:130px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--muted)}
  .bar-track{flex:1;background:var(--panel2);border-radius:5px;height:18px;overflow:hidden}
  .bar-fill{height:100%;border-radius:5px;background:var(--accent)}
  .bar-row .val{width:34px;text-align:right;font-variant-numeric:tabular-nums}

  /* controls */
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
  input,select{background:var(--panel2);color:var(--text);border:1px solid var(--border);
       border-radius:6px;padding:7px 10px;font-size:13px}
  input.search{min-width:240px}

  /* table */
  table{width:100%;border-collapse:collapse;background:var(--panel);
        border:1px solid var(--border);border-radius:10px;overflow:hidden}
  th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
  th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;
     cursor:pointer;user-select:none;white-space:nowrap}
  th:hover{color:var(--text)}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#1e2630}
  .role{font-weight:600}
  .company{color:var(--muted);font-size:12px}
  .score{font-weight:700;font-variant-numeric:tabular-nums}
  .score.g{color:var(--green)} .score.a{color:var(--amber)} .score.r{color:var(--red)}
  .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;
        background:var(--panel2);border:1px solid var(--border);color:var(--muted)}
  a.link{color:var(--accent);text-decoration:none}
  a.link:hover{text-decoration:underline}
  select.status{font-weight:600}
  select.status.applied{color:var(--green);border-color:var(--green)}
  select.status.pending{color:var(--amber);border-color:var(--amber)}
  .notes{width:100%;min-width:140px}
  .muted{color:var(--muted)}
  .saved{color:var(--green);font-size:11px;opacity:0;transition:opacity .2s}
  .saved.show{opacity:1}
  .empty{padding:50px;text-align:center;color:var(--muted)}
</style>
</head>
<body>
<header>
  <div>
    <h1>📊 Job Application Tracker</h1>
    <div class="sub" id="subtitle">Loading…</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:10px;align-items:center">
    <span class="saved" id="saveIndicator">Saved ✓</span>
    <button class="primary" id="syncBtn">⟳ Sync tracker.json</button>
  </div>
</header>

<main>
  <div class="cards" id="cards"></div>
  <div class="charts" id="charts"></div>

  <div class="controls">
    <input class="search" id="search" placeholder="Search company or role…">
    <select id="filterStatus">
      <option value="all">All statuses</option>
      <option value="applied">Applied only</option>
      <option value="pending">Pending only</option>
    </select>
    <select id="filterScore">
      <option value="0">Any score</option>
      <option value="0.7">Strong (≥70%)</option>
      <option value="0.5">Medium (≥50%)</option>
    </select>
    <span class="muted" id="count"></span>
  </div>

  <table>
    <thead>
      <tr>
        <th data-sort="company">Company</th>
        <th data-sort="role_title">Role</th>
        <th data-sort="date_posted">Posted</th>
        <th data-sort="fit_score">Score</th>
        <th data-sort="pipeline_status">Pipeline</th>
        <th data-sort="applied_status">Applied?</th>
        <th>Contact</th>
        <th>Docs</th>
        <th>Notes</th>
        <th>Link</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none">No jobs match your filters.</div>
</main>

<script>
let DATA = [];
let sortKey = "fit_score", sortDir = -1;

async function load(){
  const res = await fetch("/api/applications");
  DATA = await res.json();
  renderCards(); renderCharts(); renderTable();
}

function fmtDate(s){
  if(!s) return "—";
  const d = new Date(s);
  if(isNaN(d)) return s;
  const today = new Date();
  const days = Math.floor((today - d)/86400000);
  const ds = d.toISOString().slice(0,10);
  if(days < 0) return ds;
  if(days === 0) return ds + " (today)";
  if(days === 1) return ds + " (1d ago)";
  return ds + " (" + days + "d ago)";
}
function scoreClass(s){ return s>=0.7?"g":s>=0.5?"a":"r"; }
function fmtScore(s){ return (s==null)?"—":Math.round(s*100)+"%"; }

function renderCards(){
  const total = DATA.length;
  const applied = DATA.filter(d=>d.applied_status==="applied").length;
  const pending = total - applied;
  const scored = DATA.filter(d=>d.fit_score!=null);
  const avg = scored.length ? scored.reduce((a,d)=>a+d.fit_score,0)/scored.length : 0;
  const strong = DATA.filter(d=>(d.fit_score||0)>=0.7).length;
  const weekAgo = Date.now() - 7*86400000;
  const fresh = DATA.filter(d=>{const t=Date.parse(d.date_posted); return t && t>=weekAgo;}).length;

  const cards = [
    ["Total jobs", total],
    ["Applied", applied],
    ["Pending", pending],
    ["Avg match", Math.round(avg*100)+"%"],
    ["Strong (≥70%)", strong],
    ["Posted this week", fresh],
  ];
  document.getElementById("cards").innerHTML = cards.map(
    ([l,n])=>`<div class="card"><div class="num">${n}</div><div class="lbl">${l}</div></div>`
  ).join("");
  document.getElementById("subtitle").textContent =
    `${total} jobs tracked · ${applied} applied · ${pending} pending`;
}

function barChart(title, rows, color){
  const max = Math.max(1, ...rows.map(r=>r[1]));
  const bars = rows.map(([name,val])=>`
    <div class="bar-row">
      <span class="name" title="${name}">${name}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(val/max)*100}%;background:${color||'var(--accent)'}"></span></span>
      <span class="val">${val}</span>
    </div>`).join("");
  return `<div class="chart"><h3>${title}</h3>${bars||'<div class="muted">No data</div>'}</div>`;
}

function renderCharts(){
  // Applied vs pending
  const applied = DATA.filter(d=>d.applied_status==="applied").length;
  const pending = DATA.length - applied;
  const statusChart = barChart("Application status",
    [["Applied",applied],["Pending",pending]], "var(--green)");

  // Score distribution
  const strong = DATA.filter(d=>(d.fit_score||0)>=0.7).length;
  const medium = DATA.filter(d=>(d.fit_score||0)>=0.5 && (d.fit_score||0)<0.7).length;
  const weak   = DATA.filter(d=>(d.fit_score||0)<0.5).length;
  const scoreChart = barChart("Match-score distribution",
    [["Strong ≥70%",strong],["Medium 50-69%",medium],["Weak <50%",weak]], "var(--amber)");

  // Top companies
  const byCo = {};
  DATA.forEach(d=>{byCo[d.company]=(byCo[d.company]||0)+1;});
  const topCo = Object.entries(byCo).sort((a,b)=>b[1]-a[1]).slice(0,8);
  const coChart = barChart("Top companies", topCo);

  // Pipeline status
  const byPs = {};
  DATA.forEach(d=>{const k=d.pipeline_status||"—";byPs[k]=(byPs[k]||0)+1;});
  const psChart = barChart("Pipeline status",
    Object.entries(byPs).sort((a,b)=>b[1]-a[1]), "var(--accent)");

  document.getElementById("charts").innerHTML = statusChart+scoreChart+coChart+psChart;
}

function currentRows(){
  const q = document.getElementById("search").value.toLowerCase();
  const fs = document.getElementById("filterStatus").value;
  const sc = parseFloat(document.getElementById("filterScore").value);
  let rows = DATA.filter(d=>{
    if(fs!=="all" && d.applied_status!==fs) return false;
    if((d.fit_score||0) < sc) return false;
    if(q && !((d.company||"").toLowerCase().includes(q) ||
              (d.role_title||"").toLowerCase().includes(q))) return false;
    return true;
  });
  rows.sort((a,b)=>{
    let x=a[sortKey], y=b[sortKey];
    if(x==null)x=""; if(y==null)y="";
    if(typeof x==="number"||typeof y==="number") return (x-y)*sortDir;
    return String(x).localeCompare(String(y))*sortDir;
  });
  return rows;
}

function renderTable(){
  const rows = currentRows();
  document.getElementById("count").textContent = rows.length+" shown";
  const tbody = document.getElementById("tbody");
  document.getElementById("empty").style.display = rows.length?"none":"block";

  tbody.innerHTML = rows.map(d=>{
    const link = d.role_url
      ? `<a class="link" href="${d.role_url}" target="_blank" rel="noopener">Open ↗</a>`
      : `<span class="muted">—</span>`;
    const docs = [fileLink(d.cover_letter_path,"CL"), fileLink(d.resume_path,"Resume")]
      .filter(Boolean).join(" · ") || `<span class="muted">—</span>`;
    const contact = d.spoc_email
      ? `<a class="link" href="mailto:${esc(d.spoc_email)}">${esc(d.spoc_email)}</a>`
      : `<span class="muted">—</span>`;
    const sc = d.fit_score;
    return `<tr>
      <td class="company">${esc(d.company)}</td>
      <td><div class="role">${esc(d.role_title)}</div></td>
      <td class="muted">${fmtDate(d.date_posted)}</td>
      <td class="score ${scoreClass(sc||0)}">${fmtScore(sc)}</td>
      <td><span class="pill">${esc(d.pipeline_status||"—")}</span></td>
      <td>
        <select class="status ${d.applied_status}" data-id="${d.id}" onchange="onStatus(this)">
          <option value="pending" ${d.applied_status==="pending"?"selected":""}>Pending</option>
          <option value="applied" ${d.applied_status==="applied"?"selected":""}>Applied</option>
        </select>
      </td>
      <td>${contact}</td>
      <td>${docs}</td>
      <td><input class="notes" data-id="${d.id}" value="${esc(d.notes||"")}"
                 placeholder="add note…" onchange="onNotes(this)"></td>
      <td>${link}</td>
    </tr>`;
  }).join("");
}

function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function fileLink(path,label){return path?`<a class="link" href="/file?path=${encodeURIComponent(path)}" target="_blank" rel="noopener">${label}</a>`:"";}

async function save(id, field, value){
  await fetch("/api/update",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({id, field, value})});
  const local = DATA.find(d=>d.id===id); if(local) local[field]=value;
  const ind = document.getElementById("saveIndicator");
  ind.classList.add("show"); setTimeout(()=>ind.classList.remove("show"), 1200);
}
function onStatus(el){
  el.className = "status "+el.value;
  save(parseInt(el.dataset.id), "applied_status", el.value).then(()=>{renderCards();renderCharts();});
}
function onNotes(el){ save(parseInt(el.dataset.id), "notes", el.value); }

document.querySelectorAll("th[data-sort]").forEach(th=>{
  th.addEventListener("click",()=>{
    const k=th.dataset.sort;
    if(sortKey===k) sortDir*=-1; else {sortKey=k; sortDir = (k==="fit_score")?-1:1;}
    renderTable();
  });
});
["search","filterStatus","filterScore"].forEach(id=>
  document.getElementById(id).addEventListener("input", renderTable));

document.getElementById("syncBtn").addEventListener("click", async ()=>{
  const btn=document.getElementById("syncBtn"); btn.textContent="Syncing…"; btn.disabled=true;
  const res = await fetch("/api/sync",{method:"POST"});
  const r = await res.json();
  btn.textContent="⟳ Sync tracker.json"; btn.disabled=false;
  alert(`Synced: ${r.added} new, ${r.updated} refreshed, ${r.total} total.`);
  load();
});

load();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _serve_file(self):
        """Serve a generated doc (cover letter / resume) — restricted to OUTPUT_DIR."""
        from urllib.parse import parse_qs, unquote
        qs = parse_qs(urlparse(self.path).query)
        raw = unquote((qs.get("path") or [""])[0])
        if not raw:
            return self._json(400, {"error": "missing path"})
        try:
            target = Path(raw).resolve()
            # Path-traversal guard: only files inside the applications dir.
            if OUTPUT_DIR not in target.parents or not target.is_file():
                return self._json(404, {"error": "not found"})
            body = target.read_bytes()
        except (OSError, ValueError):
            return self._json(404, {"error": "not found"})
        self._send(200, body, "text/markdown; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/applications":
            self._json(200, fetch_all())
        elif path == "/file":
            self._serve_file()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        if path == "/api/update":
            ok = update_field(int(payload.get("id", 0)),
                              payload.get("field", ""), payload.get("value"))
            return self._json(200 if ok else 400, {"ok": ok})
        if path == "/api/sync":
            return self._json(200, sync_from_json())
        self._json(404, {"error": "not found"})

    def log_message(self, *args):  # quieter console
        pass


def main():
    parser = argparse.ArgumentParser(description="Job application tracker dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    init_db()
    result = sync_from_json()
    if result.get("error"):
        print(f"⚠  {result['error']} — starting with an empty database.")
    else:
        print(f"Imported tracker.json: {result['added']} new, "
              f"{result['updated']} refreshed, {result['total']} total.")

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Dashboard running at {url}")
    print(f"  Database: {DB_PATH}")
    print("  Press Ctrl+C to stop.\n")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
