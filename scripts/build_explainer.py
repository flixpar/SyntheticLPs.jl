#!/usr/bin/env python3
"""Build a self-contained HTML explainer for the SyntheticLPs problem generators.

Reads the markdown pages under docs/ and assembles a single static HTML file
(docs/explainer.html) with inlined CSS and an inlined MathJax (SVG) bundle so
the result works fully offline. Run from the repository root:

    python3 scripts/build_explainer.py
"""
import glob
import html
import os
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
MATHJAX = "/tmp/mathjax-tex-svg.js"
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
OUT = os.path.join(DOCS, "explainer.html")


def load_mathjax():
    """Return the MathJax tex-svg bundle, downloading it once if absent."""
    if not os.path.exists(MATHJAX):
        print(f"Downloading MathJax bundle to {MATHJAX} ...")
        urllib.request.urlretrieve(MATHJAX_URL, MATHJAX)
    return open(MATHJAX).read()

# --- Curated metadata: family grouping + headline tags ------------------------
# family key -> (label, short blurb)
FAMILIES = {
    "network": ("Network & Routing", "Flows, multicommodity routing, and capacity sharing over graphs."),
    "facility": ("Facility & Supply Chain", "Where to open capacity and how to serve demand from it."),
    "blending": ("Blending & Diet", "Mix ingredients to hit composition targets at least cost."),
    "production": ("Production & Planning", "Allocate shared capacity across products and time."),
    "scheduling": ("Assignment & Scheduling", "Match discrete entities to tasks, shifts, or patterns."),
    "selection": ("Selection & Finance", "Pick a subset / weighting under budget and risk limits."),
    "land": ("Land & Agriculture", "Allocate parcels and acreage under physical limits."),
}

# file stem -> dict(family, sense, vclass, tagline)
META = {
    "transportation":        dict(family="network",   sense="Min",     vclass="Continuous", tag="Min-cost shipping over supply/demand lanes"),
    "network_flow":          dict(family="network",   sense="Min/Max", vclass="Continuous", tag="Single-commodity max-flow or min-cost flow"),
    "multi_commodity_flow":  dict(family="network",   sense="Min",     vclass="Continuous", tag="Several commodities sharing arc capacities"),
    "load_balancing":        dict(family="network",   sense="Min-max", vclass="Continuous", tag="Minimize the most-utilized link"),
    "telecom_network_design":dict(family="network",   sense="Min",     vclass="Mixed",      tag="Install links + route traffic under a budget"),
    "facility_location":     dict(family="facility",  sense="Min",     vclass="Mixed",      tag="Open capacitated facilities, then serve demand"),
    "supply_chain":          dict(family="facility",  sense="Min",     vclass="Mixed",      tag="Open facilities + multi-mode shipping"),
    "blending":              dict(family="blending",  sense="Min",     vclass="Continuous", tag="Hit quality bands at minimum ingredient cost"),
    "feed_blending":         dict(family="blending",  sense="Min",     vclass="Continuous", tag="Feed formulation with nutrient floors/caps"),
    "diet_problem":          dict(family="blending",  sense="Min",     vclass="Continuous", tag="Classic min-cost diet with nutrient bounds"),
    "production_planning":   dict(family="production",sense="Max",     vclass="Continuous", tag="Profit-max under shared resource capacities"),
    "product_mix":           dict(family="production",sense="Max",     vclass="Continuous", tag="Profit-max with market lower/upper bounds"),
    "resource_allocation":   dict(family="production",sense="Max",     vclass="Continuous", tag="Allocate scarce resources among activities"),
    "inventory":             dict(family="production",sense="Min",     vclass="Continuous", tag="Single-item lot sizing over a horizon"),
    "energy":                dict(family="production",sense="Min",     vclass="Continuous", tag="Dispatch generators over time to meet demand"),
    "assignment":            dict(family="scheduling",sense="Min",     vclass="Binary",     tag="Worker-task matching with compatibility"),
    "scheduling":            dict(family="scheduling",sense="Min",     vclass="Binary",     tag="Workforce shift scheduling with coverage"),
    "airline_crew":          dict(family="scheduling",sense="Min",     vclass="Binary",     tag="Set-partitioning crew pairing"),
    "cutting_stock":         dict(family="scheduling",sense="Min",     vclass="Integer",    tag="One-dimensional pattern cutting"),
    "knapsack":              dict(family="selection", sense="Max",     vclass="Continuous", tag="Fractional knapsack under a capacity"),
    "project_selection":     dict(family="selection", sense="Max",     vclass="Binary",     tag="Pick projects under budget/risk/dependency"),
    "portfolio":             dict(family="selection", sense="Max",     vclass="Continuous", tag="CVaR portfolio with policy constraints"),
    "land_use":              dict(family="land",      sense="Max",     vclass="Binary",     tag="Assign parcels to zoning types"),
    "crop_planning":         dict(family="land",      sense="Max",     vclass="Continuous", tag="Allocate acreage across crops"),
}

FAMILY_ORDER = ["network", "facility", "blending", "production", "scheduling", "selection", "land"]

SECTION_ICONS = {
    "Overview": "◆",
    "Generator Data and Sizing": "▦",
    "LP Formulation": "∑",
    "Feasibility Controls": "⚖",
    "Model Characteristics": "▤",
    "Practical Notes": "✎",
}


# --- Minimal markdown -> HTML converter, tailored to these docs ---------------
def inline(text):
    """Convert inline markdown (already plain) to HTML: escape, code spans, links."""
    out = html.escape(text)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def math_block(body):
    safe = body.replace("&", "&amp;").replace("<", "&lt;")
    return f'<div class="math">\\[{safe}\\]</div>'


def render_blocks(lines):
    """Render a list of markdown lines (a section body) into HTML."""
    html_parts = []
    i = 0
    n = len(lines)
    para = []

    def flush_para():
        if para:
            html_parts.append("<p>" + inline(" ".join(para).strip()) + "</p>")
            para.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced blocks
        if stripped == "```math" or stripped == "```text":
            flush_para()
            kind = stripped[3:]
            body = []
            i += 1
            while i < n and lines[i].strip() != "```":
                body.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            content = "\n".join(body)
            if kind == "math":
                html_parts.append(math_block(content.strip()))
            else:
                html_parts.append('<pre class="formula">' + html.escape(content) + "</pre>")
            continue

        # subsection heading
        if stripped.startswith("### "):
            flush_para()
            html_parts.append("<h4>" + inline(stripped[4:]) + "</h4>")
            i += 1
            continue

        # tables
        if stripped.startswith("|"):
            flush_para()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            html_parts.append(render_table(tbl))
            continue

        # unordered list
        if re.match(r"^\s*-\s+", line):
            flush_para()
            items, i = collect_list(lines, i, r"^\s*-\s+")
            html_parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        # ordered list
        if re.match(r"^\s*\d+\.\s+", line):
            flush_para()
            items, i = collect_list(lines, i, r"^\s*\d+\.\s+")
            html_parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        # blank line -> paragraph break
        if stripped == "":
            flush_para()
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    return "\n".join(html_parts)


def collect_list(lines, i, marker):
    """Collect (possibly nested) list items starting at line i matching marker."""
    items = []
    n = len(lines)
    base_indent = len(lines[i]) - len(lines[i].lstrip())
    while i < n:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        m = re.match(marker, line)
        indent = len(line) - len(line.lstrip())
        if m and indent == base_indent:
            content = line[m.end():].strip()
            # gather nested lines (deeper indent) for this item
            sub = []
            i += 1
            while i < n:
                nxt = lines[i]
                if nxt.strip() == "":
                    sub.append(nxt)
                    i += 1
                    continue
                nind = len(nxt) - len(nxt.lstrip())
                if nind > base_indent:
                    sub.append(nxt)
                    i += 1
                else:
                    break
            item_html = inline(content)
            if any(s.strip() for s in sub):
                item_html += render_blocks(sub)
            items.append("<li>" + item_html + "</li>")
        else:
            break
    return items, i


def render_table(rows):
    def cells(r):
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        return parts

    header = cells(rows[0])
    body = rows[2:]  # rows[1] is the |---|---| separator
    thead = "<tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr>"
    trs = []
    for r in body:
        cs = cells(r)
        trs.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cs) + "</tr>")
    return f"<table><thead>{thead}</thead><tbody>{''.join(trs)}</tbody></table>"


def parse_doc(path):
    """Return (title, intro, sections[list of (name, html)])."""
    text = open(path).read()
    lines = text.split("\n")
    title = ""
    intro_lines = []
    sections = []
    cur_name = None
    cur_lines = []
    seen_section = False

    def push():
        if cur_name is not None:
            sections.append((cur_name, render_blocks(cur_lines)))

    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            push()
            cur_name = line[3:].strip()
            cur_lines = []
            seen_section = True
            continue
        if not seen_section:
            if line.strip():
                intro_lines.append(line.strip())
        else:
            cur_lines.append(line)
    push()
    intro = " ".join(intro_lines).strip()
    return title, intro, sections


# --- Page assembly ------------------------------------------------------------
def build():
    docs = {}
    for path in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == "README":
            continue
        docs[stem] = parse_doc(path)

    # sidebar nav
    nav = ['<a class="nav-top" href="#overview" data-target="overview">Overview</a>']
    for fam in FAMILY_ORDER:
        label = FAMILIES[fam][0]
        nav.append(f'<div class="nav-group"><div class="nav-group-label">{html.escape(label)}</div>')
        stems = [s for s in META if META[s]["family"] == fam]
        stems.sort(key=lambda s: docs[s][0])
        for s in stems:
            nav.append(f'<a class="nav-item" href="#{s}" data-target="{s}" data-search="{html.escape(docs[s][0].lower()+" "+META[s]["tag"].lower())}">{html.escape(docs[s][0])}</a>')
        nav.append("</div>")
    nav_html = "\n".join(nav)

    # overview cards grouped by family
    cards_html = []
    for fam in FAMILY_ORDER:
        label, blurb = FAMILIES[fam]
        cards_html.append(f'<div class="fam-block" data-family="{fam}">')
        cards_html.append(f'<div class="fam-head"><span class="fam-dot fam-{fam}"></span><h3>{html.escape(label)}</h3><p>{html.escape(blurb)}</p></div>')
        cards_html.append('<div class="card-grid">')
        stems = [s for s in META if META[s]["family"] == fam]
        stems.sort(key=lambda s: docs[s][0])
        for s in stems:
            title, intro, _ = docs[s]
            m = META[s]
            search = html.escape((title + " " + intro + " " + m["tag"]).lower())
            cards_html.append(f'''<a class="card fam-border-{fam}" href="#{s}" data-target="{s}" data-search="{search}">
  <div class="card-top"><span class="card-title">{html.escape(title)}</span></div>
  <p class="card-tag">{html.escape(m["tag"])}</p>
  <div class="chips">
    <span class="chip chip-sense">{html.escape(m["sense"])}</span>
    <span class="chip chip-{m["vclass"].lower()}">{html.escape(m["vclass"])}</span>
  </div>
</a>''')
        cards_html.append("</div></div>")
    cards_html = "\n".join(cards_html)

    # detail articles
    articles = []
    for fam in FAMILY_ORDER:
        stems = [s for s in META if META[s]["family"] == fam]
        stems.sort(key=lambda s: docs[s][0])
        for s in stems:
            title, intro, sections = docs[s]
            m = META[s]
            fam_label = FAMILIES[fam][0]
            secs = []
            for name, body in sections:
                icon = SECTION_ICONS.get(name, "§")
                secs.append(f'''<section class="doc-section">
  <h3><span class="sec-icon">{icon}</span>{html.escape(name)}</h3>
  <div class="sec-body">{body}</div>
</section>''')
            articles.append(f'''<article id="{s}" class="view article">
  <a class="back" href="#overview" data-target="overview">← All generators</a>
  <header class="art-head fam-border-{fam}">
    <div class="art-eyebrow"><span class="fam-dot fam-{fam}"></span>{html.escape(fam_label)}</div>
    <h2>{html.escape(title)}</h2>
    <p class="art-intro">{inline(intro)}</p>
    <div class="chips">
      <span class="chip chip-sense">Objective: {html.escape(m["sense"])}</span>
      <span class="chip chip-{m["vclass"].lower()}">{html.escape(m["vclass"])} variables</span>
    </div>
  </header>
  {''.join(secs)}
</article>''')
    articles_html = "\n".join(articles)

    mathjax_js = load_mathjax()

    page = TEMPLATE.format(
        nav=nav_html,
        cards=cards_html,
        articles=articles_html,
        css=CSS,
        mathjax=mathjax_js,
        count=len(docs),
    )
    with open(OUT, "w") as f:
        f.write(page)
    print(f"Wrote {OUT} ({len(page)/1024:.0f} KB, {len(docs)} generators)")


CSS = r"""
:root{
  --bg:#f6f4ef; --panel:#fffdf9; --ink:#1c1b22; --muted:#6c6a78; --faint:#928f9e;
  --line:#e7e3da; --line2:#efece4; --accent:#4338ca; --accent-soft:#eceaff;
  --code-bg:#1d1c27; --code-ink:#e8e6f5;
  --net:#2563eb; --fac:#0d9488; --ble:#d97706; --pro:#9333ea; --sch:#dc2626; --sel:#0891b2; --lan:#65a30d;
  --sidebar-w:288px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
code,pre,.math,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,"Cascadia Code",Consolas,monospace}
a{color:var(--accent);text-decoration:none}

/* layout */
.layout{display:flex;min-height:100vh}
.sidebar{
  width:var(--sidebar-w);flex:0 0 var(--sidebar-w);position:sticky;top:0;height:100vh;
  overflow-y:auto;border-right:1px solid var(--line);background:var(--panel);padding:22px 16px 60px;
}
.brand{display:flex;flex-direction:column;gap:2px;padding:6px 10px 16px}
.brand b{font-size:16px;letter-spacing:-.01em}
.brand span{font-size:12px;color:var(--muted)}
.search{width:100%;margin:4px 0 14px;padding:9px 12px;border:1px solid var(--line);border-radius:9px;
  background:var(--bg);font-size:13px;color:var(--ink)}
.search:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.nav-top{display:block;padding:8px 12px;border-radius:8px;font-weight:600;color:var(--ink);font-size:14px}
.nav-top:hover,.nav-item:hover{background:var(--line2)}
.nav-group{margin-top:16px}
.nav-group-label{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--faint);
  font-weight:700;padding:4px 12px;margin-bottom:2px}
.nav-item{display:block;padding:6px 12px;border-radius:7px;color:var(--muted);font-size:13.5px}
.nav-item.active{background:var(--accent-soft);color:var(--accent);font-weight:600}
.nav-top.active{background:var(--accent-soft);color:var(--accent)}
.nav-item.hidden,.card.hidden,.fam-block.hidden{display:none}

.main{flex:1;min-width:0;max-width:920px;margin:0 auto;padding:0 40px 120px}
.view{display:none}
.view.active{display:block;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

/* hero */
.hero{padding:64px 0 30px;border-bottom:1px solid var(--line);margin-bottom:34px}
.hero .kicker{font-size:12.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.hero h1{font-size:42px;line-height:1.08;letter-spacing:-.025em;margin:14px 0 0;font-weight:800}
.hero .lede{font-size:17px;color:var(--muted);max-width:680px;margin:18px 0 0;line-height:1.55}
.contract{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:30px 0 8px}
.contract .c{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 16px}
.contract .c h4{margin:0 0 6px;font-size:13.5px}
.contract .c p{margin:0;font-size:13px;color:var(--muted);line-height:1.5}
.contract .c .tagk{font-family:ui-monospace,monospace;color:var(--accent);font-size:12.5px}
.statuses{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
.status{font-size:12.5px;padding:3px 10px;border-radius:999px;font-weight:600}
.status.f{background:#e6f6ec;color:#157347}
.status.i{background:#fdeaea;color:#c0392b}
.status.u{background:#eef0f4;color:#4a5568}

/* family blocks + cards */
.fam-block{margin:38px 0}
.fam-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin-bottom:14px}
.fam-head h3{margin:0;font-size:19px;letter-spacing:-.01em}
.fam-head p{margin:0;flex:1 1 240px;color:var(--muted);font-size:13.5px}
.fam-dot{width:10px;height:10px;border-radius:3px;display:inline-block;flex:0 0 auto}
.fam-network,.fam-net{background:var(--net)} .fam-facility,.fam-fac{background:var(--fac)}
.fam-blending,.fam-ble{background:var(--ble)} .fam-production,.fam-pro{background:var(--pro)}
.fam-scheduling,.fam-sch{background:var(--sch)} .fam-selection,.fam-sel{background:var(--sel)}
.fam-land,.fam-lan{background:var(--lan)}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line);
  border-radius:12px;padding:15px 16px 14px;display:flex;flex-direction:column;gap:8px;
  transition:transform .12s ease,box-shadow .12s ease,border-color .12s}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 24px -14px rgba(30,25,60,.35)}
.card-title{font-weight:700;font-size:15px;color:var(--ink);letter-spacing:-.01em}
.card-tag{margin:0;font-size:12.5px;color:var(--muted);line-height:1.45;flex:1}
.fam-border-network{border-left-color:var(--net)} .fam-border-facility{border-left-color:var(--fac)}
.fam-border-blending{border-left-color:var(--ble)} .fam-border-production{border-left-color:var(--pro)}
.fam-border-scheduling{border-left-color:var(--sch)} .fam-border-selection{border-left-color:var(--sel)}
.fam-border-land{border-left-color:var(--lan)}

/* chips */
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chip{font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;background:var(--line2);color:var(--muted);
  border:1px solid var(--line)}
.chip-sense{background:#eef1ff;color:#3730a3;border-color:#dfe3ff}
.chip-continuous{background:#e9f6f0;color:#0f7a52;border-color:#d4ede2}
.chip-binary{background:#fdecec;color:#b0322a;border-color:#f6d8d6}
.chip-mixed{background:#f3ecfd;color:#7b35c4;border-color:#e7d8f8}
.chip-integer{background:#fff2e2;color:#b5650c;border-color:#f7e2c6}

/* article */
.article{padding-top:34px}
.back{display:inline-block;font-size:13px;color:var(--muted);margin-bottom:18px;font-weight:600}
.back:hover{color:var(--accent)}
.art-head{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);
  border-radius:14px;padding:24px 26px 22px;margin-bottom:30px}
.art-eyebrow{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:700;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.art-head h2{margin:12px 0 0;font-size:31px;letter-spacing:-.02em;font-weight:800}
.art-intro{margin:12px 0 16px;font-size:16px;color:#454250;line-height:1.55;max-width:680px}

/* doc sections */
.doc-section{margin:30px 0;padding-bottom:6px}
.doc-section h3{display:flex;align-items:center;gap:10px;font-size:18px;letter-spacing:-.01em;
  padding-bottom:9px;border-bottom:1px solid var(--line);margin:0 0 14px}
.sec-icon{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;
  border-radius:7px;background:var(--accent-soft);color:var(--accent);font-size:14px;flex:0 0 auto}
.sec-body h4{font-size:14.5px;margin:20px 0 8px;color:var(--ink)}
.sec-body p{margin:11px 0}
.sec-body ul,.sec-body ol{margin:11px 0;padding-left:22px}
.sec-body li{margin:5px 0}
.sec-body li>ul,.sec-body li>ol{margin:6px 0}

code{background:#eceae3;padding:1.5px 5px;border-radius:5px;font-size:.86em;color:#3a3550}
.art-intro code,p code{white-space:nowrap}

/* formula + math blocks */
pre.formula{background:var(--code-bg);color:var(--code-ink);border-radius:11px;padding:15px 18px;
  overflow-x:auto;font-size:13px;line-height:1.5;margin:14px 0;border:1px solid #2c2a3a;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
.math{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px 16px;
  margin:14px 0;overflow-x:auto;text-align:left}
.math mjx-container{margin:0 !important}

/* tables */
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px;
  background:var(--panel);border:1px solid var(--line);border-radius:11px;overflow:hidden}
thead th{background:#f0ede5;text-align:left;font-weight:700;color:#403c4a}
th,td{padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:#faf8f2}
td code,th code{background:#ece9e1}

/* mobile */
.menu-btn{display:none}
@media(max-width:880px){
  .layout{flex-direction:column}
  .sidebar{position:fixed;left:0;top:0;z-index:40;transform:translateX(-100%);transition:transform .2s;
    box-shadow:0 0 40px rgba(0,0,0,.2);width:280px}
  .sidebar.open{transform:none}
  .main{padding:0 20px 100px;max-width:100%}
  .hero{padding-top:74px}
  .contract{grid-template-columns:1fr}
  .menu-btn{display:flex;position:fixed;top:14px;left:14px;z-index:50;width:42px;height:42px;
    align-items:center;justify-content:center;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;font-size:18px;cursor:pointer}
  .hero h1{font-size:32px}
}
.scrim{display:none}
@media(max-width:880px){.scrim.show{display:block;position:fixed;inset:0;background:rgba(20,18,30,.35);z-index:30}}
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SyntheticLPs · Generator Explainer</title>
<style>{css}</style>
<script>
window.MathJax={{
  tex:{{displayMath:[['\\[','\\]']],inlineMath:[['\\(','\\)']]}},
  svg:{{fontCache:'global'}},
  options:{{enableMenu:false}}
}};
</script>
<script>{mathjax}</script>
</head>
<body>
<button class="menu-btn" id="menuBtn" aria-label="Menu">☰</button>
<div class="scrim" id="scrim"></div>
<div class="layout">
  <aside class="sidebar" id="sidebar">
    <div class="brand"><b>SyntheticLPs</b><span>Generator Explainer · {count} problem types</span></div>
    <input class="search" id="search" type="search" placeholder="Filter generators…" autocomplete="off">
    <nav id="nav">
      {nav}
    </nav>
  </aside>

  <main class="main" id="main">
    <!-- OVERVIEW -->
    <div id="overview" class="view">
      <header class="hero">
        <div class="kicker">Synthetic Linear Programs</div>
        <h1>A Guide to SyntheticLPs.jl</h1>
        <p class="lede">Each generator turns three knobs — a target variable count, a feasibility status, and a
          random seed — into a realistic, reproducible linear program. This page pairs a high-level map of the
          families with the precise formulation, sizing rules, and feasibility tricks behind every generator.</p>

        <div class="contract">
          <div class="c"><h4>The shared contract</h4>
            <p>Call <span class="tagk">generate_problem(sym, target_variables, feasibility_status, seed)</span>.
               All randomness lives in the constructor; <span class="tagk">build_model</span> is deterministic, so a
               seed reproduces an instance exactly.</p></div>
          <div class="c"><h4>Sizing by variables</h4>
            <p><span class="tagk">target_variables</span> is interpreted per generator — usually by choosing
               dimensions whose product or sum approximates the requested count. Ranges and group counts often scale
               with problem size.</p></div>
          <div class="c"><h4>Integer relaxation</h4>
            <p>Several problems are naturally mixed-integer. <span class="tagk">generate_problem</span> defaults to
               <span class="tagk">relax_integer=true</span>, so binary variables are relaxed unless you opt out. Pages
               describe the intended MIP and note the difference.</p></div>
        </div>
        <div class="statuses">
          <span class="status f">feasible — bounds widened so a solution is constructed or likely</span>
          <span class="status i">infeasible — a binding resource is deliberately over-tightened</span>
          <span class="status u">unknown — a realistic random draw with no guarantee</span>
        </div>
      </header>

      <p style="color:var(--muted);font-size:13.5px;margin:0 0 4px">Pick a generator to see its full write-up, or filter from the sidebar.</p>
      {cards}
    </div>

    <!-- ARTICLES -->
    {articles}
  </main>
</div>

<script>
const views=document.querySelectorAll('.view');
const navItems=document.querySelectorAll('[data-target]');
function show(id){{
  if(!document.getElementById(id)) id='overview';
  views.forEach(v=>v.classList.toggle('active',v.id===id));
  document.querySelectorAll('#nav [data-target]').forEach(a=>
    a.classList.toggle('active',a.dataset.target===id));
  window.scrollTo(0,0);
  closeSidebar();
}}
function route(){{ show((location.hash||'#overview').slice(1)); }}
window.addEventListener('hashchange',route);
route();

// sidebar toggle (mobile)
const sb=document.getElementById('sidebar'), scrim=document.getElementById('scrim');
function closeSidebar(){{sb.classList.remove('open');scrim.classList.remove('show');}}
document.getElementById('menuBtn').onclick=()=>{{sb.classList.toggle('open');scrim.classList.toggle('show');}};
scrim.onclick=closeSidebar;

// live filter across cards + sidebar items
const search=document.getElementById('search');
search.addEventListener('input',()=>{{
  const q=search.value.trim().toLowerCase();
  document.querySelectorAll('.card').forEach(c=>
    c.classList.toggle('hidden', q && !c.dataset.search.includes(q)));
  document.querySelectorAll('#nav .nav-item').forEach(a=>
    a.classList.toggle('hidden', q && !a.dataset.search.includes(q)));
  document.querySelectorAll('.fam-block').forEach(b=>{{
    const any=b.querySelectorAll('.card:not(.hidden)').length>0;
    b.classList.toggle('hidden', q && !any);
  }});
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
