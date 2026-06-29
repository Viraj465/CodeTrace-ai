"""
CodeTrace Architecture Visualization — HTML Template.

Generates a self-contained HTML file using D3.js that renders
an interactive folder-level dependency map of the codebase.

Layer 1: Force-directed graph of folders with cross-folder call edges.
Layer 3: Click-to-open sidebar showing files, symbols, and connections.
"""
import json


# Shared type-color palette (matches the JavaScript side)
TYPE_COLORS = {
    "function": "#22d3ee",
    "class":    "#fb923c",
    "method":   "#34d399",
    "module":   "#a78bfa",
    "variable": "#fbbf24",
    "unknown":  "#64748b",
}


def render(data: dict) -> str:
    """
    Render the visualization HTML with the given architecture data.

    :param data: dict with keys projectName, folders, folderEdges, files,
                 totalSymbols, totalFiles, totalFolders, totalConnections
    :returns: Complete HTML string (utf-8 safe, self-contained)
    """
    json_str = json.dumps(data, ensure_ascii=False)
    # Prevent accidental closing script tags inside the JSON payload.
    json_str = json_str.replace("</", "<\\/")

    html = _TEMPLATE
    html = html.replace("__DATA_JSON__", json_str)
    html = html.replace("__PROJECT_NAME__", data.get("projectName", "Project"))
    return html


# HTML Template

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CodeTrace — __PROJECT_NAME__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Reset & Design Tokens */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0e1a;--surface:#131825;--card:#1a2035;
  --border:#1e293b;--border-h:#334155;
  --text:#e2e8f0;--dim:#94a3b8;--muted:#475569;
  --accent:#22d3ee;--accent-bg:rgba(34,211,238,.12);
  --fn:#22d3ee;--cls:#fb923c;--meth:#34d399;--mod:#a78bfa;--var:#fbbf24;
  --font:'Inter',system-ui,sans-serif;
  --mono:'JetBrains Mono','Consolas',monospace;
  --sidebar-w:380px;--topbar-h:52px;
}
html,body{width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:var(--font)}

/* Top Bar */
#topbar{
  position:fixed;top:0;left:0;right:0;height:var(--topbar-h);
  display:flex;align-items:center;gap:24px;padding:0 24px;
  background:rgba(10,14,26,.88);backdrop-filter:blur(14px);
  border-bottom:1px solid var(--border);z-index:100;
}
.brand{font-weight:700;font-size:15px;letter-spacing:-.3px;color:var(--accent);user-select:none}
.brand span{color:var(--dim);font-weight:400}
.stats{display:flex;gap:20px;font-size:13px;color:var(--dim)}
.stat-val{color:var(--text);font-weight:600;margin-right:4px}
.search-box{margin-left:auto;position:relative}
.search-box input{
  width:260px;padding:7px 12px 7px 34px;
  background:var(--surface);border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-size:13px;font-family:var(--font);outline:none;
  transition:border-color .2s,box-shadow .2s;
}
.search-box input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-bg)}
.search-box input::placeholder{color:var(--muted)}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--muted);pointer-events:none}

/* Canvas */
#canvas{position:fixed;top:var(--topbar-h);left:0;right:0;bottom:0}
#canvas svg{
  width:100%;height:100%;display:block;
  background:
    radial-gradient(ellipse at 50% 45%,rgba(34,211,238,.025) 0%,transparent 65%),
    linear-gradient(rgba(30,41,59,.25) 1px,transparent 1px),
    linear-gradient(90deg,rgba(30,41,59,.25) 1px,transparent 1px);
  background-size:100% 100%,48px 48px,48px 48px;
}

/* Legend */
#legend{
  position:fixed;bottom:20px;left:20px;
  display:flex;gap:16px;padding:10px 18px;
  background:rgba(19,24,37,.92);backdrop-filter:blur(10px);
  border:1px solid var(--border);border-radius:10px;
  font-size:12px;color:var(--dim);z-index:90;user-select:none;
}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-dot{width:10px;height:10px;border-radius:50%}

/* Empty State */
.empty-msg{
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;color:var(--dim);font-size:14px;line-height:1.8;
}
.empty-msg .icon{font-size:48px;margin-bottom:12px;opacity:.4}

/* Sidebar */
#sidebar{
  position:fixed;top:var(--topbar-h);right:0;bottom:0;width:var(--sidebar-w);
  background:rgba(19,24,37,.96);backdrop-filter:blur(20px);
  border-left:1px solid var(--border);
  transform:translateX(100%);transition:transform .32s cubic-bezier(.4,0,.2,1);
  z-index:95;overflow-y:auto;overflow-x:hidden;
}
#sidebar.open{transform:translateX(0)}

.sb-header{
  padding:20px 20px 16px;border-bottom:1px solid var(--border);
  position:sticky;top:0;background:rgba(19,24,37,.98);backdrop-filter:blur(20px);z-index:1;
}
.sb-close{
  position:absolute;top:14px;right:14px;
  background:none;border:none;color:var(--dim);cursor:pointer;font-size:20px;
  width:30px;height:30px;border-radius:6px;display:flex;align-items:center;justify-content:center;
  transition:background .15s,color .15s;
}
.sb-close:hover{background:var(--border);color:var(--text)}
.sb-title{font-size:18px;font-weight:700;margin-bottom:2px;padding-right:36px}
.sb-subtitle{font-size:13px;color:var(--dim)}
.sb-type-bar{display:flex;gap:14px;margin-top:14px;flex-wrap:wrap}
.sb-type-chip{
  display:flex;align-items:center;gap:5px;font-size:12px;
  padding:3px 10px;border-radius:6px;background:rgba(255,255,255,.04);
}
.sb-type-chip .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.sb-type-chip .val{font-weight:600;color:var(--text)}

.sb-section{padding:14px 20px;border-bottom:1px solid var(--border)}
.sb-section-title{
  font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.9px;
  color:var(--muted);margin-bottom:10px;
}

/* Connections */
.sb-conn{
  padding:8px 10px;margin-bottom:4px;border-radius:8px;font-size:13px;
  display:flex;align-items:center;gap:6px;cursor:pointer;
  transition:background .12s;
}
.sb-conn:hover{background:rgba(255,255,255,.04)}
.sb-conn-folder{font-weight:500}
.sb-conn-arrow{color:var(--accent);font-size:12px;flex-shrink:0}
.sb-conn-badge{
  margin-left:auto;background:var(--accent-bg);color:var(--accent);
  padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;flex-shrink:0;
}
.sb-conn-details{
  display:none;padding:6px 0 2px 12px;font-size:11px;font-family:var(--mono);
  color:var(--dim);line-height:1.8;
}
.sb-conn.expanded .sb-conn-details{display:block}
.sb-conn-call{display:flex;align-items:center;gap:4px}
.sb-conn-call .arr{color:var(--accent);font-size:10px}

/* Files */
.sb-file{
  padding:8px 12px;margin-bottom:3px;border-radius:8px;cursor:pointer;
  transition:background .12s;
}
.sb-file:hover{background:rgba(255,255,255,.04)}
.sb-file-name{font-family:var(--mono);font-size:13px;font-weight:500}
.sb-file-meta{font-size:11px;color:var(--muted);margin-top:1px}
.sb-symbols{padding-left:8px;margin-top:6px;display:none}
.sb-file.expanded .sb-symbols{display:block}
.sb-symbol{
  padding:3px 6px;font-size:12px;font-family:var(--mono);
  display:flex;align-items:center;gap:6px;
  border-radius:4px;margin-bottom:1px;
}
.sb-symbol:hover{background:rgba(255,255,255,.03)}
.sb-sym-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.sb-sym-type{color:var(--muted);margin-left:auto;font-size:10px}

/* Tooltip */
#tooltip{
  position:fixed;pointer-events:none;z-index:200;
  padding:10px 14px;border-radius:10px;font-size:12px;
  background:rgba(26,32,53,.96);border:1px solid var(--border);
  backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,.5);
  opacity:0;transition:opacity .12s;max-width:340px;line-height:1.6;
}
#tooltip.visible{opacity:1}
.tt-title{font-weight:700;font-size:13px;margin-bottom:4px}
.tt-row{color:var(--dim)}
.tt-val{color:var(--text);font-weight:500}
.tt-break{color:var(--dim);font-size:11px;margin-top:6px;line-height:1.7}

/* Scrollbar */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--border-h)}
</style>
</head>
<body>

<!-- Top Bar -->
<div id="topbar">
  <div class="brand">CodeTrace <span>Architecture</span></div>
  <div class="stats" id="stats"></div>
  <div class="search-box">
    <svg class="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
      <circle cx="11" cy="11" r="7"/><path d="M21 21l-4.35-4.35"/>
    </svg>
    <input type="text" id="search" placeholder="Search symbols..." autocomplete="off" spellcheck="false">
  </div>
</div>

<!-- Canvas -->
<div id="canvas"></div>

<!-- Legend -->
<div id="legend">
  <div class="legend-item"><div class="legend-dot" style="background:var(--fn)"></div>function</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--cls)"></div>class</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--meth)"></div>method</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--mod)"></div>module</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--var)"></div>variable</div>
</div>

<!-- Sidebar -->
<div id="sidebar">
  <div class="sb-header">
    <button class="sb-close" id="sb-close" title="Close">&times;</button>
    <div class="sb-title" id="sb-title"></div>
    <div class="sb-subtitle" id="sb-subtitle"></div>
    <div class="sb-type-bar" id="sb-types"></div>
  </div>
  <div id="sb-body"></div>
</div>

<!-- Tooltip -->
<div id="tooltip"></div>

<!-- D3.js -->
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
/* CodeTrace Architecture: Folder Dependency Map */
(function () {
"use strict";

/* 1. Data and Constants */
var DATA = __DATA_JSON__;

var TYPE_COLORS = {
  "function":"#22d3ee","class":"#fb923c","method":"#34d399",
  "module":"#a78bfa","variable":"#fbbf24","unknown":"#64748b"
};

var folders  = DATA.folders;
var fEdges   = DATA.folderEdges;
var files    = DATA.files;

/* 2. Compute derived properties */
var maxSym = Math.max.apply(null, folders.map(function(f){return f.symbolCount})) || 1;

folders.forEach(function(f) {
  f.radius = 28 + 48 * Math.sqrt(f.symbolCount / maxSym);
  f.color  = TYPE_COLORS[f.dominantType] || TYPE_COLORS.unknown;
});

var maxW = Math.max.apply(null, fEdges.map(function(e){return e.weight})) || 1;

// Mark bidirectional pairs so edges curve apart.
var edgeKeys = {};
fEdges.forEach(function(e) { edgeKeys[e.source + "|" + e.target] = true; });
fEdges.forEach(function(e) {
  e.lineWidth = 1.5 + 5 * (e.weight / maxW);
  e.bidir = !!edgeKeys[e.target + "|" + e.source];
});

/* 3. Stats Bar */
document.getElementById("stats").innerHTML =
  '<div><span class="stat-val">' + DATA.totalFolders  + '</span>folders</div>' +
  '<div><span class="stat-val">' + DATA.totalFiles    + '</span>files</div>' +
  '<div><span class="stat-val">' + DATA.totalSymbols  + '</span>symbols</div>' +
  '<div><span class="stat-val">' + DATA.totalConnections + '</span>cross-folder calls</div>';

/* 4. SVG Setup */
var container = document.getElementById("canvas");
var W = container.clientWidth;
var H = container.clientHeight;

var svg = d3.select("#canvas").append("svg");
var defs = svg.append("defs");

// Glow filter
var glow = defs.append("filter").attr("id","glow")
  .attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
glow.append("feGaussianBlur").attr("in","SourceGraphic").attr("stdDeviation","6").attr("result","b");
glow.append("feMerge").selectAll("feMergeNode")
  .data(["b","SourceGraphic"]).enter().append("feMergeNode").attr("in",function(d){return d});

// Arrow marker
defs.append("marker").attr("id","arrow")
  .attr("viewBox","0 -5 10 10").attr("refX",10).attr("refY",0)
  .attr("markerWidth",10).attr("markerHeight",10)
  .attr("markerUnits","userSpaceOnUse").attr("orient","auto")
  .append("path").attr("d","M0,-4L10,0L0,4").attr("fill","rgba(34,211,238,0.55)");

// Highlighted arrow
defs.append("marker").attr("id","arrow-hl")
  .attr("viewBox","0 -5 10 10").attr("refX",10).attr("refY",0)
  .attr("markerWidth",10).attr("markerHeight",10)
  .attr("markerUnits","userSpaceOnUse").attr("orient","auto")
  .append("path").attr("d","M0,-4L10,0L0,4").attr("fill","rgba(34,211,238,0.95)");

// Main group (zoomable)
var g = svg.append("g");
var zoomBehavior = d3.zoom().scaleExtent([0.15, 4])
  .on("zoom", function(ev) { g.attr("transform", ev.transform); });
svg.call(zoomBehavior);

/* 5. Force Simulation */
var simulation = d3.forceSimulation(folders)
  .force("charge",    d3.forceManyBody().strength(-700))
  .force("center",    d3.forceCenter(W / 2, H / 2))
  .force("collision", d3.forceCollide().radius(function(d){return d.radius + 28}))
  .force("link",      d3.forceLink(fEdges).id(function(d){return d.id}).distance(240).strength(0.4))
  .alphaDecay(0.028)
  .on("tick", ticked);

/* 6. Render Edges */
var edgeG = g.append("g").attr("class","edges");

var edgePaths = edgeG.selectAll("path").data(fEdges).enter().append("path")
  .attr("fill","none")
  .attr("stroke","rgba(34,211,238,0.22)")
  .attr("stroke-width",function(d){return d.lineWidth})
  .attr("marker-end","url(#arrow)")
  .style("cursor","pointer");

var edgeLabels = edgeG.selectAll("text").data(fEdges).enter().append("text")
  .text(function(d){return d.weight})
  .attr("font-size",11).attr("font-family","'JetBrains Mono',monospace")
  .attr("font-weight",600).attr("fill","rgba(34,211,238,0.5)")
  .attr("text-anchor","middle").attr("dy",-8)
  .style("pointer-events","none");

/* 7. Render Nodes */
var nodeG = g.append("g").attr("class","nodes");

var nodeEls = nodeG.selectAll("g").data(folders).enter().append("g")
  .style("cursor","pointer")
  .call(d3.drag()
    .on("start", function(ev,d){if(!ev.active)simulation.alphaTarget(0.25).restart();d.fx=d.x;d.fy=d.y})
    .on("drag",  function(ev,d){d.fx=ev.x;d.fy=ev.y})
    .on("end",   function(ev,d){if(!ev.active)simulation.alphaTarget(0);d.fx=null;d.fy=null})
  );

// Outer ring (type composition)
nodeEls.each(function(d) {
  var types = d.types || {};
  var entries = Object.keys(types).map(function(k){return {type:k,count:types[k]}})
    .sort(function(a,b){return b.count - a.count});
  var total = d.symbolCount || 1;
  if (entries.length <= 1) return; // Single-type folders do not need a ring.

  var arcGen = d3.arc().innerRadius(d.radius - 1).outerRadius(d.radius + 3);
  var pieGen = d3.pie().sort(null).value(function(e){return e.count});
  var arcs = pieGen(entries);

  d3.select(this).selectAll("path.ring").data(arcs).enter().append("path")
    .attr("class","ring")
    .attr("d", arcGen)
    .attr("fill", function(a){return TYPE_COLORS[a.data.type] || TYPE_COLORS.unknown})
    .attr("opacity", 0.7);
});

// Main circle
nodeEls.append("circle")
  .attr("r", function(d){return d.radius})
  .attr("fill", function(d){
    return d.color.replace(")", ",0.12)").replace("rgb","rgba");
  })
  .attr("stroke", function(d){return d.color})
  .attr("stroke-width", 2)
  .attr("stroke-opacity", 0.55);

// Folder name
nodeEls.append("text")
  .text(function(d){
    var n = d.name;
    // Truncate long names.
    return n.length > 18 ? n.slice(0,16) + "\u2026" : n;
  })
  .attr("text-anchor","middle").attr("dy", -4)
  .attr("font-size", function(d){return Math.max(11, Math.min(15, d.radius / 3.2))})
  .attr("font-weight", 600).attr("fill","#e2e8f0")
  .style("pointer-events","none");

// Symbol count
nodeEls.append("text")
  .text(function(d){return d.symbolCount + " sym \u00B7 " + d.fileCount + " files"})
  .attr("text-anchor","middle").attr("dy", 14)
  .attr("font-size", 10).attr("fill","#64748b")
  .style("pointer-events","none");

/* 8. Edge Path Computation */
function linkPath(d) {
  var sx = d.source.x, sy = d.source.y;
  var tx = d.target.x, ty = d.target.y;
  var dx = tx - sx, dy = ty - sy;
  var dist = Math.sqrt(dx*dx + dy*dy) || 1;

  // Shorten to circle edges.
  var sr = (typeof d.source.radius === "number" ? d.source.radius : 30) + 4;
  var tr = (typeof d.target.radius === "number" ? d.target.radius : 30) + 14;
  var x1 = sx + (dx/dist)*sr, y1 = sy + (dy/dist)*sr;
  var x2 = tx - (dx/dist)*tr, y2 = ty - (dy/dist)*tr;

  if (!d.bidir) {
    // Straight line for unidirectional edges.
    return "M"+x1+","+y1+"L"+x2+","+y2;
  }
  // Curved for bidirectional edges.
  var nx = -dy/dist, ny = dx/dist;
  var cx = (x1+x2)/2 + nx*38;
  var cy = (y1+y2)/2 + ny*38;
  return "M"+x1+","+y1+"Q"+cx+","+cy+","+x2+","+y2;
}

function labelPos(d) {
  var sx = d.source.x, sy = d.source.y;
  var tx = d.target.x, ty = d.target.y;
  var mx = (sx+tx)/2, my = (sy+ty)/2;
  if (d.bidir) {
    var dx = tx-sx, dy = ty-sy;
    var dist = Math.sqrt(dx*dx+dy*dy) || 1;
    mx += (-dy/dist)*20;
    my += (dx/dist)*20;
  }
  return {x:mx, y:my};
}

/* 9. Tick */
function ticked() {
  edgePaths.attr("d", linkPath);
  edgeLabels.each(function(d) {
    var p = labelPos(d);
    d3.select(this).attr("x",p.x).attr("y",p.y);
  });
  nodeEls.attr("transform",function(d){return "translate("+d.x+","+d.y+")"});
}

/* 10. Hover Events for Nodes */
var tooltipEl = document.getElementById("tooltip");

nodeEls.on("mouseover", function(ev, d) {
  // Apply glow effect to the hovered node.
  d3.select(this).select("circle").attr("stroke-width",3).attr("stroke-opacity",1)
    .attr("filter","url(#glow)");

  // Find connected folders.
  var conn = {};
  conn[d.id] = true;
  fEdges.forEach(function(e){
    var sid = typeof e.source==="object"?e.source.id:e.source;
    var tid = typeof e.target==="object"?e.target.id:e.target;
    if(sid===d.id) conn[tid]=true;
    if(tid===d.id) conn[sid]=true;
  });

  // Dim unconnected nodes.
  nodeEls.transition().duration(120).style("opacity",function(n){return conn[n.id]?1:0.12});
  edgePaths.transition().duration(120)
    .attr("stroke",function(e){
      var sid=typeof e.source==="object"?e.source.id:e.source;
      var tid=typeof e.target==="object"?e.target.id:e.target;
      return(sid===d.id||tid===d.id)?"rgba(34,211,238,0.8)":"rgba(34,211,238,0.04)";
    })
    .attr("marker-end",function(e){
      var sid=typeof e.source==="object"?e.source.id:e.source;
      var tid=typeof e.target==="object"?e.target.id:e.target;
      return(sid===d.id||tid===d.id)?"url(#arrow-hl)":"url(#arrow)";
    });
  edgeLabels.transition().duration(120).style("opacity",function(e){
    var sid=typeof e.source==="object"?e.source.id:e.source;
    var tid=typeof e.target==="object"?e.target.id:e.target;
    return(sid===d.id||tid===d.id)?1:0.04;
  });

  // Tooltip
  var types = d.types || {};
  var breakdown = Object.keys(types).sort(function(a,b){return types[b]-types[a]})
    .map(function(t){
      return '<span style="color:'+(TYPE_COLORS[t]||TYPE_COLORS.unknown)+'">&#9679;</span> '+types[t]+' '+t+(types[t]>1?'s':'');
    }).join("<br>");
  tooltipEl.innerHTML = '<div class="tt-title">'+d.name+'/</div>'
    +'<div class="tt-row"><span class="tt-val">'+d.fileCount+'</span> files &middot; <span class="tt-val">'+d.symbolCount+'</span> symbols</div>'
    +'<div class="tt-break">'+breakdown+'</div>';
  tooltipEl.classList.add("visible");
  positionTooltip(ev);
})
.on("mousemove", positionTooltip)
.on("mouseout", function() {
  d3.select(this).select("circle").attr("stroke-width",2).attr("stroke-opacity",0.55).attr("filter",null);
  nodeEls.transition().duration(180).style("opacity",1);
  edgePaths.transition().duration(180).attr("stroke","rgba(34,211,238,0.22)").attr("marker-end","url(#arrow)");
  edgeLabels.transition().duration(180).style("opacity",1);
  tooltipEl.classList.remove("visible");
});

/* 11. Hover Events for Edges */
edgePaths.on("mouseover", function(ev, d) {
  d3.select(this).attr("stroke","rgba(34,211,238,0.9)").attr("stroke-width",d.lineWidth+2)
    .attr("marker-end","url(#arrow-hl)");

  var calls = d.calls || [];
  var srcId = typeof d.source==="object"?d.source.id:d.source;
  var tgtId = typeof d.target==="object"?d.target.id:d.target;
  var list = calls.slice(0,6).map(function(c){
    return c.sourceFile+':<b>'+c.sourceName+'</b> <span style="color:#22d3ee">\u2192</span> '+c.targetFile+':<b>'+c.targetName+'</b>';
  }).join("<br>");
  var more = calls.length>6 ? '<div style="color:var(--muted);margin-top:3px">+'+(calls.length-6)+' more</div>' : '';

  tooltipEl.innerHTML = '<div class="tt-title">'+srcId+' \u2192 '+tgtId+'</div>'
    +'<div class="tt-row"><span class="tt-val">'+d.weight+'</span> calls</div>'
    +'<div class="tt-break" style="font-family:var(--mono);font-size:11px">'+list+'</div>'+more;
  tooltipEl.classList.add("visible");
  positionTooltip(ev);
})
.on("mousemove", positionTooltip)
.on("mouseout", function(ev, d) {
  d3.select(this).attr("stroke","rgba(34,211,238,0.22)").attr("stroke-width",d.lineWidth)
    .attr("marker-end","url(#arrow)");
  tooltipEl.classList.remove("visible");
});

function positionTooltip(ev) {
  var x = ev.clientX + 18, y = ev.clientY - 12;
  // Keep the tooltip within the viewport.
  var tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
  if (x + tw > window.innerWidth - 12) x = ev.clientX - tw - 12;
  if (y + th > window.innerHeight - 12) y = window.innerHeight - th - 12;
  if (y < 60) y = 60;
  tooltipEl.style.left = x + "px";
  tooltipEl.style.top = y + "px";
}

/* 12. Click Events for Sidebar */
var sidebar  = document.getElementById("sidebar");
var sbTitle  = document.getElementById("sb-title");
var sbSub    = document.getElementById("sb-subtitle");
var sbTypes  = document.getElementById("sb-types");
var sbBody   = document.getElementById("sb-body");
var activeFolder = null;

nodeEls.on("click", function(ev, d) {
  ev.stopPropagation();
  openSidebar(d);
});
document.getElementById("sb-close").addEventListener("click", closeSidebar);
svg.on("click", function() { closeSidebar(); });

function openSidebar(folder) {
  activeFolder = folder.id;
  sbTitle.textContent = folder.name + "/";
  sbSub.textContent = folder.fileCount + " files \u00B7 " + folder.symbolCount + " symbols";

  // Type chips
  var types = folder.types || {};
  sbTypes.innerHTML = Object.keys(types).sort(function(a,b){return types[b]-types[a]})
    .map(function(t) {
      return '<div class="sb-type-chip"><div class="dot" style="background:'
        +(TYPE_COLORS[t]||TYPE_COLORS.unknown)+'"></div><span class="val">'
        +types[t]+'</span> '+t+(types[t]>1?'s':'')+'</div>';
    }).join("");

  // Build sections
  var html = "";

  // Connections
  var outgoing = fEdges.filter(function(e){
    return (typeof e.source==="object"?e.source.id:e.source) === folder.id;
  });
  var incoming = fEdges.filter(function(e){
    return (typeof e.target==="object"?e.target.id:e.target) === folder.id;
  });

  if (outgoing.length || incoming.length) {
    html += '<div class="sb-section"><div class="sb-section-title">Dependencies</div>';
    outgoing.forEach(function(e) {
      var tid = typeof e.target==="object"?e.target.id:e.target;
      var callsHtml = (e.calls||[]).map(function(c){
        return '<div class="sb-conn-call">'+c.sourceFile+':'+c.sourceName
          +' <span class="arr">\u2192</span> '+c.targetFile+':'+c.targetName+'</div>';
      }).join("");
      html += '<div class="sb-conn" onclick="this.classList.toggle(\'expanded\')">'
        +'<span class="sb-conn-folder">'+folder.name+'</span>'
        +'<span class="sb-conn-arrow">\u2192</span>'
        +'<span class="sb-conn-folder">'+tid+'</span>'
        +'<span class="sb-conn-badge">'+e.weight+' calls</span>'
        +'<div class="sb-conn-details">'+callsHtml+'</div></div>';
    });
    incoming.forEach(function(e) {
      var sid = typeof e.source==="object"?e.source.id:e.source;
      var callsHtml = (e.calls||[]).map(function(c){
        return '<div class="sb-conn-call">'+c.sourceFile+':'+c.sourceName
          +' <span class="arr">\u2192</span> '+c.targetFile+':'+c.targetName+'</div>';
      }).join("");
      html += '<div class="sb-conn" onclick="this.classList.toggle(\'expanded\')">'
        +'<span class="sb-conn-folder">'+sid+'</span>'
        +'<span class="sb-conn-arrow">\u2192</span>'
        +'<span class="sb-conn-folder">'+folder.name+'</span>'
        +'<span class="sb-conn-badge">'+e.weight+' calls</span>'
        +'<div class="sb-conn-details">'+callsHtml+'</div></div>';
    });
    html += '</div>';
  }

  // Files
  var folderFiles = files.filter(function(f){return f.folder === folder.id});
  if (folderFiles.length) {
    html += '<div class="sb-section"><div class="sb-section-title">Files</div>';
    folderFiles.forEach(function(f) {
      var syms = f.symbols || [];
      var symsHtml = syms.map(function(s) {
        return '<div class="sb-symbol"><div class="sb-sym-dot" style="background:'
          +(TYPE_COLORS[s.type]||TYPE_COLORS.unknown)+'"></div>'
          +'<span>'+s.name+'</span><span class="sb-sym-type">'+s.type+'</span></div>';
      }).join("");
      html += '<div class="sb-file" onclick="this.classList.toggle(\'expanded\')">'
        +'<div class="sb-file-name">'+f.name+'</div>'
        +'<div class="sb-file-meta">'+syms.length+' symbol'+(syms.length!==1?'s':'')+'</div>'
        +'<div class="sb-symbols">'+symsHtml+'</div></div>';
    });
    html += '</div>';
  }

  sbBody.innerHTML = html;
  sidebar.classList.add("open");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  activeFolder = null;
}

/* 13. Search */
var searchInput = document.getElementById("search");
searchInput.addEventListener("input", function() {
  var q = this.value.toLowerCase().trim();
  if (!q) {
    nodeEls.transition().duration(180).style("opacity",1);
    edgePaths.transition().duration(180).attr("stroke","rgba(34,211,238,0.22)");
    edgeLabels.transition().duration(180).style("opacity",1);
    return;
  }

  var matchFolders = {};
  files.forEach(function(f) {
    (f.symbols||[]).forEach(function(s) {
      if (s.name.toLowerCase().indexOf(q) !== -1) matchFolders[f.folder] = true;
    });
  });
  // Match folder names.
  folders.forEach(function(f) {
    if (f.name.toLowerCase().indexOf(q) !== -1) matchFolders[f.id] = true;
  });

  nodeEls.transition().duration(150).style("opacity",function(d){return matchFolders[d.id]?1:0.08});
  edgePaths.transition().duration(150).attr("stroke",function(e){
    var sid=typeof e.source==="object"?e.source.id:e.source;
    var tid=typeof e.target==="object"?e.target.id:e.target;
    return(matchFolders[sid]&&matchFolders[tid])?"rgba(34,211,238,0.6)":"rgba(34,211,238,0.03)";
  });
  edgeLabels.transition().duration(150).style("opacity",function(e){
    var sid=typeof e.source==="object"?e.source.id:e.source;
    var tid=typeof e.target==="object"?e.target.id:e.target;
    return(matchFolders[sid]&&matchFolders[tid])?1:0.03;
  });
});

/* 14. Zoom to Fit */
simulation.on("end", function() {
  var pad = 80;
  var x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  folders.forEach(function(d){
    x0=Math.min(x0,d.x-d.radius); y0=Math.min(y0,d.y-d.radius);
    x1=Math.max(x1,d.x+d.radius); y1=Math.max(y1,d.y+d.radius);
  });
  if (!isFinite(x0)) return;
  var bw=x1-x0+pad*2, bh=y1-y0+pad*2;
  var scale=Math.min(W/bw, H/bh, 1.8);
  var cx=(x0+x1)/2, cy=(y0+y1)/2;
  svg.transition().duration(900).ease(d3.easeCubicOut)
    .call(zoomBehavior.transform,
      d3.zoomIdentity.translate(W/2,H/2).scale(scale).translate(-cx,-cy));
});

/* 15. Handle empty state */
if (folders.length === 0) {
  document.getElementById("canvas").innerHTML =
    '<div class="empty-msg"><div class="icon">&#128194;</div>'
    +'No indexed symbols found.<br>Run <code>codetrace index .</code> first.</div>';
}

})();
</script>
</body>
</html>
"""
