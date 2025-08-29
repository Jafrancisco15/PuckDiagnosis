import React, { useCallback, useMemo, useRef, useState, useEffect } from "react";

const MAX_DIM = 1024;
function clamp01(x){ return Math.min(1, Math.max(0, x)); }
function lerp(a,b,t){ return a+(b-a)*t; }
function toGray(r,g,b){ return (0.2126*r + 0.7152*g + 0.0722*b)/255; }

async function fileToImageBitmap(file) {
  const blobURL = URL.createObjectURL(file);
  const img = await createImageBitmap(await fetch(blobURL).then(r=>r.blob()));
  URL.revokeObjectURL(blobURL);
  return img;
}
function drawToCanvas(img, maxDim = MAX_DIM) {
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const cx = canvas.getContext("2d");
  cx.drawImage(img, 0, 0, w, h);
  return { canvas, cx, w, h, scale };
}
function getImageData(canvas) {
  const cx = canvas.getContext("2d");
  return cx.getImageData(0, 0, canvas.width, canvas.height);
}
function grayscale(imgData) {
  const { data, width, height } = imgData;
  const gray = new Float32Array(width * height);
  let sum=0, sum2=0;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const g = toGray(data[i], data[i+1], data[i+2]);
    gray[j] = g;
    sum += g; sum2 += g*g;
  }
  const n = width*height;
  const mean = sum / n;
  const variance = sum2/n - mean*mean;
  return { gray, width, height, mean, std: Math.sqrt(Math.max(variance,1e-8)) };
}
function darkMask({gray,width,height,mean,std}, k=0.35) {
  const thr = mean - k*std;
  const mask = new Uint8Array(width*height);
  let count = 0;
  for (let i=0;i<gray.length;i++){
    if (gray[i] < thr){ mask[i]=1; count++; }
  }
  return { mask, width, height, thr, count };
}
function centroid(mask,width,height) {
  let sx=0, sy=0, c=0;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const i = y*width+x;
      if (mask[i]){ sx+=x; sy+=y; c++; }
    }
  }
  if (c===0) return {cx: width/2, cy: height/2, count:0};
  return {cx: sx/c, cy: sy/c, count:c};
}
function robustRadius(mask, width, height, cx, cy) {
  const dists = [];
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const i = y*width+x;
      if (mask[i]){
        const dx = x-cx, dy=y-cy;
        dists.push(Math.hypot(dx,dy));
      }
    }
  }
  if (dists.length<20) return Math.min(width,height)/2 * 0.48;
  dists.sort((a,b)=>a-b);
  const p80 = dists[Math.floor(dists.length*0.80)];
  return p80;
}
function cropCircle(imgBitmap, srcCanvas, cx, cy, r, pad=1.15) {
  const size = Math.max(64, Math.ceil(2*r*pad));
  const out = document.createElement("canvas");
  out.width = size; out.height = size;
  const ctx = out.getContext("2d");
  ctx.clearRect(0,0,size,size);
  ctx.save();
  ctx.beginPath();
  ctx.arc(size/2, size/2, r, 0, Math.PI*2);
  ctx.clip();
  const dx = Math.round(size/2 - cx);
  const dy = Math.round(size/2 - cy);
  ctx.drawImage(srcCanvas, dx, dy);
  ctx.restore();
  // explicit transparency outside
  const id = ctx.getImageData(0,0,size,size);
  const data = id.data;
  for (let y=0;y<size;y++){
    for (let x=0;x<size;x++){
      const dx2 = x-size/2, dy2=y-size/2;
      if (dx2*dx2 + dy2*dy2 > r*r){
        const p = (y*size + x)*4 + 3;
        data[p] = 0;
      }
    }
  }
  ctx.putImageData(id,0,0);
  return out;
}
function polarStats(gray, width, height, cx, cy, r) {
  const bins = 100;
  const sum = new Float64Array(bins);
  const cnt = new Uint32Array(bins);
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx = x-cx, dy=y-cy;
      const d = Math.hypot(dx,dy);
      if (d<=r){
        const idx = Math.min(bins-1, Math.floor((d/r)*(bins)));
        const p = y*width+x;
        sum[idx]+=gray[p]; cnt[idx]++;
      }
    }
  }
  const prof = new Float64Array(bins);
  for (let i=0;i<bins;i++) prof[i] = cnt[i] ? sum[i]/cnt[i] : 0;
  const ring = (a,b)=>{
    let s=0,c=0;
    const i0=Math.floor(a*bins), i1=Math.min(bins-1, Math.floor(b*bins));
    for (let i=i0;i<=i1;i++){ s+=prof[i]; c++; }
    return c? s/c : 0;
  };
  const mid = ring(0.45,0.65);
  const edge = ring(0.85,0.99);
  const eci = edge - mid;
  return { profile: prof, eci, mid, edge };
}
function localVariance(gray, width, height, cx, cy, r, block=16) {
  const wBlocks = Math.ceil(width/block), hBlocks = Math.ceil(height/block);
  const varmap = new Float32Array(wBlocks*hBlocks);
  let vmax = 0;
  for (let by=0; by<hBlocks; by++){
    for (let bx=0; bx<wBlocks; bx++){
      let s=0, s2=0, c=0;
      const x0=bx*block, y0=by*block;
      for (let y=y0; y<Math.min(y0+block, height); y++){
        for (let x=x0; x<Math.min(x0+block, width); x++){
          const dx=x-cx, dy=y-cy;
          if (dx*dx+dy*dy<=r*r){
            const p = y*width+x;
            const g = gray[p];
            s += g; s2 += g*g; c++;
          }
        }
      }
      const idx = by*wBlocks+bx;
      if (c>0){
        const m=s/c, v = Math.max(s2/c - m*m, 0);
        varmap[idx]=v;
        if (v>vmax) vmax=v;
      } else {
        varmap[idx]=0;
      }
    }
  }
  return { varmap, wBlocks, hBlocks, vmax, block };
}
function extremesRatio(gray, width, height, cx, cy, r) {
  let s=0, c=0;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      if (dx*dx+dy*dy<=r*r){ s += gray[y*width+x]; c++; }
    }
  }
  const mean = s/Math.max(c,1);
  let s2=0;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      if (dx*dx+dy*dy<=r*r){ const g=gray[y*width+x]; s2 += (g-mean)*(g-mean); }
    }
  }
  const std = Math.sqrt(Math.max(s2/Math.max(c,1), 1e-8));
  const low = mean - 1.0*std, high = mean + 1.0*std;
  let ext=0;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      if (dx*dx+dy*dy<=r*r){
        const g=gray[y*width+x];
        if (g<low || g>high) ext++;
      }
    }
  }
  return ext/Math.max(c,1);
}
// Sobel edges (simple) -> magnitude
function sobel(gray, width, height){
  const mag = new Float32Array(width*height);
  const kx = [-1,0,1,-2,0,2,-1,0,1];
  const ky = [-1,-2,-1,0,0,0,1,2,1];
  for (let y=1;y<height-1;y++){
    for (let x=1;x<width-1;x++){
      let gx=0, gy=0;
      let idx=0;
      for (let j=-1;j<=1;j++){
        for (let i=-1;i<=1;i++){
          const p = (y+j)*width + (x+i);
          const g = gray[p];
          gx += g * kx[idx];
          gy += g * ky[idx];
          idx++;
        }
      }
      const m = Math.hypot(gx,gy);
      mag[y*width+x] = m;
    }
  }
  // normalize approx
  let s=0, s2=0, c=0;
  for (let i=0;i<mag.length;i++){ const v=mag[i]; if (v>0){ s+=v; s2+=v*v; c++; } }
  const mean = c? s/c : 0;
  const std = c? Math.sqrt(Math.max(s2/c - mean*mean, 1e-8)) : 1;
  return { mag, mean, std };
}
function sectorDeviation(gray, width, height, cx, cy, r, sectors=24){
  const sums = new Float64Array(sectors);
  const cnts = new Uint32Array(sectors);
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      const rr = Math.hypot(dx,dy);
      if (rr<=r){
        const a = Math.atan2(dy,dx); // -pi..pi
        let k = Math.round(((a + Math.PI) / (2*Math.PI)) * (sectors-1));
        if (k<0) k=0; if (k>=sectors) k=sectors-1;
        sums[k]+=gray[y*width+x]; cnts[k]++;
      }
    }
  }
  const means = new Float64Array(sectors);
  let globalS=0, globalC=0;
  for (let i=0;i<sectors;i++){ means[i] = cnts[i]? sums[i]/cnts[i] : 0; globalS+=sums[i]; globalC+=cnts[i]; }
  const globalMean = globalC? globalS/globalC : 0;
  let s2=0;
  for (let i=0;i<sectors;i++){ const d=means[i]-globalMean; s2 += d*d; }
  const std = Math.sqrt(Math.max(s2/sectors, 1e-8));
  return { means, globalMean, std };
}

function makeHeatmapCanvas(varmap, wBlocks, hBlocks, vmax, block=16) {
  const W = wBlocks*block, H = hBlocks*block;
  const can = document.createElement("canvas");
  can.width = W; can.height = H;
  const ctx = can.getContext("2d");
  const id = ctx.createImageData(W,H);
  function colormap(t) {
    const r = t<0.5 ? 0 : Math.floor(lerp(0,255,(t-0.5)*2));
    const g = t<0.5 ? Math.floor(lerp(0,255,t*2)) : Math.floor(lerp(255,128,(t-0.5)*2));
    const b = Math.floor(lerp(128,0,t));
    return [r,g,b];
  }
  for (let by=0; by<hBlocks; by++){
    for (let bx=0; bx<wBlocks; bx++){
      const v = varmap[by*wBlocks+bx];
      const t = vmax>0 ? clamp01(v/vmax) : 0;
      const [R,G,B] = colormap(t);
      for (let y=0;y<block;y++){
        for (let x=0;x<block;x++){
          const X=bx*block+x, Y=by*block+y, p=(Y*W+X)*4;
          id.data[p]=R; id.data[p+1]=G; id.data[p+2]=B; id.data[p+3]=180;
        }
      }
    }
  }
  ctx.putImageData(id,0,0);
  return can;
}

// Draw overlay helpers
function drawGuides(size, r, spokes=12){
  const can = document.createElement("canvas");
  can.width=size; can.height=size;
  const ctx = can.getContext("2d");
  ctx.clearRect(0,0,size,size);
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.lineWidth = 1.2;
  const cx=size/2, cy=size/2;
  const rings=[0.3, 0.6, 0.9];
  rings.forEach(t=>{
    ctx.beginPath();
    ctx.arc(cx, cy, r*t, 0, Math.PI*2);
    ctx.stroke();
  });
  ctx.strokeStyle="rgba(147,197,253,0.6)"; // bluish for spokes
  for (let i=0;i<spokes;i++){
    const a = (i/spokes)*Math.PI*2;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a)*r*0.05, cy + Math.sin(a)*r*0.05);
    ctx.lineTo(cx + Math.cos(a)*r*0.98, cy + Math.sin(a)*r*0.98);
    ctx.stroke();
  }
  return can;
}
function drawEdgesOverlay(gray, width, height, cx, cy, r){
  const { mag, mean, std } = sobel(gray, width, height);
  const can = document.createElement("canvas");
  can.width = width; can.height = height;
  const ctx = can.getContext("2d");
  const id = ctx.createImageData(width, height);
  const thr = mean + 1.1*std;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const i=y*width+x;
      const dx=x-cx, dy=y-cy;
      const inside = (dx*dx+dy*dy)<=r*r;
      const m = mag[i];
      const p=i*4;
      if (inside && m>thr){
        id.data[p]=235; id.data[p+1]=74; id.data[p+2]=80; id.data[p+3]=220; // red-ish
      } else {
        id.data[p+3]=0;
      }
    }
  }
  ctx.putImageData(id,0,0);
  return { can, thr, mean, std };
}
function drawSectorsOverlay(gray, width, height, cx, cy, r, sectors=24){
  const { means, globalMean, std } = sectorDeviation(gray, width, height, cx, cy, r, sectors);
  const can = document.createElement("canvas");
  can.width = width; can.height = height;
  const ctx = can.getContext("2d");
  ctx.translate(cx, cy);
  ctx.globalAlpha = 0.28;
  const step = (Math.PI*2)/sectors;
  for (let k=0;k<sectors;k++){
    const dev = means[k]-globalMean;
    const t = clamp01(Math.abs(dev) / (std*1.2 + 1e-6));
    let color = dev>0 ? `rgba(59,130,246,${0.20 + 0.5*t})` : `rgba(239,68,68,${0.20 + 0.5*t})`; // blue=claro, red=oscuro
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0,0);
    const a0 = k*step, a1 = (k+1)*step;
    ctx.arc(0,0,r, a0, a1);
    ctx.closePath();
    ctx.fill();
  }
  ctx.setTransform(1,0,0,1,0,0);
  return { can, means, globalMean, std };
}

// Simple charts
function drawProfileChart(profile){
  const W=360, H=120, pad=8;
  const can = document.createElement("canvas"); can.width=W; can.height=H;
  const ctx = can.getContext("2d");
  ctx.fillStyle="rgba(255,255,255,0.05)"; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle="rgba(255,255,255,0.15)"; ctx.lineWidth=1;
  for (let i=0;i<5;i++){ const y = pad + (H-2*pad)*i/4; ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(W-pad,y); ctx.stroke(); }
  const min = Math.min(...profile), max = Math.max(...profile);
  const nx = (i)=> pad + (W-2*pad)*(i/(profile.length-1));
  const ny = (v)=> pad + (H-2*pad)*(1 - (v-min)/(max-min+1e-6));
  ctx.strokeStyle="#60a5fa"; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(nx(0), ny(profile[0]));
  for (let i=1;i<profile.length;i++){ ctx.lineTo(nx(i), ny(profile[i])); }
  ctx.stroke();
  ctx.fillStyle="rgba(255,255,255,0.6)";
  ctx.font="12px ui-sans-serif";
  ctx.fillText("Radial profile (0 centro → 1 borde)", 10, 16);
  return can;
}
function drawHistogram(gray, width, height, cx, cy, r){
  const bins=32;
  const hist=new Uint32Array(bins);
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      if (dx*dx+dy*dy<=r*r){
        const g = gray[y*width+x];
        const b = Math.max(0, Math.min(bins-1, Math.floor(g*bins)));
        hist[b]++;
      }
    }
  }
  const W=360, H=120, pad=8;
  const can = document.createElement("canvas"); can.width=W; can.height=H;
  const ctx = can.getContext("2d");
  ctx.fillStyle="rgba(255,255,255,0.05)"; ctx.fillRect(0,0,W,H);
  const maxv = Math.max(...hist);
  const bw = (W-2*pad)/bins;
  ctx.fillStyle="#22c55e";
  for (let i=0;i<bins;i++){
    const h = (H-2*pad)* (hist[i]/(maxv||1));
    ctx.fillRect(pad + i*bw, H-pad-h, bw-1, h);
  }
  ctx.fillStyle="rgba(255,255,255,0.6)";
  ctx.font="12px ui-sans-serif";
  ctx.fillText("Histograma de intensidades", 10, 16);
  return can;
}

function classify(eci, hetero, extremes) {
  const eciAbs = Math.abs(eci);
  let level = "OK";
  let notes = [];
  if (eciAbs > 0.08) notes.push("Contraste borde-centro notable (posible canalización en bordes).");
  if (hetero > 0.055) notes.push("Textura heterogénea (flujos irregulares).");
  if (extremes > 0.12) notes.push("Muchos extremos tonales (huecos/fisuras o zonas sobre-extraídas).");
  const score = (eciAbs/0.08)*0.4 + (hetero/0.055)*0.35 + (extremes/0.12)*0.25;
  if (score > 2.2) level = "Severa";
  else if (score > 1.0) level = "Leve";
  const recs = [];
  if (eciAbs > 0.08) recs.push("Mejorar distribución en periferia (WDT más fino, chequeo de dosis/basket).");
  if (hetero > 0.055) recs.push("Revisar consistencia de molienda y compactado (tamper recto, fuerza estable).");
  if (extremes > 0.12) recs.push("Reducir jetting: ajustar calibración del molino, evaluar preinfusión/ratio.");
  if (recs.length===0) recs.push("Puck homogéneo. Mantener parámetros actuales.");
  return { level, notes, recs };
}

function downloadBlob(filename, blob){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
}

export default function App(){
  const [items, setItems] = useState([]);
  const [busy, setBusy] = useState(false);
  const [overlay, setOverlay] = useState({
    heatmap: false,
    guides: true,
    edges: true,
    sectors: false
  });

  const onFiles = useCallback(async (files)=>{
    setBusy(true);
    const arr = [];
    for (const file of files){
      try {
        const img = await fileToImageBitmap(file);
        const { canvas } = drawToCanvas(img, MAX_DIM);
        const id = getImageData(canvas);
        const g = grayscale(id);
        const m = darkMask(g, 0.35);
        const c = centroid(m.mask, m.width, m.height);
        const r = robustRadius(m.mask, m.width, m.height, c.cx, c.cy);
        const cropped = cropCircle(img, canvas, c.cx, c.cy, r, 1.18);

        const cid = cropped.getContext("2d").getImageData(0,0,cropped.width, cropped.height);
        const cg = grayscale(cid);

        const stats = polarStats(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r);
        const lv = localVariance(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r, 14);
        const ext = extremesRatio(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r);
        const cls = classify(stats.eci, lv.vmax, ext);

        const heatmap = makeHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, lv.block);
        // Mask heatmap to circle
        const hmCtx = heatmap.getContext("2d");
        hmCtx.globalCompositeOperation = "destination-in";
        hmCtx.beginPath();
        hmCtx.arc(cropped.width/2, cropped.height/2, r, 0, Math.PI*2);
        hmCtx.fill();

        // Overlays
        const guides = drawGuides(cropped.width, r, 12);
        const edges = drawEdgesOverlay(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r);
        const sectors = drawSectorsOverlay(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r, 24);

        // charts
        const profileChart = drawProfileChart(Array.from(stats.profile));
        const histChart = drawHistogram(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r);

        const blob = await new Promise(res=>cropped.toBlob(res, "image/png"));
        const croppedURL = URL.createObjectURL(blob);
        const hmBlob = await new Promise(res=>heatmap.toBlob(res, "image/png"));
        const hmURL = URL.createObjectURL(hmBlob);

        const edgeBlob = await new Promise(res=>edges.can.toBlob(res, "image/png"));
        const edgeURL = URL.createObjectURL(edgeBlob);
        const sectorBlob = await new Promise(res=>sectors.can.toBlob(res, "image/png"));
        const sectorURL = URL.createObjectURL(sectorBlob);
        const guideBlob = await new Promise(res=>guides.toBlob(res, "image/png"));
        const guideURL = URL.createObjectURL(guideBlob);

        const pBlob = await new Promise(res=>profileChart.toBlob(res, "image/png"));
        const pURL = URL.createObjectURL(pBlob);
        const hBlob = await new Promise(res=>histChart.toBlob(res, "image/png"));
        const hURL = URL.createObjectURL(hBlob);

        arr.push({
          name: file.name,
          croppedW: cropped.width,
          croppedH: cropped.height,
          r: Math.round(r),
          urls: { croppedURL, hmURL, edgeURL, sectorURL, guideURL, profileURL: pURL, histURL: hURL },
          metrics: { eci: stats.eci, extremes: ext },
          level: cls.level,
          notes: cls.notes,
          recs: cls.recs
        });
      } catch (e){
        console.error(e);
        arr.push({ name: file.name, error: String(e) });
      }
    }
    setItems(prev=>[...prev, ...arr]);
    setBusy(false);
  },[]);

  const handleDrop = (e)=>{
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files || []).filter(f=>f.type.startsWith("image/"));
    onFiles(files);
  };
  const handleBrowse = (e)=>{
    const files = Array.from(e.target.files || []).filter(f=>f.type.startsWith("image/"));
    onFiles(files);
  };

  return (
    <div className="container">
      <div className="row" style={{justifyContent:"space-between", alignItems:"flex-start", gap:16}}>
        <div>
          <h1 style={{margin:"0 0 6px 0"}}>Puck Diagnosis</h1>
          <div className="muted">Sube fotos del puck. Recorte y centrado automáticos; análisis con overlays opcionales.</div>
        </div>
        <div className="row">
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.heatmap} onChange={e=>setOverlay(o=>({...o, heatmap:e.target.checked}))} />
            <span>Heatmap</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.guides} onChange={e=>setOverlay(o=>({...o, guides:e.target.checked}))} />
            <span>Líneas guía</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.edges} onChange={e=>setOverlay(o=>({...o, edges:e.target.checked}))} />
            <span>Resaltar grietas</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.sectors} onChange={e=>setOverlay(o=>({...o, sectors:e.target.checked}))} />
            <span>Mapa de sectores</span>
          </label>
          <button className="btn" disabled={busy} onClick={()=>document.getElementById("file-input").click()}>
            {busy? "Procesando..." : "Seleccionar imágenes"}
          </button>
          <input id="file-input" type="file" accept="image/*" multiple style={{display:"none"}} onChange={handleBrowse} />
        </div>
      </div>

      <div style={{height:16}} />

      <div className="card">
        <div className="drop" onDrop={handleDrop} onDragOver={(e)=>e.preventDefault()}>
          <div style={{fontWeight:700, fontSize:18, marginBottom:6}}>Arrastra y suelta imágenes aquí</div>
          <div className="muted">o usa <span className="kbd">Seleccionar imágenes</span>. Formatos: PNG/JPG.</div>
          <div style={{height:8}} />
          <div className="row">
            <span className="tag">Recorte + centrado</span>
            <span className="tag">Fondo transparente</span>
            <span className="tag">Heatmap opcional</span>
            <span className="tag">Grietas (Sobel)</span>
            <span className="tag">Guías + sectores</span>
          </div>
        </div>
      </div>

      <div style={{height:18}} />

      <div className="grid">
        {items.map((it, idx)=>(
          <div key={idx} className="card">
            <div className="row" style={{justifyContent:"space-between"}}>
              <div style={{fontWeight:600}} className="small">{it.name}</div>
              {!it.error && (
                <div className="row small" style={{gap:8}}>
                  <span className="muted">r={it.r}px</span>
                  <span className="tag">{it.level==="OK" ? "OK" : it.level==="Leve" ? "Leve" : "Severa"}</span>
                </div>
              )}
            </div>

            {it.error ? (
              <div className="err small" style={{marginTop:8}}>Error: {it.error}</div>
            ) : (
              <>
                <div className="thumb canvas-stack" style={{marginTop:12}}>
                  <img className="base" src={overlay.heatmap ? it.urls.hmURL : it.urls.croppedURL} alt="puck" />
                  {overlay.guides && <img src={it.urls.guideURL} alt="guides" />}
                  {overlay.edges && <img src={it.urls.edgeURL} alt="edges" />}
                  {overlay.sectors && <img src={it.urls.sectorURL} alt="sectors" />}
                </div>

                <div style={{height:10}} />

                <div className="metrics">
                  <div>Δ Borde-Centro (ECI): <span className="mono">{it.metrics.eci.toFixed(3)}</span></div>
                  <div>Extremos tonales: <span className="mono">{(it.metrics.extremes*100).toFixed(1)}%</span></div>
                </div>

                <div className="charts">
                  <div className="chart"><img src={it.urls.profileURL} alt="profile" /></div>
                  <div className="chart"><img src={it.urls.histURL} alt="histogram" /></div>
                </div>

                <div style={{height:10}} />

                <div className="small">
                  {it.notes && it.notes.length>0 ? (
                    <ul style={{marginTop:4, paddingLeft:18}}>
                      {it.notes.map((n,i)=>(<li key={i}>{n}</li>))}
                    </ul>
                  ) : <div className="ok">Puck homogéneo.</div>}
                </div>

                <div style={{height:10}} />

                <div className="small">
                  <div style={{fontWeight:600, marginBottom:4}}>Sugerencias:</div>
                  <ul style={{marginTop:0, paddingLeft:18}}>
                    {it.recs.map((r,i)=>(<li key={i}>{r}</li>))}
                  </ul>
                </div>

                <div style={{height:12}} />

                <div className="row">
                  <button className="btn" onClick={async ()=>{
                    const resp = await fetch(it.urls.croppedURL);
                    const blob = await resp.blob();
                    downloadBlob(`puck-${idx+1}.png`, blob);
                  }}>Descargar recorte</button>
                  <button className="btn" onClick={async ()=>{
                    const resp = await fetch(it.urls.hmURL);
                    const blob = await resp.blob();
                    downloadBlob(`puck-heatmap-${idx+1}.png`, blob);
                  }}>Descargar heatmap</button>
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      <div style={{height:24}} />
      <div className="muted small">
        * Overlays: Guías (anillos + radios), Grietas (Sobel adaptativo), Sectores (desviación angular).<br/>
        Ajusta umbrales en <span className="kbd">App.jsx</span> para tus datasets.
      </div>
    </div>
  );
}
