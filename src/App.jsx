import React, { useCallback, useState, useRef } from "react";

// ==== Helpers (heredados de v0.4) ====
const MAX_DIM = 1024;
function clamp01(x){ return Math.min(1, Math.max(0, x)); }
function lerp(a,b,t){ return a+(b-a)*t; }
function toGray(r,g,b){ return (0.2126*r + 0.7152*g + 0.0722*b)/255; }
function round(n){ return Math.round(n*1000)/1000; }

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
function cropCircle(srcCanvas, cx, cy, r, pad=1.15) {
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
  // transparency outside
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
  const bins = 120;
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
  const ringAvg = (a,b)=>{
    let s=0,c=0;
    const i0=Math.floor(a*bins), i1=Math.min(bins-1, Math.floor(b*bins));
    for (let i=i0;i<=i1;i++){ s+=prof[i]; c++; }
    return c? s/c : 0;
  };
  const mid = ringAvg(0.45,0.65);
  const edge = ringAvg(0.90,0.99);
  const eci = edge - mid;
  return { profile: prof, eci, mid, edge, ringAvg };
}
function localVariance(gray, width, height, cx, cy, r, blocks=64) {
  const wBlocks = blocks, hBlocks = blocks;
  const bw = Math.floor(width / wBlocks);
  const bh = Math.floor(height / hBlocks);
  const varmap = new Float32Array(wBlocks*hBlocks);
  let vmax = 0;
  for (let by=0; by<hBlocks; by++){
    for (let bx=0; bx<wBlocks; bx++){
      let s=0, s2=0, c=0;
      const x0=bx*bw, y0=by*bh;
      for (let y=y0; y<Math.min(y0+bh, height); y++){
        for (let x=x0; x<Math.min(x0+bw, width); x++){
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
  return { varmap, wBlocks, hBlocks, vmax, bw, bh };
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
  return { ratio: ext/Math.max(c,1), mean, std };
}
function sobel(gray, width, height){
  const mag = new Float32Array(width*height);
  const kx = [-1,0,1,-2,0,2,-1,0,1];
  const ky = [-1,-2,-1,0,0,0,1,2,1];
  for (let y=1;y<height-1;y++){
    for (let x=1;x<width-1;x++){
      let gx=0, gy=0, idx=0;
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
  let s=0, s2=0, c=0;
  for (let i=0;i<mag.length;i++){ const v=mag[i]; if (v>0){ s+=v; s2+=v*v; c++; } }
  const mean = c? s/c : 0;
  const std = c? Math.sqrt(Math.max(s2/c - mean*mean, 1e-8)) : 1;
  return { mag, mean, std };
}
function drawHeatmapCanvas(varmap, wBlocks, hBlocks, vmax, bw, bh, width, height) {
  const can = document.createElement("canvas");
  can.width = width; can.height = height;
  const ctx = can.getContext("2d");
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
      ctx.fillStyle = `rgba(${R},${G},${B},0.7)`;
      ctx.fillRect(bx*bw, by*bh, bw, bh);
    }
  }
  return can;
}
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
  ctx.strokeStyle="rgba(147,197,253,0.7)";
  for (let i=0;i<spokes;i++){
    const a = (i/spokes)*Math.PI*2;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a)*r*0.08, cy + Math.sin(a)*r*0.08);
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
  let insideEdges=0, area=0;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const i=y*width+x;
      const dx=x-cx, dy=y-cy;
      const inside = (dx*dx+dy*dy)<=r*r;
      if (inside) area++;
      const m = mag[i];
      const p=i*4;
      if (inside && m>thr){
        id.data[p]=235; id.data[p+1]=74; id.data[p+2]=80; id.data[p+3]=220;
        insideEdges++;
      } else {
        id.data[p+3]=0;
      }
    }
  }
  ctx.putImageData(id,0,0);
  const density = area? insideEdges/area : 0;
  return { can, thr, mean, std, density };
}
// === Dark ring detection (annulus near edge) ===
function detectDarkRing(profile, band=[0.78,0.95]){
  const n = profile.length;
  const i0 = Math.floor(band[0]*n);
  const i1 = Math.min(n-1, Math.floor(band[1]*n));
  let minV = Infinity, minI = i0;
  for (let i=i0;i<=i1;i++){
    const v = profile[i];
    if (v < minV){ minV=v; minI=i; }
  }
  const pos = minI / n; // 0..1 radius
  return { minV, pos };
}
function drawRingOverlay(size, r, pos, band=0.06){
  const can = document.createElement("canvas");
  can.width=size; can.height=size;
  const ctx = can.getContext("2d");
  const cx=size/2, cy=size/2;
  const rMid = pos * r;
  const r1 = Math.max(0, rMid - band*r);
  const r2 = Math.min(r, rMid + band*r);
  ctx.fillStyle="rgba(168,85,247,0.28)"; // purple
  ctx.beginPath(); ctx.arc(cx, cy, r2, 0, Math.PI*2); ctx.arc(cx, cy, r1, 0, Math.PI*2, true); ctx.closePath(); ctx.fill();
  // label
  ctx.fillStyle="rgba(0,0,0,0.5)"; ctx.fillRect(8, 8, 150, 20);
  ctx.fillStyle="rgba(255,255,255,0.9)"; ctx.font="12px ui-sans-serif"; ctx.fillText("Anillo oscuro (estimado)", 12, 22);
  return can;
}

function drawSectorsOverlay(gray, width, height, cx, cy, r, sectors=24){
  const sums = new Float64Array(sectors);
  const cnts = new Uint32Array(sectors);
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      const rr = Math.hypot(dx,dy);
      if (rr<=r){
        const a = Math.atan2(dy,dx);
        let k = Math.floor(((a + Math.PI) / (2*Math.PI)) * sectors);
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
  const can = document.createElement("canvas");
  can.width = width; can.height = height;
  const ctx = can.getContext("2d");
  ctx.translate(cx, cy);
  ctx.globalAlpha = 0.28;
  const step = (Math.PI*2)/sectors;
  for (let k=0;k<sectors;k++){
    const dev = means[k]-globalMean;
    const t = clamp01(Math.abs(dev) / (std*1.2 + 1e-6));
    let color = dev>0 ? `rgba(59,130,246,${0.25 + 0.5*t})` : `rgba(239,68,68,${0.25 + 0.5*t})`;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0,0);
    const a0 = k*step, a1 = (k+1)*step;
    ctx.arc(0,0,r, a0, a1);
    ctx.closePath();
    ctx.fill();
  }
  // legend
  ctx.setTransform(1,0,0,1,0,0);
  ctx.globalAlpha = 1;
  const pad=8;
  ctx.fillStyle="rgba(0,0,0,0.45)";
  ctx.fillRect(pad, pad, 210, 52);
  ctx.strokeStyle="rgba(255,255,255,0.25)";
  ctx.strokeRect(pad, pad, 210, 52);
  ctx.font="12px ui-sans-serif"; ctx.fillStyle="rgba(255,255,255,0.85)";
  ctx.fillText("Mapa de sectores (desv. angular)", pad+8, pad+16);
  ctx.fillStyle="rgba(239,68,68,0.9)"; ctx.fillRect(pad+8, pad+24, 16, 12);
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("rojo: más oscuro (atascos)", pad+28, pad+34);
  ctx.fillStyle="rgba(59,130,246,0.9)"; ctx.fillRect(pad+8, pad+38, 16, 12);
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("azul: más claro (flujo rápido)", pad+28, pad+48);
  return { can, means, globalMean, std };
}
function drawProfileChart(profile){
  const W=520, H=150, pad=8;
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
  ctx.fillStyle="rgba(255,255,255,0.8)";
  ctx.font="12px ui-sans-serif";
  ctx.fillText("Perfil radial (0 centro → 1 borde)", 10, 16);
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
  const W=520, H=150, pad=8;
  const can = document.createElement("canvas"); can.width=W; can.height=H;
  const ctx = can.getContext("2d");
  ctx.fillStyle="rgba(255,255,255,0.05)"; ctx.fillRect(0,0,W,H);
  const maxv = Math.max(...hist);
  const bw = (W-2*pad)/bins;
  ctx.fillStyle="#22c55e";
  for (let i=0;i<bins;i++){
    const h = (H-2*pad)* (hist[i]/(maxv||1));
    ctx.fillRect(pad + i*bw, H-pad-h, Math.max(1,bw-1), h);
  }
  ctx.fillStyle="rgba(255,255,255,0.8)";
  ctx.font="12px ui-sans-serif";
  ctx.fillText("Histograma de intensidades (puck)", 10, 16);
  return can;
}
async function canvasToURL(can){
  const b = await new Promise(res=>can.toBlob(res, "image/png"));
  return URL.createObjectURL(b);
}
function corr(a,b){
  const n = Math.min(a.length, b.length);
  let sx=0, sy=0, sxx=0, syy=0, sxy=0;
  for (let i=0;i<n;i++){
    const x=a[i], y=b[i];
    sx += x; sy += y;
    sxx += x*x; syy += y*y; sxy += x*y;
  }
  const cov = sxy/n - (sx/n)*(sy/n);
  const vx = sxx/n - (sx/n)*(sx/n);
  const vy = syy/n - (sy/n)*(sy/n);
  const denom = Math.sqrt(Math.max(vx,1e-12)*Math.max(vy,1e-12));
  return denom>0? clamp01((cov/denom + 1)/2) : 0;
}

// Analizar imagen (cara puck o huella)
async function analyzeImageFile(file){
  const img = await fileToImageBitmap(file);
  const { canvas } = drawToCanvas(img, MAX_DIM);
  const id = getImageData(canvas);
  const g = grayscale(id);
  const m = darkMask(g, 0.35);
  const c = centroid(m.mask, m.width, m.height);
  const r = robustRadius(m.mask, m.width, m.height, c.cx, c.cy);
  const cropped = cropCircle(canvas, c.cx, c.cy, r, 1.18);

  const cid = cropped.getContext("2d").getImageData(0,0,cropped.width, cropped.height);
  const cg = grayscale(cid);
  const centerX = cropped.width/2, centerY = cropped.height/2;

  const stats = polarStats(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const lv = localVariance(cg.gray, cg.width, cg.height, centerX, centerY, r, 64);
  const ext = extremesRatio(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const edges = drawEdgesOverlay(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const sectors = drawSectorsOverlay(cg.gray, cg.width, cg.height, centerX, centerY, r, 24);
  const heatmap = drawHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, lv.bw, lv.bh, cg.width, cg.height);
  const guides = drawGuides(cropped.width, r, 12);

  // Dark ring (annulus)
  const ring = detectDarkRing(stats.profile, [0.78,0.96]);
  const ringDepth = (stats.mid - ring.minV); // >0 => más oscuro que el medio
  const ringOverlay = drawRingOverlay(cropped.width, r, ring.pos, 0.06);

  // charts
  const profileChart = drawProfileChart(Array.from(stats.profile));
  const histChart = drawHistogram(cg.gray, cg.width, cg.height, centerX, centerY, r);

  // urls
  const [croppedURL, hmURL, edgeURL, sectorURL, guideURL, ringURL, profileURL, histURL] = await Promise.all([
    canvasToURL(cropped), canvasToURL(heatmap), canvasToURL(edges.can),
    canvasToURL(sectors.can), canvasToURL(guides), canvasToURL(ringOverlay),
    canvasToURL(profileChart), canvasToURL(histChart)
  ]);

  return {
    dim: { w: cropped.width, h: cropped.height, r: Math.round(r) },
    urls: { croppedURL, hmURL, edgeURL, sectorURL, guideURL, ringURL, profileURL, histURL },
    metrics: {
      eci: stats.eci,
      extremes: ext.ratio,
      sectorStd: sectors.std,
      edgeDensity: edges.density,
      ringDepth,
      ringPos: ring.pos
    },
    arrays: { sectorMeans: sectors.means, radialProfile: stats.profile }
  };
}

function pairRecommendations(top, bottom, pair){
  const L = [];
  const B = [];

  L.push(`• ECI superior: ${round(top.metrics.eci)} | inferior: ${round(bottom.metrics.eci)} (Δ=${round(top.metrics.eci - bottom.metrics.eci)})`);
  L.push(`• Extremos sup/inf: ${round(top.metrics.extremes*100)}% / ${round(bottom.metrics.extremes*100)}%`);
  L.push(`• σ sectores sup/inf: ${round(top.metrics.sectorStd)} / ${round(bottom.metrics.sectorStd)}`);
  L.push(`• Grietas sup/inf: ${round(top.metrics.edgeDensity*100)}% / ${round(bottom.metrics.edgeDensity*100)}%`);
  L.push(`• Anillo oscuro (inf): profundidad ${round(bottom.metrics.ringDepth)} @ r≈${Math.round(bottom.metrics.ringPos*100)}%`);
  L.push(`• Correlación sectores (ajustada rotación): ${Math.round(pair.sectorCorr*100)}%`);

  if (pair.sectorCorr>0.65){
    B.push("Canales atraviesan el puck (coinciden sectores). WDT profundo y uniforme, preinfusión suave, verifica nivelación del tamper.");
  }
  if (bottom.metrics.ringDepth>0.02){
    B.push("Anillo oscuro en la cara inferior: posible bypass por la periferia. Revisa headspace, diámetro del tamper (ajustado), y valora filtro de papel inferior.");
  }
  if (bottom.metrics.edgeDensity > top.metrics.edgeDensity + 0.03){
    B.push("Más grietas en inferior: reduce golpes post-tamper, baja el ramp‑up de presión y evalúa paper filter abajo.");
  }
  if (top.metrics.extremes > bottom.metrics.extremes + 0.05){
    B.push("La cara superior presenta más extremos: mejora la distribución antes del tamper (WDT 10–15 s, romper grumos, RDT si hay estática).");
  }
  if (B.length===0){
    B.push("Puck consistente entre caras. Mantén parámetros y ajusta fino ratio/tiempo por sabor.");
  }
  return { lines: L, bullets: B };
}

function downloadBlob(filename, blob){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
}

export default function App(){
  const [sets, setSets] = useState([]);
  const [busy, setBusy] = useState(false);
  const [overlay, setOverlay] = useState({ heatmap:false, guides:true, edges:true, sectors:false, ring:true });
  const [rotateBack, setRotateBack] = useState(true); // 180° al comparar sectores
  const topRef = useRef(); const bottomRef = useRef();
  const topPrintRef = useRef(); const bottomPrintRef = useRef(); // huellas en papel (opcional)

  async function addSet(){
    const fTop = topRef.current.files?.[0];
    const fBottom = bottomRef.current.files?.[0];
    if(!fTop || !fBottom){ alert("Sube una imagen SUPERIOR y otra INFERIOR del puck."); return; }
    setBusy(true);
    try{
      const [top, bottomRaw] = await Promise.all([analyzeImageFile(fTop), analyzeImageFile(fBottom)]);

      // rotación inferior para comparación de sectores
      let bottom = bottomRaw;
      if (rotateBack){
        const imgRes = await fetch(bottomRaw.urls.croppedURL);
        const blob = await imgRes.blob();
        const bmp = await createImageBitmap(blob);
        const tmp = document.createElement("canvas");
        tmp.width = bmp.width; tmp.height = bmp.height;
        const ctx = tmp.getContext("2d");
        ctx.translate(tmp.width/2, tmp.height/2);
        ctx.rotate(Math.PI);
        ctx.drawImage(bmp, -bmp.width/2, -bmp.height/2);
        const rotatedURL = await (async()=>{
          const b = await new Promise(res=>tmp.toBlob(res, "image/png"));
          return URL.createObjectURL(b);
        })();
        bottom = { ...bottomRaw, urls: { ...bottomRaw.urls, croppedURL: rotatedURL } };
      }

      // correlación sectorial (ajustada)
      const a = Array.from(top.arrays.sectorMeans);
      let b = Array.from(bottomRaw.arrays.sectorMeans);
      if (rotateBack) b = b.slice().reverse();
      const sectorCorr = corr(a,b);

      const pair = { sectorCorr };
      const rec = pairRecommendations(top, bottomRaw, pair);

      // Huellas en papel (opcionales)
      let topPrint = null, bottomPrint = null;
      const fTopP = topPrintRef.current.files?.[0];
      const fBottomP = bottomPrintRef.current.files?.[0];
      if (fTopP) topPrint = await analyzeImageFile(fTopP);
      if (fBottomP) bottomPrint = await analyzeImageFile(fBottomP);

      setSets(prev=>[...prev, { top, bottom, bottomRaw, pair, rec, topPrint, bottomPrint }]);
      topRef.current.value = null; bottomRef.current.value = null;
      if (topPrintRef.current) topPrintRef.current.value = null;
      if (bottomPrintRef.current) bottomPrintRef.current.value = null;
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container">
      <div className="row" style={{justifyContent:"space-between", alignItems:"flex-start", gap:16}}>
        <div>
          <h1 style={{margin:"0 0 6px 0"}}>Puck Diagnosis</h1>
          <div className="muted">Sube <b>dos fotos por puck</b> con estas definiciones:
            <span className="tag" style={{marginLeft:8}}>Superior = lado que toca la ducha / group head</span>
            <span className="tag">Inferior = lado que toca la canasta/boquilla</span>
          </div>
        </div>
        <div className="row">
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.heatmap} onChange={e=>setOverlay(o=>({...o, heatmap:e.target.checked}))} />
            <span>Heatmap</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.guides} onChange={e=>setOverlay(o=>({...o, guides:e.target.checked}))} />
            <span>Guías</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.edges} onChange={e=>setOverlay(o=>({...o, edges:e.target.checked}))} />
            <span>Grietas</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.sectors} onChange={e=>setOverlay(o=>({...o, sectors:e.target.checked}))} />
            <span>Sectores</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.ring} onChange={e=>setOverlay(o=>({...o, ring:e.target.checked}))} />
            <span>Anillo oscuro</span>
          </label>
        </div>
      </div>

      <div style={{height:12}} />

      <div className="card">
        <div className="row" style={{justifyContent:"space-between", alignItems:"center"}}>
          <div className="row" style={{gap:16, alignItems:"center"}}>
            <div>
              <div className="small" title="Lado que toca la ducha / group head">Superior (group head)</div>
              <input ref={topRef} type="file" accept="image/*" />
            </div>
            <div>
              <div className="small" title="Lado que toca la canasta/boquilla">Inferior (canasta)</div>
              <input ref={bottomRef} type="file" accept="image/*" />
            </div>
            <div className="muted small" style={{marginLeft:8}}>Huellas en papel (opcional):</div>
            <div>
              <div className="small">Superior (papel)</div>
              <input ref={topPrintRef} type="file" accept="image/*" />
            </div>
            <div>
              <div className="small">Inferior (papel)</div>
              <input ref={bottomPrintRef} type="file" accept="image/*" />
            </div>
          </div>
          <div className="row" style={{gap:16}}>
            <label className="row small" style={{gap:8}}>
              <input type="checkbox" checked={rotateBack} onChange={e=>setRotateBack(e.target.checked)} />
              <span>Rotar inferior 180° para comparar</span>
            </label>
            <button className="btn" disabled={busy} onClick={addSet}>{busy? "Procesando..." : "Añadir set"}</button>
          </div>
        </div>
      </div>

      <div style={{height:18}} />

      <div className="sets">
        {sets.map((st, idx)=>(
          <div key={idx} className="card">
            <div style={{fontWeight:600}} className="small">Puck #{idx+1}</div>

            {/* Dos columnas: superior / inferior */}
            <div className="grid" style={{gridTemplateColumns:"1fr 1fr", gap:16, marginTop:10}}>
              {[{lab:"Superior (group head)", it: st.top}, {lab:"Inferior (canasta)", it: st.bottom}].map((obj, k)=>(
                <div key={k}>
                  <div className="small" style={{fontWeight:700, marginBottom:6}}>{obj.lab}</div>
                  <div className="thumb canvas-stack">
                    <img src={obj.it.urls.croppedURL} alt={obj.lab} />
                    {overlay.heatmap && <img className="overlay" src={obj.it.urls.hmURL} alt="heatmap" />}
                    {overlay.guides && <img className="overlay" src={obj.it.urls.guideURL} alt="guides" />}
                    {overlay.edges && <img className="overlay" src={obj.it.urls.edgeURL} alt="edges" />}
                    {overlay.sectors && <img className="overlay" src={obj.it.urls.sectorURL} alt="sectors" />}
                    {overlay.ring && <img className="overlay" src={obj.it.urls.ringURL} alt="ring" />}
                  </div>

                  <div style={{height:8}} />
                  <div className="metrics">
                    <div>ECI: <span className="mono">{obj.it.metrics.eci.toFixed(3)}</span></div>
                    <div>Extremos: <span className="mono">{(obj.it.metrics.extremes*100).toFixed(1)}%</span></div>
                    <div>σ sectores: <span className="mono">{obj.it.metrics.sectorStd.toFixed(3)}</span></div>
                    <div>Grietas: <span className="mono">{(obj.it.metrics.edgeDensity*100).toFixed(2)}%</span></div>
                    <div>Anillo oscuro (prof.): <span className="mono">{obj.it.metrics.ringDepth.toFixed(3)}</span></div>
                  </div>

                  <div className="charts">
                    <div className="chart"><img src={obj.it.urls.profileURL} alt="perfil radial" /></div>
                    <div className="chart"><img src={obj.it.urls.histURL} alt="histograma" /></div>
                  </div>
                </div>
              ))}
            </div>

            {/* Huellas en papel (opcionales) */}
            {(st.topPrint || st.bottomPrint) && (
              <>
                <div style={{height:12}} />
                <div className="small" style={{fontWeight:700}}>Huellas en papel</div>
                <div className="grid" style={{gridTemplateColumns:"1fr 1fr", gap:16, marginTop:8}}>
                  {st.topPrint && (
                    <div>
                      <div className="small" style={{fontWeight:700, marginBottom:6}}>Superior (papel)</div>
                      <div className="thumb canvas-stack">
                        <img src={st.topPrint.urls.croppedURL} alt="huella sup" />
                        {overlay.sectors && <img className="overlay" src={st.topPrint.urls.sectorURL} alt="sectors" />}
                        {overlay.ring && <img className="overlay" src={st.topPrint.urls.ringURL} alt="ring" />}
                      </div>
                    </div>
                  )}
                  {st.bottomPrint && (
                    <div>
                      <div className="small" style={{fontWeight:700, marginBottom:6}}>Inferior (papel)</div>
                      <div className="thumb canvas-stack">
                        <img src={st.bottomPrint.urls.croppedURL} alt="huella inf" />
                        {overlay.sectors && <img className="overlay" src={st.bottomPrint.urls.sectorURL} alt="sectors" />}
                        {overlay.ring && <img className="overlay" src={st.bottomPrint.urls.ringURL} alt="ring" />}
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

            <div style={{height:12}} />

            <div className="small">
              <div style={{fontWeight:700, marginBottom:4}}>Análisis combinado</div>
              <ul style={{marginTop:4, paddingLeft:18}}>
                {st.rec.lines.map((n,i)=>(<li key={i}>{n}</li>))}
              </ul>
            </div>

            <div style={{height:10}} />

            <div className="small">
              <div style={{fontWeight:700, marginBottom:4}}>Recomendaciones</div>
              <ul style={{marginTop:0, paddingLeft:18}}>
                {st.rec.bullets.map((r,i)=>(<li key={i}>{r}</li>))}
              </ul>
            </div>
          </div>
        ))}
      </div>

      <div style={{height:24}} />
      <div className="muted small">
        * Definiciones: <b>Superior</b> = lado contra la ducha (group head). <b>Inferior</b> = lado hacia la canasta/boquilla.<br/>
        * Sube también las <b>huellas en papel</b> (superior e inferior) para reforzar el diagnóstico de bypass/anillos.
      </div>
    </div>
  );
}
