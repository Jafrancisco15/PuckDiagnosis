import React, { useCallback, useState, useRef } from "react";
import { jsPDF } from "jspdf";

// ==== Helpers ====
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
// ---- Circle fit (Kåsa) for paper prints ----
function circleFitKasa(points){
  const n = points.length;
  if (n<3) return null;
  let sumx=0,sumy=0,sumx2=0,sumy2=0,sumxy=0,sumz=0,sumxz=0,sumyz=0;
  for (const [x,y] of points){
    const z = x*x + y*y;
    sumx+=x; sumy+=y; sumx2+=x*x; sumy2+=y*y; sumxy+=x*y; sumz+=z; sumxz+=x*z; sumyz+=y*z;
  }
  const A = [[sumx, sumy, n],[sumx2, sumxy, sumx],[sumxy, sumy2, sumy]];
  const B = [ -sumz, -sumxz, -sumyz ];
  function det3(m){
    return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
  }
  function solve3(a,b){
    const d = det3(a);
    if (Math.abs(d) < 1e-8) return null;
    const a1 = [[b[0],a[0][1],a[0][2]],[b[1],a[1][1],a[1][2]],[b[2],a[2][1],a[2][2]]];
    const a2 = [[a[0][0],b[0],a[0][2]],[a[1][0],b[1],a[1][2]],[a[2][0],b[2],a[2][2]]];
    const a3 = [[a[0][0],a[0][1],b[0]],[a[1][0],a[1][1],b[1]],[a[2][0],a[2][1],b[2]]];
    const A = det3(a1)/d, B = det3(a2)/d, C = det3(a3)/d;
    return [A,B,C];
  }
  const sol = solve3(A,B);
  if (!sol) return null;
  const [A1,B1,C1] = sol;
  const cx = -A1/2, cy = -B1/2;
  const R = Math.sqrt(Math.max(0, cx*cx + cy*cy - C1));
  return { cx, cy, r: R };
}

function cropCircleFromCanvas(srcCanvas, cx, cy, r, pad=1.15){
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
// Dark ring detection
function detectDarkRing(profile, band=[0.78,0.96]){
  const n = profile.length;
  const i0 = Math.floor(band[0]*n);
  const i1 = Math.min(n-1, Math.floor(band[1]*n));
  let minV = Infinity, minI = i0;
  for (let i=i0;i<=i1;i++){
    const v = profile[i];
    if (v < minV){ minV=v; minI=i; }
  }
  const pos = minI / n;
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
  ctx.fillStyle="rgba(168,85,247,0.28)";
  ctx.beginPath(); ctx.arc(cx, cy, r2, 0, Math.PI*2); ctx.arc(cx, cy, r1, 0, Math.PI*2, true); ctx.closePath(); ctx.fill();
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
  ctx.fillRect(pad, pad, 230, 52);
  ctx.strokeStyle="rgba(255,255,255,0.25)";
  ctx.strokeRect(pad, pad, 230, 52);
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
function corrPearson(a,b){
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
  return denom>0? cov/denom : 0;
}
function alignSectorMeans(a, b, allowReverse=true){
  // Try all cyclic shifts; if allowReverse, try reversed as well. Return best aligned b' and shift info.
  const n = a.length;
  let best = { corr:-2, shift:0, reversed:false, aligned:null };
  function shiftArr(arr,k){
    const out = new Array(n);
    for (let i=0;i<n;i++){ out[i] = arr[(i-k+n)%n]; }
    return out;
  }
  const variants = [ {arr:b, reversed:false} ];
  if (allowReverse) variants.push({arr:[...b].reverse(), reversed:true});
  for (const v of variants){
    for (let k=0;k<n;k++){
      const bb = shiftArr(v.arr,k);
      const c = corrPearson(a, bb);
      if (c > best.corr){ best = { corr:c, shift:k, reversed:v.reversed, aligned:bb }; }
    }
  }
  return best;
}
function zScore(arr){
  const n = arr.length;
  let s=0, s2=0;
  for (let i=0;i<n;i++){ s+=arr[i]; s2+=arr[i]*arr[i]; }
  const mean = s/n;
  const std = Math.sqrt(Math.max(s2/n - mean*mean, 1e-8));
  return { mean, std, z: arr.map(v=>(v-mean)/(std||1)) };
}
function drawSectorHighlights(size, r, sectors, indicesDark, indicesLight){
  const can = document.createElement("canvas");
  can.width=size; can.height=size;
  const ctx = can.getContext("2d");
  const cx=size/2, cy=size/2;
  const step = (Math.PI*2)/sectors;
  ctx.lineWidth = 6;
  // dark = red arcs at ~85-95% radius, light = cyan arcs
  function arcForIndex(i, color){
    const a0 = i*step, a1 = (i+1)*step;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, r*0.92, a0+0.04, a1-0.04);
    ctx.stroke();
  }
  indicesDark.forEach(i=>arcForIndex(i, "rgba(239,68,68,0.85)"));
  indicesLight.forEach(i=>arcForIndex(i, "rgba(34,211,238,0.85)"));
  // legend
  ctx.fillStyle="rgba(0,0,0,0.5)"; ctx.fillRect(8,8,190,40);
  ctx.fillStyle="rgba(255,255,255,0.9)"; ctx.font="12px ui-sans-serif"; ctx.fillText("Sectores coincidentes", 14, 22);
  ctx.fillStyle="rgba(239,68,68,0.85)"; ctx.fillRect(14,26,14,6);
  ctx.fillStyle="rgba(255,255,255,0.9)"; ctx.fillText("oscuros (canales)", 34, 32);
  ctx.fillStyle="rgba(34,211,238,0.85)"; ctx.fillRect(120,26,14,6);
  ctx.fillStyle="rgba(255,255,255,0.9)"; ctx.fillText("claros", 140, 32);
  return can;
}

// ---- Analyze puck face or paper print ----
async function analyzeImageFile(file, mode="puck"){
  const img = await fileToImageBitmap(file);
  const { canvas } = drawToCanvas(img, MAX_DIM);
  const id = getImageData(canvas);
  const g = grayscale(id);

  // initial estimate
  let m = darkMask(g, 0.35);
  let c = centroid(m.mask, m.width, m.height);
  let r = robustRadius(m.mask, m.width, m.height, c.cx, c.cy);

  // If paper print: refine center/radius by circle fit on edges near rim
  if (mode === "print"){
    const { mag, mean, std } = sobel(g.gray, g.width, g.height);
    const thr = mean + 1.2*std;
    const pts = [];
    const rMin = r*0.75, rMax = r*1.25;
    for (let y=1;y<g.height-1;y++){
      for (let x=1;x<g.width-1;x++){
        const i = y*g.width + x;
        if (mag[i] > thr){
          const dx = x-c.cx, dy=y-c.cy;
          const d = Math.hypot(dx,dy);
          if (d>=rMin && d<=rMax){
            pts.push([x,y]);
          }
        }
      }
    }
    if (pts.length>50){
      const fit = circleFitKasa(pts);
      if (fit){
        c = { cx: fit.cx, cy: fit.cy, count: pts.length };
        r = fit.r;
      }
    }
  }

  const cropped = cropCircleFromCanvas(canvas, c.cx, c.cy, r, 1.18);
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

  const ring = detectDarkRing(stats.profile, [0.78,0.96]);
  const ringDepth = (stats.mid - ring.minV);
  const ringOverlay = drawRingOverlay(cropped.width, r, ring.pos, 0.06);

  // charts
  const profileChart = drawProfileChart(Array.from(stats.profile));
  const histChart = drawHistogram(cg.gray, cg.width, cg.height, centerX, centerY, r);

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

// Pair recommendations (same as v0.5 with text tweaks)
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
    B.push("Anillo oscuro en la cara inferior: posible bypass periférico. Revisa headspace, diámetro del tamper (ajustado), y valora filtro de papel inferior.");
  }
  if (bottom.metrics.edgeDensity > top.metrics.edgeDensity + 0.03){
    B.push("Más grietas en inferior: reduce golpes post-tamper, baja el ramp-up de presión y evalúa paper filter abajo.");
  }
  if (top.metrics.extremes > bottom.metrics.extremes + 0.05){
    B.push("La cara superior presenta más extremos: mejora la distribución antes del tamper (WDT 10–15 s, romper grumos, RDT si hay estática).");
  }
  if (B.length===0){
    B.push("Puck consistente entre caras. Mantén parámetros y ajusta fino ratio/tiempo por sabor.");
  }
  return { lines: L, bullets: B };
}

// Build composite image (base + selected overlays + optional sector highlights) -> dataURL
async function composeImage(urls, overlays, highlightURL){
  const baseImg = new Image(); baseImg.src = urls.croppedURL; await baseImg.decode().catch(()=>{});
  const can = document.createElement("canvas");
  can.width = baseImg.width; can.height = baseImg.height;
  const ctx = can.getContext("2d");
  ctx.drawImage(baseImg,0,0);
  for (const u of overlays){
    if (!u) continue;
    const img = new Image(); img.src = u; await img.decode().catch(()=>{});
    ctx.drawImage(img,0,0,can.width,can.height);
  }
  if (highlightURL){
    const img = new Image(); img.src = highlightURL; await img.decode().catch(()=>{});
    ctx.drawImage(img,0,0,can.width,can.height);
  }
  return can.toDataURL("image/png");
}

export default function App(){
  const [sets, setSets] = useState([]);
  const [busy, setBusy] = useState(false);
  const [overlay, setOverlay] = useState({ heatmap:false, guides:true, edges:true, sectors:false, ring:true, coinc:false });
  const [rotateBack, setRotateBack] = useState(true);
  const topRef = useRef(); const bottomRef = useRef();
  const topPrintRef = useRef(); const bottomPrintRef = useRef();

  async function addSet(){
    const fTop = topRef.current.files?.[0];
    const fBottom = bottomRef.current.files?.[0];
    if(!fTop || !fBottom){ alert("Sube una imagen SUPERIOR y otra INFERIOR del puck."); return; }
    setBusy(true);
    try{
      const [top, bottomRaw] = await Promise.all([analyzeImageFile(fTop, "puck"), analyzeImageFile(fBottom, "puck")]);

      // Sector alignment to find coincident sectors
      const a = Array.from(top.arrays.sectorMeans);
      const bRaw = Array.from(bottomRaw.arrays.sectorMeans);
      const align = alignSectorMeans(a, bRaw, true);
      const sectorCorr = Math.max(0, Math.min(1, (align.corr+1)/2));
      // Compute coincident indices (> |z| thresh in both, same sign)
      const zA = zScore(a), zB = zScore(align.aligned);
      const TH = 1.0;
      const coincDark = [], coincLight = [];
      for (let i=0;i<a.length;i++){
        if (zA.z[i] <= -TH && zB.z[i] <= -TH) coincDark.push(i);
        if (zA.z[i] >= TH && zB.z[i] >= TH) coincLight.push(i);
      }
      // Draw highlight overlays for both faces (map indices back to bottomRaw indexing)
      async function makeHighlights(face, indicesDark, indicesLight, sectors=24){
        const size = face.dim.w, r = face.dim.r;
        const can = drawSectorHighlights(size, r, sectors, indicesDark, indicesLight);
        return await canvasToURL(can);
      }
      const hiTopURL = await makeHighlights(top, coincDark, coincLight, a.length);
      // Map indices to raw bottom:
      function unalignIndex(i){
        // reversed? shift?
        let idx = i;
        if (align.reversed){
          idx = (a.length - 1 - idx);
        }
        idx = (idx + align.shift) % a.length;
        return idx;
      }
      const darkBottom = coincDark.map(unalignIndex);
      const lightBottom = coincLight.map(unalignIndex);

      const hiBottomURL = await makeHighlights(bottomRaw, darkBottom, lightBottom, a.length);

      // visual rotation (optional) for bottom face
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

      const pair = { sectorCorr, align };
      const rec = pairRecommendations(top, bottomRaw, pair);

      // Optional paper prints
      let topPrint = null, bottomPrint = null;
      const fTopP = topPrintRef.current.files?.[0];
      const fBottomP = bottomPrintRef.current.files?.[0];
      if (fTopP) topPrint = await analyzeImageFile(fTopP, "print");
      if (fBottomP) bottomPrint = await analyzeImageFile(fBottomP, "print");

      setSets(prev=>[...prev, {
        top, bottom, bottomRaw, pair, rec,
        highlights: { top: hiTopURL, bottom: hiBottomURL },
        topPrint, bottomPrint
      }]);
      topRef.current.value = null; bottomRef.current.value = null;
      if (topPrintRef.current) topPrintRef.current.value = null;
      if (bottomPrintRef.current) bottomPrintRef.current.value = null;
    } finally {
      setBusy(false);
    }
  }

  async function exportPDF(index){
    const st = sets[index];
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const W = doc.internal.pageSize.getWidth();
    const M = 36;
    let y = M;

    doc.setFontSize(16); doc.text(`Puck Diagnosis — Set #${index+1}`, M, y); y+=18;
    doc.setFontSize(10);
    doc.text(`Correlación sectores (ajustada): ${Math.round(st.pair.sectorCorr*100)}%`, M, y); y+=14;
    doc.text(`Superior — ECI: ${round(st.top.metrics.eci)} | Extremos: ${(st.top.metrics.extremes*100).toFixed(1)}% | σ: ${round(st.top.metrics.sectorStd)} | Grietas: ${(st.top.metrics.edgeDensity*100).toFixed(2)}%`, M, y); y+=12;
    doc.text(`Inferior — ECI: ${round(st.bottomRaw.metrics.eci)} | Extremos: ${(st.bottomRaw.metrics.extremes*100).toFixed(1)}% | σ: ${round(st.bottomRaw.metrics.sectorStd)} | Grietas: ${(st.bottomRaw.metrics.edgeDensity*100).toFixed(2)}% | Anillo: prof ${round(st.bottomRaw.metrics.ringDepth)} @ r≈${Math.round(st.bottomRaw.metrics.ringPos*100)}%`, M, y); y+=16;

    // Compose display images with overlays + highlights
    const oTop = [];
    const oBottom = [];
    if (overlay.heatmap) { oTop.push(st.top.urls.hmURL); oBottom.push(st.bottom.urls.hmURL); }
    if (overlay.guides)  { oTop.push(st.top.urls.guideURL); oBottom.push(st.bottom.urls.guideURL); }
    if (overlay.edges)   { oTop.push(st.top.urls.edgeURL); oBottom.push(st.bottom.urls.edgeURL); }
    if (overlay.sectors) { oTop.push(st.top.urls.sectorURL); oBottom.push(st.bottom.urls.sectorURL); }
    if (overlay.ring)    { oTop.push(st.top.urls.ringURL); oBottom.push(st.bottom.urls.ringURL); }
    if (overlay.coinc)   { /* highlights added later in compose */ }

    const topDataURL = await composeImage(st.top.urls, oTop, overlay.coinc ? st.highlights.top : null);
    const bottomDataURL = await composeImage(st.bottom.urls, oBottom, overlay.coinc ? st.highlights.bottom : null);

    const imgW = (W - M*3)/2;
    const imgH = imgW; // square
    doc.addImage(topDataURL, "PNG", M, y, imgW, imgH);
    doc.addImage(bottomDataURL, "PNG", M*2+imgW, y, imgW, imgH);
    y += imgH + 18;

    // Recommendations (wrap)
    doc.setFontSize(12); doc.text("Recomendaciones:", M, y); y+=14;
    doc.setFontSize(10);
    const wrapWidth = W - M*2;
    const allRec = st.rec.bullets.join(" • ");
    const split = doc.splitTextToSize(allRec, wrapWidth);
    doc.text(split, M, y);
    y += 14 * split.length;

    // Second page for paper prints if present
    if (st.topPrint || st.bottomPrint){
      doc.addPage();
      let y2 = M;
      doc.setFontSize(14); doc.text("Huellas en papel", M, y2); y2+=18;
      const imgW2 = (W - M*3)/2;
      const imgH2 = imgW2;
      if (st.topPrint){
        const d1 = await composeImage(st.topPrint.urls, [overlay.sectors? st.topPrint.urls.sectorURL : null, overlay.ring? st.topPrint.urls.ringURL:null], null);
        doc.addImage(d1, "PNG", M, y2, imgW2, imgH2);
        doc.setFontSize(10); doc.text("Superior (papel)", M, y2+imgH2+12);
      }
      if (st.bottomPrint){
        const d2 = await composeImage(st.bottomPrint.urls, [overlay.sectors? st.bottomPrint.urls.sectorURL : null, overlay.ring? st.bottomPrint.urls.ringURL:null], null);
        doc.addImage(d2, "PNG", M*2+imgW2, y2, imgW2, imgH2);
        doc.setFontSize(10); doc.text("Inferior (papel)", M*2+imgW2, y2+imgH2+12);
      }
    }

    doc.save(`puck-report-set-${index+1}.pdf`);
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
          <label className="row small" style={{gap:8}} title="Marca los sectores problemáticos que coinciden en ambas caras">
            <input type="checkbox" checked={overlay.coinc} onChange={e=>setOverlay(o=>({...o, coinc:e.target.checked}))} />
            <span>Sectores coincidentes</span>
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
            <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", gap:12}}>
              <div style={{fontWeight:600}} className="small">Puck #{idx+1}</div>
              <button className="btn" onClick={()=>exportPDF(idx)}>Exportar PDF</button>
            </div>

            {/* Dos columnas: superior / inferior */}
            <div className="grid" style={{gridTemplateColumns:"1fr 1fr", gap:16, marginTop:10}}>
              {[{lab:"Superior (group head)", it: st.top, hi: st.highlights.top}, {lab:"Inferior (canasta)", it: st.bottom, hi: st.highlights.bottom}].map((obj, k)=>(
                <div key={k}>
                  <div className="small" style={{fontWeight:700, marginBottom:6}}>{obj.lab}</div>
                  <div className="thumb canvas-stack">
                    <img src={obj.it.urls.croppedURL} alt={obj.lab} />
                    {overlay.heatmap && <img className="overlay" src={obj.it.urls.hmURL} alt="heatmap" />}
                    {overlay.guides && <img className="overlay" src={obj.it.urls.guideURL} alt="guides" />}
                    {overlay.edges && <img className="overlay" src={obj.it.urls.edgeURL} alt="edges" />}
                    {overlay.sectors && <img className="overlay" src={obj.it.urls.sectorURL} alt="sectors" />}
                    {overlay.ring && <img className="overlay" src={obj.it.urls.ringURL} alt="ring" />}
                    {overlay.coinc && <img className="overlay" src={obj.hi} alt="coinc" />}
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
        * Huellas en papel: el centrado ahora se afina por <b>ajuste de círculo</b> (Kåsa) sobre el borde → evita desplazamientos.
      </div>
    </div>
  );
}
