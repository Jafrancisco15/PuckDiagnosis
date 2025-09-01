import React, { useEffect, useRef, useState } from "react";

/** =================== Utils =================== */
const MAX_DIM = 1200;
const clamp01 = x => Math.min(1, Math.max(0, x));
const lerp = (a,b,t)=> a+(b-a)*t;
const toGray = (r,g,b)=> (0.2126*r + 0.7152*g + 0.0722*b)/255;
const round = n => Math.round(n*1000)/1000;

async function fileToImageBitmap(file) {
  const url = URL.createObjectURL(file);
  const img = await createImageBitmap(await (await fetch(url)).blob());
  URL.revokeObjectURL(url);
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
  return canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
}
function grayscale(imgData) {
  const { data, width, height } = imgData;
  const gray = new Float32Array(width * height);
  let sum=0, sum2=0;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const g = toGray(data[i], data[i+1], data[i+2]);
    gray[j] = g; sum += g; sum2 += g*g;
  }
  const n = width*height;
  const mean = sum / n;
  const variance = sum2/n - mean*mean;
  return { gray, width, height, mean, std: Math.sqrt(Math.max(variance,1e-8)) };
}
function darkMask({gray,width,height,mean,std}, k=0.35) {
  const thr = mean - k*std;
  const mask = new Uint8Array(width*height);
  for (let i=0;i<gray.length;i++) mask[i] = gray[i] < thr ? 1 : 0;
  return { mask, width, height, thr };
}
function centroid(mask,width,height) {
  let sx=0, sy=0, c=0;
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){ const i=y*width+x; if (mask[i]){ sx+=x; sy+=y; c++; } }
  if (c===0) return {cx: width/2, cy: height/2, count:0};
  return {cx: sx/c, cy: sy/c, count:c};
}
function robustRadius(mask, width, height, cx, cy) {
  const dists = [];
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){ const i=y*width+x; if (mask[i]) dists.push(Math.hypot(x-cx,y-cy)); }
  if (dists.length<20) return Math.min(width,height)/2 * 0.48;
  dists.sort((a,b)=>a-b);
  return dists[Math.floor(dists.length*0.80)]; // p80
}
function cropCircleFromCanvas(srcCanvas, cx, cy, r, pad=1.18){
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
  const id = ctx.getImageData(0,0,size,size);
  const data = id.data;
  for (let y=0;y<size;y++) for (let x=0;x<size;x++){
    const dx2 = x-size/2, dy2=y-size/2;
    if (dx2*dx2 + dy2*dy2 > r*r){ data[(y*size + x)*4 + 3] = 0; }
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
      mag[y*width+x] = Math.hypot(gx,gy);
    }
  }
  let s=0, s2=0, c=0;
  for (let i=0;i<mag.length;i++){ const v=mag[i]; if (v>0){ s+=v; s2+=v*v; c++; } }
  const mean = c? s/c : 0;
  const std = c? Math.sqrt(Math.max(s2/c - mean*mean, 1e-8)) : 1;
  return { mag, mean, std };
}
function drawEdgesOverlay(gray, width, height, cx, cy, r){
  const { mag, mean, std } = sobel(gray, width, height);
  const can = document.createElement("canvas");
  can.width = width; can.height = height;
  const ctx = can.getContext("2d");
  const id = ctx.createImageData(width, height);
  const thr = mean + 1.0*std;
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
  return { can, density };
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
      } else varmap[idx]=0;
    }
  }
  return { varmap, wBlocks, hBlocks, vmax, bw, bh };
}
function detectDarkRing(profile, band=[0.78,0.96]){
  const n = profile.length;
  const i0 = Math.floor(band[0]*n);
  const i1 = Math.min(n-1, Math.floor(band[1]*n));
  let minV = Infinity, minI = i0;
  for (let i=i0;i<=i1;i++){ const v=profile[i]; if (v<minV){ minV=v; minI=i; } }
  return { minV, pos: minI / n };
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
  ctx.fillStyle="rgba(0,0,0,0.5)"; ctx.fillRect(8, 8, 190, 20);
  ctx.fillStyle="rgba(255,255,255,0.9)"; ctx.font="12px ui-sans-serif"; ctx.fillText("Anillo oscuro (estimado)", 12, 22);
  return can;
}
function drawSectorsOverlay(gray, width, height, cx, cy, r, sectors=24){
  const sums = new Float64Array(sectors);
  const cnts = new Uint32Array(sectors);
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy; const rr=Math.hypot(dx,dy);
      if (rr<=r){
        const a=Math.atan2(dy,dx);
        let k=Math.floor(((a + Math.PI)/(2*Math.PI))*sectors);
        if (k<0) k=0; if (k>=sectors) k=sectors-1;
        sums[k]+=gray[y*width+x]; cnts[k]++;
      }
    }
  }
  const means = new Float64Array(sectors);
  let globalS=0, globalC=0;
  for (let i=0;i<sectors;i++){ means[i]=cnts[i]?sums[i]/cnts[i]:0; globalS+=sums[i]; globalC+=cnts[i]; }
  const globalMean = globalC? globalS/globalC : 0;
  let s2=0; for (let i=0;i<sectors;i++){ const d=means[i]-globalMean; s2+=d*d; }
  const std = Math.sqrt(Math.max(s2/sectors,1e-8));
  const can = document.createElement("canvas"); can.width=width; can.height=height;
  const ctx = can.getContext("2d"); ctx.translate(cx,cy); ctx.globalAlpha=0.28;
  const step = (Math.PI*2)/sectors;
  for (let k=0;k<sectors;k++){
    const dev = means[k]-globalMean;
    const t = clamp01(Math.abs(dev) / (std*1.2 + 1e-6));
    let color = dev>0 ? `rgba(59,130,246,${0.25 + 0.5*t})` : `rgba(239,68,68,${0.25 + 0.5*t})`;
    ctx.fillStyle=color; ctx.beginPath(); ctx.moveTo(0,0); const a0=k*step, a1=(k+1)*step; ctx.arc(0,0,r,a0,a1); ctx.closePath(); ctx.fill();
  }
  ctx.setTransform(1,0,0,1,0,0); ctx.globalAlpha=1;
  const pad=8; ctx.fillStyle="rgba(0,0,0,0.45)"; ctx.fillRect(pad, pad, 230, 52);
  ctx.strokeStyle="rgba(255,255,255,0.25)"; ctx.strokeRect(pad, pad, 230, 52);
  ctx.font="12px ui-sans-serif"; ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("Mapa de sectores (desv. angular)", pad+8, pad+16);
  ctx.fillStyle="rgba(239,68,68,0.9)"; ctx.fillRect(pad+8, pad+24, 16, 12);
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("rojo: más oscuro (atascos)", pad+28, pad+34);
  ctx.fillStyle="rgba(59,130,246,0.9)"; ctx.fillRect(pad+8, pad+38, 16, 12);
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("azul: más claro (flujo rápido)", pad+28, pad+48);
  return { can, means, globalMean, std };
}
function drawProfileChart(profile){
  const W=720, H=190, pad=8;
  const can=document.createElement("canvas"); can.width=W; can.height=H;
  const ctx=can.getContext("2d");
  ctx.fillStyle="rgba(255,255,255,0.05)"; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle="rgba(255,255,255,0.15)"; for(let i=0;i<5;i++){ const y=pad+(H-2*pad)*i/4; ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(W-pad,y); ctx.stroke(); }
  const min=Math.min(...profile), max=Math.max(...profile);
  const nx=i=> pad+(W-2*pad)*(i/(profile.length-1));
  const ny=v=> pad+(H-2*pad)*(1-(v-min)/(max-min+1e-6));
  ctx.strokeStyle="#60a5fa"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(nx(0),ny(profile[0]));
  for (let i=1;i<profile.length;i++) ctx.lineTo(nx(i),ny(profile[i])); ctx.stroke();
  ctx.fillStyle="rgba(255,255,255,0.8)"; ctx.font="12px ui-sans-serif"; ctx.fillText("Perfil radial (0 centro → 1 borde)",10,16);
  return can;
}
function drawHistogram(gray, width, height, cx, cy, r){
  const bins=40; const hist=new Uint32Array(bins);
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){ const dx=x-cx, dy=y-cy; if (dx*dx+dy*dy<=r*r){ const g=gray[y*width+x]; const b=Math.max(0, Math.min(bins-1, Math.floor(g*bins))); hist[b]++; }}
  const W=720, H=190, pad=8; const can=document.createElement("canvas"); can.width=W; can.height=H; const ctx=can.getContext("2d");
  ctx.fillStyle="rgba(255,255,255,0.05)"; ctx.fillRect(0,0,W,H);
  const maxv=Math.max(...hist); const bw=(W-2*pad)/bins; ctx.fillStyle="#22c55e";
  for (let i=0;i<bins;i++){ const h=(H-2*pad)*(hist[i]/(maxv||1)); ctx.fillRect(pad+i*bw, H-pad-h, Math.max(1,bw-1), h); }
  ctx.fillStyle="rgba(255,255,255,0.8)"; ctx.font="12px ui-sans-serif"; ctx.fillText("Histograma de intensidades (puck)",10,16);
  return can;
}
async function canvasToURL(can){ const b=await new Promise(res=>can.toBlob(res,"image/png")); return URL.createObjectURL(b); }

/* ========= Filtros base ========= */
function gaussianKernel1D(sigma){
  const r = Math.max(1, Math.round(3*sigma));
  const k = new Float32Array(2*r+1);
  const s2 = 2*sigma*sigma;
  let sum = 0;
  for (let i=-r;i<=r;i++){ const v = Math.exp(-(i*i)/s2); k[i+r]=v; sum+=v; }
  for (let i=0;i<k.length;i++) k[i]/=sum;
  return {k, r};
}
function separableBlur(gray,w,h,sigma){
  const {k,r} = gaussianKernel1D(sigma);
  const tmp = new Float32Array(w*h);
  const out = new Float32Array(w*h);
  // X
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      let s=0;
      for (let i=-r;i<=r;i++){
        const xx = Math.min(w-1, Math.max(0, x+i));
        s += gray[y*w+xx] * k[i+r];
      }
      tmp[y*w+x]=s;
    }
  }
  // Y
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      let s=0;
      for (let i=-r;i<=r;i++){
        const yy = Math.min(h-1, Math.max(0, y+i));
        s += tmp[yy*w+x] * k[i+r];
      }
      out[y*w+x]=s;
    }
  }
  return out;
}
function laplace3x3(gray,w,h){
  const out = new Float32Array(w*h);
  for (let y=1;y<h-1;y++){
    for (let x=1;x<w-1;x++){
      const p=y*w+x;
      out[p] = -4*gray[p] + gray[p-1] + gray[p+1] + gray[p-w] + gray[p+w];
    }
  }
  return out;
}

/* === Helpers para selección dentro del disco, percentiles y Hessiano === */
function collectInside(arr, w, h, cx, cy, r){
  const vals=[]; const r2=r*r;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++){
    const dx=x-cx, dy=y-cy; if (dx*dx+dy*dy<=r2) vals.push(arr[y*w+x]);
  }
  return vals;
}
function percentile(vals, p){
  if (!vals.length) return 0;
  const a = [...vals].sort((u,v)=>u-v);
  const i = Math.min(a.length-1, Math.max(0, Math.round((a.length-1)*p)));
  return a[i];
}
function hessianAtScale(gray,w,h,sigma){
  const b = separableBlur(gray,w,h,sigma);
  const s2 = sigma*sigma;
  const dxx = new Float32Array(w*h), dyy=new Float32Array(w*h), dxy=new Float32Array(w*h);
  for (let y=1;y<h-1;y++){
    for (let x=1;x<w-1;x++){
      const p=y*w+x;
      dxx[p] = (b[p-1] - 2*b[p] + b[p+1]) * s2;
      dyy[p] = (b[p-w] - 2*b[p] + b[p+w]) * s2;
      dxy[p] = (b[p+w+1] - b[p+w-1] - b[p-w+1] + b[p-w-1]) * 0.25 * s2;
    }
  }
  return { dxx, dyy, dxy };
}

/* === Normalización de iluminación/contraste (retinex simple) ========== */
function normalizeInsideDisk(arr,w,h,cx,cy,r, pLow=0.02, pHigh=0.98){
  const vals = collectInside(arr,w,h,cx,cy,r).sort((a,b)=>a-b);
  const lo = vals[Math.floor(vals.length*pLow)] ?? 0;
  const hi = vals[Math.floor(vals.length*pHigh)] ?? 1;
  const out = new Float32Array(arr.length);
  for (let i=0;i<arr.length;i++) out[i] = clamp01((arr[i]-lo)/Math.max(hi-lo,1e-6));
  return out;
}
function retinexNormalize(gray,w,h,cx,cy,r){
  const sigma = Math.max(6, Math.round(Math.min(w,h)/14));
  const bg = separableBlur(gray,w,h,sigma);
  const out = new Float32Array(gray.length);
  for (let i=0;i<gray.length;i++){
    out[i] = Math.log(gray[i]+1e-3) - Math.log(bg[i]+1e-3);
  }
  return normalizeInsideDisk(out,w,h,cx,cy,r, 0.02, 0.98);
}

/* ===== Helper: overlay sombreado a partir de un “score” por pixel ===== */
function shadedOverlayFromScore(score, w, h, cx, cy, r, rgb=[255,0,0], baseAlpha=0.45, pLow=0.97, pHigh=0.995){
  const can = document.createElement("canvas");
  can.width = w; can.height = h;
  const ctx = can.getContext("2d");
  const id = ctx.createImageData(w, h);

  const vals = [];
  const r2 = r*r;
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      const i=y*w+x;
      const dx=x-cx, dy=y-cy;
      if (dx*dx+dy*dy<=r2) vals.push(score[i]);
    }
  }
  if (!vals.length){ ctx.putImageData(id,0,0); return { can, areaPct: 0 }; }

  vals.sort((a,b)=>a-b);
  const lo = vals[Math.max(0, Math.floor(vals.length*pLow))];
  const hi = vals[Math.max(0, Math.floor(vals.length*pHigh))];
  const denom = Math.max(hi - lo, 1e-9);

  let Ainside=0, Aactive=0;
  for (let y=0;y<h;y++){
    for (let x=0;x<w;x++){
      const i=y*w+x, p=4*i;
      const dx=x-cx, dy=y-cy;
      const inside = (dx*dx+dy*dy)<=r2;
      if (inside) {
        Ainside++;
        const v = score[i];
        const t = clamp01((v - lo) / denom);     // 0 → nada, 1 → muy fuerte
        const a = Math.round(255 * baseAlpha * t);
        if (a>0){
          id.data[p]   = rgb[0];
          id.data[p+1] = rgb[1];
          id.data[p+2] = rgb[2];
          id.data[p+3] = a;
          Aactive++;
        }
      }
    }
  }
  ctx.putImageData(id,0,0);
  const areaPct = Ainside ? (Aactive / Ainside) : 0;
  return { can, areaPct };
}

/* === RELIEVE: huecos (LoG−) y bultos (LoG+) con zonas sombreadas ===== */
function detectRelief(grayRaw, w, h, cx, cy, r){
  const gray = retinexNormalize(grayRaw, w, h, cx, cy, r);
  const sigmas = [1.0, 1.8, 3.0, 4.2];
  const minResp = new Float32Array(w*h).fill(+Infinity);
  const maxResp = new Float32Array(w*h).fill(-Infinity);

  for (const s of sigmas){
    const b = separableBlur(gray, w, h, s);
    const lap = laplace3x3(b, w, h);
    for (let i=0;i<lap.length;i++){
      if (lap[i] < minResp[i]) minResp[i] = lap[i];
      if (lap[i] > maxResp[i]) maxResp[i] = lap[i];
    }
  }

  const holeScore = new Float32Array(w*h);
  const bumpScore = new Float32Array(w*h);
  for (let i=0;i<holeScore.length;i++){
    holeScore[i] = Math.max(0, -minResp[i]);   // cóncavo
    bumpScore[i] = Math.max(0,  maxResp[i]);   // convexo
  }

  const holesOv = shadedOverlayFromScore(holeScore, w, h, cx, cy, r, [245,158,11], 0.42, 0.97, 0.995); // naranja
  const bumpsOv = shadedOverlayFromScore(bumpScore, w, h, cx, cy, r, [34,211,238], 0.42, 0.97, 0.995); // cian

  return {
    holesCan: holesOv.can,
    bumpsCan: bumpsOv.can,
    holesAreaPct: holesOv.areaPct,
    bumpsAreaPct: bumpsOv.areaPct,
    holesCount: Math.round(holesOv.areaPct * Math.PI * r * r / 200)
  };
}

/* === GRIETAS (Frangi-like) con zonas sombreadas ====================== */
function detectCracksHessian(grayRaw, w, h, cx, cy, r){
  const gray = retinexNormalize(grayRaw, w, h, cx, cy, r);
  const sigmas=[0.8,1.4,2.2,3.2];
  const V = new Float32Array(w*h);

  const r2 = r*r, beta=0.5, c=0.30;
  for (const s of sigmas){
    const {dxx,dyy,dxy} = hessianAtScale(gray, w, h, s);
    for (let y=1;y<h-1;y++){
      for (let x=1;x<w-1;x++){
        const i=y*w+x;
        const dx=x-cx, dy=y-cy;
        if (dx*dx+dy*dy>r2) continue;

        const A=dxx[i], B=dxy[i], D=dyy[i];
        const tr = A + D;
        const disc = Math.sqrt(Math.max((A-D)*(A-D) + 4*B*B, 0));
        const l1 = 0.5*(tr - disc); // λ1 ≤ λ2
        const l2 = 0.5*(tr + disc);

        if (l1 >= 0) continue; // buscamos líneas oscuras
        const Rb = Math.abs(l2) / (Math.abs(l1) + 1e-9);
        const S  = Math.hypot(l1,l2);
        const v = Math.exp(-(Rb*Rb)/(2*beta*beta)) * (1 - Math.exp(-(S*S)/(2*c*c)));
        if (v > V[i]) V[i] = v;
      }
    }
  }

  const cracksOv = shadedOverlayFromScore(V, w, h, cx, cy, r, [239,68,68], 0.50, 0.97, 0.995); // rojo
  return { can: cracksOv.can, cracksPct: cracksOv.areaPct };
}

/* ========= Granularidad y headspace (como antes) ===================== */
function boxBlur1D(arr, w, h, r, horiz){
  const out=new Float32Array(arr.length); const R=Math.max(1, Math.floor(r));
  if (horiz){
    for (let y=0;y<h;y++){
      let s=0; for (let x=-R;x<=R;x++){ const xx=Math.min(w-1, Math.max(0,x)); s+=arr[y*w+xx]; }
      for (let x=0;x<w;x++){
        const i=y*w+x;
        out[i]=s/(2*R+1);
        const x0=x-R, x1=x+R+1;
        s += arr[y*w + Math.min(w-1, x1)] - arr[y*w + Math.max(0, x0)];
      }
    }
  } else {
    for (let x=0;x<w;x++){
      let s=0; for (let y=-R;y<=R;y++){ const yy=Math.min(h-1, Math.max(0,y)); s+=arr[yy*w+x]; }
      for (let y=0;y<h;y++){
        const i=y*w+x;
        out[i]=s/(2*R+1);
        const y0=y-R, y1=y+R+1;
        s += arr[Math.min(h-1, y1)*w + x] - arr[Math.max(0, y0)*w + x];
      }
    }
  }
  return out;
}
function blur(arr, w, h, r, it=2){
  let tmp=arr;
  for (let k=0;k<it;k++){ tmp=boxBlur1D(tmp,w,h,r,true); tmp=boxBlur1D(tmp,w,h,r,false); }
  return tmp;
}
function granularityMetrics(gray, width, height, cx, cy, r){
  const L = new Float32Array(gray.length); for (let i=0;i<L.length;i++) L[i]=gray[i];
  const b1=blur(L,width,height,1.5,1);
  const b2=blur(L,width,height,5,2);
  let E_f=0,E_c=0,area=0;
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){
    const dx=x-cx, dy=y-cy; if (dx*dx+dy*dy<=r*r){ const i=y*width+x; const hf=L[i]-b1[i]; const lf=b1[i]-b2[i]; E_f += hf*hf; E_c += lf*lf; area++; }
  }
  E_f/=Math.max(area,1); E_c/=Math.max(area,1);
  const BI = E_c/(E_f+1e-6);
  return { fines:E_f, coarse:E_c, bimodality:BI };
}
function headspaceRisk(m){
  const depth = Math.max(0, m.ringDepth);
  const pos = m.ringPos || 0;
  const edge = m.edgeDensity || 0;
  const extremes = m.extremes || 0;
  const edgePos = clamp01((pos-0.9)/0.08);
  const score = clamp01( 0.6*(depth/0.06)*edgePos + 0.25*edge*3 + 0.15*extremes*2 );
  return score;
}
function corrPearson(a,b){
  const n=Math.min(a.length,b.length); let sx=0,sy=0,sxx=0,syy=0,sxy=0;
  for (let i=0;i<n;i++){ const x=a[i], y=b[i]; sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; }
  const cov=sxy/n - (sx/n)*(sy/n); const vx=sxx/n - (sx/n)*(sx/n); const vy=syy/n - (sy/n)*(sy/n);
  const denom=Math.sqrt(Math.max(vx,1e-12)*Math.max(vy,1e-12)); return denom>0? cov/denom : 0;
}
function alignSectorMeans(a,b,allowReverse=true){
  const n=a.length; let best={corr:-2,shift:0,reversed:false,aligned:null};
  const shiftArr=(arr,k)=>arr.map((_,i)=>arr[(i-k+n)%n]);
  const variants=[{arr:b,reversed:false}]; if (allowReverse) variants.push({arr:[...b].reverse(),reversed:true});
  for (const v of variants){ for (let k=0;k<n;k++){ const bb=shiftArr(v.arr,k); const c=corrPearson(a,bb); if (c>best.corr) best={corr:c,shift:k,reversed:v.reversed,aligned:bb}; } }
  return best;
}

/** =================== Analysis (puck) =================== */
async function analyzeFromCircle(srcCanvas, cx, cy, r){
  const cropped = cropCircleFromCanvas(srcCanvas, cx, cy, r, 1.18);
  const cid = cropped.getContext("2d").getImageData(0,0,cropped.width, cropped.height);
  const cg = grayscale(cid);
  const centerX = cropped.width/2, centerY = cropped.height/2;

  const stats = polarStats(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const lv = localVariance(cg.gray, cg.width, cg.height, centerX, centerY, r, 64);

  // Detectores (con retinex interno donde aplica)
  const edges = drawEdgesOverlay(retinexNormalize(cg.gray, cg.width, cg.height, centerX, centerY, r), cg.width, cg.height, centerX, centerY, r);
  const sectors = drawSectorsOverlay(retinexNormalize(cg.gray, cg.width, cg.height, centerX, centerY, r), cg.width, cg.height, centerX, centerY, r, 24);
  const heatmap = drawHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, lv.bw, lv.bh, cg.width, cg.height);
  const guides = (()=>{ const can=document.createElement("canvas"); can.width=cropped.width; can.height=cropped.height; const ctx=can.getContext("2d"); ctx.strokeStyle="rgba(255,255,255,0.45)"; ctx.lineWidth=1.2; const rings=[0.3,0.6,0.9]; rings.forEach(t=>{ ctx.beginPath(); ctx.arc(centerX,centerY,r*t,0,Math.PI*2); ctx.stroke(); }); return can; })();

  const ring = detectDarkRing(stats.profile, [0.78,0.96]);
  const ringDepth = (stats.mid - ring.minV);
  const ringOverlay = drawRingOverlay(cropped.width, r, ring.pos, 0.06);

  const profileChart = drawProfileChart(Array.from(stats.profile));
  const histChart = drawHistogram(cg.gray, cg.width, cg.height, centerX, centerY, r);

  // NUEVOS overlays sombreados
  const relief = detectRelief(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const cracks = detectCracksHessian(cg.gray, cg.width, cg.height, centerX, centerY, r);

  // extremos (z-score) dentro del disco
  let s=0,c=0; for (let y=0;y<cg.height;y++) for (let x=0;x<cg.width;x++){ const dx=x-centerX, dy=y-centerY; if (dx*dx+dy*dy<=r*r){ s+=cg.gray[y*cg.width+x]; c++; } }
  const mean=s/Math.max(c,1);
  let s2=0; for (let y=0;y<cg.height;y++) for (let x=0;x<cg.width;x++){ const dx=x-centerX, dy=y-centerY; if (dx*dx+dy*dy<=r*r){ const g=cg.gray[y*cg.width+x]; s2+=(g-mean)*(g-mean); } }
  const std=Math.sqrt(Math.max(s2/Math.max(c,1),1e-8));
  const low=mean-1.0*std, high=mean+1.0*std; let ext=0;
  for (let y=0;y<cg.height;y++) for (let x=0;x<cg.width;x++){ const dx=x-centerX, dy=y-centerY; if (dx*dx+dy*dy<=r*r){ const g=cg.gray[y*cg.width+x]; if (g<low||g>high) ext++; } }
  const extremes = ext/Math.max(c,1);

  const hsRisk = headspaceRisk({ ringDepth, ringPos: ring.pos, edgeDensity: edges.density, extremes });

  const [
    croppedURL, hmURL, edgeURL, sectorURL, guideURL, ringURL, profileURL, histURL,
    holesURL, bumpsURL, cracksURL
  ] = await Promise.all([
    canvasToURL(cropped), canvasToURL(heatmap), canvasToURL(edges.can),
    canvasToURL(sectors.can), canvasToURL(guides), canvasToURL(ringOverlay),
    canvasToURL(profileChart), canvasToURL(histChart),
    canvasToURL(relief.holesCan), canvasToURL(relief.bumpsCan), canvasToURL(cracks.can)
  ]);

  return {
    dim: { w: cropped.width, h: cropped.height, r: Math.round(r) },
    urls: { croppedURL, hmURL, edgeURL, sectorURL, guideURL, ringURL, profileURL, histURL, holesURL, bumpsURL, cracksURL },
    metrics: {
      eci: stats.eci,
      extremes,
      sectorStd: sectors.std,
      edgeDensity: edges.density,
      ringDepth, ringPos: ring.pos,
      granularity: granularityMetrics(cg.gray, cg.width, cg.height, centerX, centerY, r),
      headspace: hsRisk,
      holesDepthAreaPct: relief.holesAreaPct,
      bumpsAreaPct: relief.bumpsAreaPct,
      cracksPct: cracks.cracksPct
    },
    arrays: { sectorMeans: sectors.means, radialProfile: stats.profile },
    debug: { cx, cy, r }
  };
}

/** =================== Optional: Paper print (huella) =================== */
async function analyzePrintFromCircle(srcCanvas, cx, cy, r){
  const cropped = cropCircleFromCanvas(srcCanvas, cx, cy, r, 1.18);
  const id = cropped.getContext("2d").getImageData(0,0,cropped.width,cropped.height);
  const g = grayscale(id);
  const inv = new Float32Array(g.gray.length);
  for (let i=0;i<inv.length;i++) inv[i] = 1.0 - g.gray[i];

  const cx0=cropped.width/2, cy0=cropped.height/2;
  const ps = polarStats(inv, g.width, g.height, cx0, cy0, r);
  const lv = localVariance(inv, g.width, g.height, cx0, cy0, r, 64);
  const heatmap = drawHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, lv.bw, lv.bh, g.width, g.height);
  const sectors = drawSectorsOverlay(inv, g.width, g.height, cx0, cy0, r, 24);
  const guides = (()=>{ const can=document.createElement("canvas"); can.width=cropped.width; can.height=cropped.height; const ctx=can.getContext("2d"); ctx.strokeStyle="rgba(255,255,255,0.45)"; ctx.lineWidth=1.2; const rings=[0.3,0.6,0.9]; rings.forEach(t=>{ ctx.beginPath(); ctx.arc(cx0,cy0,r*t,0,Math.PI*2); ctx.stroke(); }); return can; })();

  const means = sectors.means;
  const mAvg = means.reduce((a,b)=>a+b,0)/means.length;
  const chMask = means.map(v=> v > mAvg + sectors.std*0.8 ? 1 : 0);
  const channelPct = chMask.reduce((a,b)=>a+b,0)/chMask.length;

  const profileChart = drawProfileChart(Array.from(ps.profile));
  const histChart = drawHistogram(inv, g.width, g.height, cx0, cy0, r);

  const [croppedURL, hmURL, sectorURL, guideURL, profileURL, histURL] = await Promise.all([
    canvasToURL(cropped), canvasToURL(heatmap), canvasToURL(sectors.can),
    canvasToURL(guides), canvasToURL(profileChart), canvasToURL(histChart)
  ]);

  return {
    urls:{ croppedURL, hmURL, sectorURL, guideURL, profileURL, histURL },
    metrics:{
      channelPct,
      sectorStd: sectors.std,
      radialEdgeBoost: ps.ringAvg(0.88,0.98) - ps.ringAvg(0.55,0.7)
    },
    arrays:{ sectorMeans: sectors.means, radialProfile: ps.profile }
  };
}

/** =================== UI helpers =================== */
function Pill({children}){ return <span className="pill">{children}</span>; }

/** =================== App =================== */
export default function App(){
  const [sets, setSets] = useState([]);
  const [busy, setBusy] = useState(false);
  const [overlay, setOverlay] = useState({ heatmap:false, guides:true, edges:true, sectors:false, ring:true, holes:true, bumps:false, cracks:true });
  const [expanded, setExpanded] = useState(0);

  const topRef = useRef(); const bottomRef = useRef();
  const topPrintRef = useRef(); const bottomPrintRef = useRef();

  const [adj, setAdj] = useState({ open:false, slot:null, file:null });
  const overrides = useRef({});

  function openAdjust(slot){
    const map = { top: topRef, bottom: bottomRef, topPrint: topPrintRef, bottomPrint: bottomPrintRef };
    const file = map[slot].current?.files?.[0];
    if (!file){ alert("Primero selecciona un archivo."); return; }
    setAdj({ open:true, slot, file });
  }

  function AdjustModal({open, onClose, file, label, onConfirm}){
    const canvasRef = useRef(null);
    const [raw, setRaw] = useState(null);
    const [cx, setCx] = useState(0.5);
    const [cy, setCy] = useState(0.5);
    const [rad, setRad] = useState(0.45);
    useEffect(()=>{
      if(!open || !file) return;
      (async()=>{
        const det = await (async()=>{
          const img = await fileToImageBitmap(file);
          const { canvas } = drawToCanvas(img, MAX_DIM);
          const id = getImageData(canvas);
          const g = grayscale(id);
          const m = darkMask(g, 0.35);
          const c = centroid(m.mask, m.width, m.height);
          const r = robustRadius(m.mask, m.width, m.height, c.cx, c.cy);
          return { src: canvas, width: canvas.width, height: canvas.height, cx: c.cx, cy: c.cy, r };
        })();
        setRaw(det);
        setCx(det.cx/det.width);
        setCy(det.cy/det.height);
        const rRel = det.r / Math.min(det.width, det.height);
        setRad(Math.min(0.49, rRel));
      })();
    }, [open, file]);
    useEffect(()=>{
      if (!raw || !open) return;
      const can = canvasRef.current;
      can.width = raw.width; can.height = raw.height;
      const ctx = can.getContext("2d");
      ctx.clearRect(0,0,can.width,can.height);
      ctx.drawImage(raw.src, 0, 0);
      const rpx = rad * Math.min(raw.width, raw.height);
      const px = cx * raw.width, py = cy * raw.height;
      ctx.save();
      ctx.strokeStyle = "rgba(59,130,246,0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, rpx, 0, Math.PI*2); ctx.stroke();
      ctx.restore();
    }, [raw, cx, cy, rad, open]);
    if (!open) return null;
    return (
      <div className="modal-backdrop" onClick={onClose}>
        <div className="modal" onClick={e=>e.stopPropagation()}>
          <header>
            <div style={{fontWeight:700}}>Ajustar recorte — {label}</div>
            <button className="btn" onClick={onClose}>Cerrar</button>
          </header>
          <div className="body">
            {!raw ? <div className="muted">Cargando...</div> : (
              <div className="preview">
                <div className="preview-canvas">
                  <canvas ref={canvasRef} style={{width:"100%",height:"100%"}} />
                </div>
                <div>
                  <div className="slider"><span className="small">X</span><input type="range" min={0.0} max={1.0} step={0.001} value={cx} onChange={e=>setCx(parseFloat(e.target.value))}/></div>
                  <div className="slider"><span className="small">Y</span><input type="range" min={0.0} max={1.0} step={0.001} value={cy} onChange={e=>setCy(parseFloat(e.target.value))}/></div>
                  <div className="slider"><span className="small">R</span><input type="range" min={0.2} max={0.49} step={0.001} value={rad} onChange={e=>setRad(parseFloat(e.target.value))}/></div>
                  <div className="muted small">Tip: mueve X/Y para centrar; ajusta R hasta el borde.</div>
                </div>
              </div>
            )}
          </div>
          <div className="footer">
            <button className="btn" onClick={()=>{
              if(!raw) return;
              const Rpx = rad * Math.min(raw.width, raw.height);
              onConfirm({ srcCanvas: raw.src, cx: cx*raw.width, cy: cy*raw.height, r: Rpx });
              onClose();
            }}>Usar este recorte</button>
          </div>
        </div>
      </div>
    );
  }

  async function addSet(){
    const fTop = topRef.current.files?.[0];
    const fBottom = bottomRef.current.files?.[0];
    if(!fTop || !fBottom){ alert("Sube una imagen SUPERIOR y otra INFERIOR del puck."); return; }
    setBusy(true);
    try{
      const cropOrDetect = async (slot, file)=>{
        if (overrides.current[slot]) return overrides.current[slot];
        const img = await fileToImageBitmap(file);
        const { canvas } = drawToCanvas(img, MAX_DIM);
        const id = getImageData(canvas);
        const g = grayscale(id);
        const m = darkMask(g, 0.35);
        const c = centroid(m.mask, m.width, m.height);
        const r = robustRadius(m.mask, m.width, m.height, c.cx, c.cy);
        return { srcCanvas: canvas, cx: c.cx, cy: c.cy, r };
      };
      const topCrop = await cropOrDetect("top", fTop);
      const bottomCrop = await cropOrDetect("bottom", fBottom);
      const top = await analyzeFromCircle(topCrop.srcCanvas, topCrop.cx, topCrop.cy, topCrop.r);
      const bottom = await analyzeFromCircle(bottomCrop.srcCanvas, bottomCrop.cx, bottomCrop.cy, bottomCrop.r);

      // Optional prints
      let topPrint=null, bottomPrint=null;
      const fTopP = topPrintRef.current.files?.[0];
      const fBotP = bottomPrintRef.current.files?.[0];
      if (fTopP){
        const det = await cropOrDetect("topPrint", fTopP);
        topPrint = await analyzePrintFromCircle(det.srcCanvas, det.cx, det.cy, det.r);
      }
      if (fBotP){
        const det = await cropOrDetect("bottomPrint", fBotP);
        bottomPrint = await analyzePrintFromCircle(det.srcCanvas, det.cx, det.cy, det.r);
      }

      const a = Array.from(top.arrays.sectorMeans);
      const b = Array.from(bottom.arrays.sectorMeans);
      const abCorr = (corrPearson(a,b)+1)/2;
      const ringFlag = bottom.metrics.ringDepth>0.02 ? "anillo" : "—";

      let paperSummary=null;
      if (topPrint || bottomPrint){
        const tp = topPrint?.arrays?.sectorMeans || a;
        const bp = bottomPrint?.arrays?.sectorMeans || b;
        const ppCorr = (corrPearson(tp,bp)+1)/2;
        paperSummary = {
          topChPct: topPrint?.metrics?.channelPct ?? null,
          botChPct: bottomPrint?.metrics?.channelPct ?? null,
          ppCorr
        };
      }

      const rec = buildRecommendations({top,bottom,abCorr,paperSummary});

      const setItem = {
        top, bottom, topPrint, bottomPrint, pair: { sectorCorr: abCorr },
        summary: {
          eciTop: round(top.metrics.eci),
          eciBot: round(bottom.metrics.eci),
          corr: Math.round(abCorr*100),
          ring: ringFlag,
          grind: round(top.metrics.granularity.bimodality)
        },
        rec
      };
      setSets(prev=> [setItem, ...prev]);
      setExpanded(0);
      if (topRef.current) topRef.current.value=null;
      if (bottomRef.current) bottomRef.current.value=null;
      overrides.current={};
    } finally { setBusy(false); }
  }

  function buildRecommendations(ctx){
    const { top, bottom, abCorr, paperSummary } = ctx;
    const lines = [];
    const bullets = [];

    if (bottom.metrics.holesDepthAreaPct > 0.010){
      lines.push("Depresiones/huecos visibles en la cara inferior.");
      bullets.push("Canasta moderna y tamper base plana auto-nivelante (>6 kg, sin golpes).");
      bullets.push("WDT suave para asentar en capa plana; evita cavidades.");
      bullets.push("Considera papel bajo el puck para sellar micro-huecos.");
    }
    if (top.metrics.bumpsAreaPct > 0.010 || bottom.metrics.bumpsAreaPct > 0.010){
      lines.push("Abultamientos localizados (bultos) en el puck.");
      bullets.push("Reduce sobre-compactación o golpes; revisa tornillo/ducha marcada.");
      bullets.push("Preinfusión más suave y evita rampas bruscas.");
    }

    if (bottom.metrics.cracksPct > 0.006 || top.metrics.cracksPct > 0.006){
      lines.push("Grietas lineales detectadas.");
      bullets.push("Evita golpes al acoplar y descargas de presión bruscas.");
      bullets.push("Tamp recto/constante; WDT menos profundo si usas agujas finas.");
    }

    if (bottom.metrics.ringDepth > 0.03 && bottom.metrics.eci > 0.02){
      lines.push("Anillo oscuro pronunciado (borde subextraído).");
      bullets.push("Más headspace (baja dosis o cesta más alta) y WDT hasta el borde sin barrer finos al perímetro.");
      bullets.push("Prueba puck-screen arriba y preinfusión 2–6 s.");
    }

    if (bottom.metrics.sectorStd > 0.015){
      lines.push("Alta desviación angular: sectores con caudal distinto.");
      bullets.push("Nivelación/WDT homogéneos; revisa ducha o caídas asimétricas.");
      bullets.push("Si persiste, papel sólo abajo para homogeneizar.");
    }

    if (bottom.metrics.edgeDensity > 0.015 || bottom.metrics.extremes > 0.20){
      lines.push("Zonas muy tensas o roturas internas.");
      bullets.push("Evita torsión en el tamp y picos de presión.");
    }

    if ((top.metrics.granularity?.bimodality ?? 0) > 0.9){
      lines.push("Molienda bimodal (mucho fino y grueso).");
      bullets.push("Cierra 1–2 clics y distribuye mejor; controla estática para reducir finos.");
    }

    if (abCorr < 0.55){
      lines.push("Baja correlación entre caras.");
      bullets.push("Refuerza nivelación; objetivo: ↑ correlación angular.");
    }

    if (paperSummary){
      const { topChPct, botChPct, ppCorr } = paperSummary;
      if (ppCorr !== null && ppCorr < 0.55){
        lines.push("Huella en papel no coincide entre caras; migraciones internas.");
        bullets.push("Papel abajo o menor caudal inicial.");
      }
      if ((botChPct ?? 0) > 0.3){
        lines.push("Huella inferior con muchos sectores rápidos (>30%).");
        bullets.push("Revisa perímetro y reduce presión pico.");
      }
    }

    if (lines.length===0) lines.push("Distribución razonablemente uniforme; afina parámetros.");
    if (bullets.length===0) bullets.push("Mantén WDT homogéneo, headspace 2–6 mm y evita sobre-compactar.");

    return { lines, bullets };
  }

  return (
    <div className="container">
      <div className="row" style={{justifyContent:"space-between", alignItems:"flex-start", gap:16}}>
        <div>
          <h1 style={{margin:"0 0 6px 0"}}>Puck Diagnosis</h1>
          <div className="muted">
            Sube <b>dos fotos por puck</b>: <span className="tag">Superior = lado ducha</span> <span className="tag">Inferior = lado canasta</span>.{" "}
            <span className="tag">Opcional: huella en papel (superior/inferior)</span>
          </div>
        </div>
        <div className="row">
          {[
            ["heatmap","Heatmap"],["guides","Guías"],["edges","Líneas"],["sectors","Sectores"],
            ["ring","Anillo"],["holes","Huecos"],["bumps","Bultos"],["cracks","Grietas"]
          ].map(([key,label])=>(
            <label key={key} className="row small" style={{gap:8}}>
              <input type="checkbox" checked={overlay[key]} onChange={e=>setOverlay(o=>({...o, [key]:e.target.checked}))}/>
              <span>{label}</span>
            </label>
          ))}
        </div>
      </div>

      <div style={{height:12}} />

      <div className="card">
        <div className="row" style={{gap:16, alignItems:"flex-end", flexWrap:"wrap"}}>
          <div>
            <div className="small">Superior (lado ducha)</div>
            <input ref={topRef} type="file" accept="image/*" />
            <div style={{marginTop:6}}><button className="btn" onClick={()=>openAdjust("top")}>Ajustar recorte</button></div>
          </div>
          <div>
            <div className="small">Inferior (lado canasta)</div>
            <input ref={bottomRef} type="file" accept="image/*" />
            <div style={{marginTop:6}}><button className="btn" onClick={()=>openAdjust("bottom")}>Ajustar recorte</button></div>
          </div>
          <div>
            <div className="small">Huella — Superior (opcional)</div>
            <input ref={topPrintRef} type="file" accept="image/*" />
            <div style={{marginTop:6}}><button className="btn" onClick={()=>openAdjust("topPrint")}>Ajustar recorte</button></div>
          </div>
          <div>
            <div className="small">Huella — Inferior (opcional)</div>
            <input ref={bottomPrintRef} type="file" accept="image/*" />
            <div style={{marginTop:6}}><button className="btn" onClick={()=>openAdjust("bottomPrint")}>Ajustar recorte</button></div>
          </div>
          <div style={{flex:1}} />
          <div><button className="btn" disabled={busy} onClick={addSet}>{busy? "Procesando..." : "Añadir set"}</button></div>
        </div>
      </div>

      <AdjustModal
        open={adj.open}
        label={
          adj.slot==="top" ? "Superior (ducha)" :
          adj.slot==="bottom" ? "Inferior (canasta)" :
          adj.slot==="topPrint" ? "Huella superior" : "Huella inferior"
        }
        file={adj.file}
        onClose={()=>setAdj({open:false, slot:null, file:null})}
        onConfirm={(ov)=>{ overrides.current[adj.slot]=ov; }}
      />

      <div style={{height:18}} />

      <div className="sets">
        {sets.map((st, idx)=>{
          const displayIndex = sets.length - idx;
          const isOpen = expanded===idx;
          return (
            <div key={idx} className="card">
              <div className="accordion-summary" onClick={()=>setExpanded(isOpen? -1 : idx)}>
                <div style={{fontWeight:700}}>Puck #{displayIndex}</div>
                <div className="summary-metrics small">
                  <Pill>ECI sup/inf: <span className="mono">{st.summary.eciTop}</span>/<span className="mono">{st.summary.eciBot}</span></Pill>
                  <Pill>Corr top/bot: <span className="mono">{st.summary.corr}%</span></Pill>
                  <Pill>Granul.: <span className="mono">{st.summary.grind}</span></Pill>
                  <Pill>Anillo: <span className="mono">{st.summary.ring}</span></Pill>
                </div>
              </div>

              {isOpen && (
                <div style={{marginTop:12}}>
                  <div className="grid" style={{gridTemplateColumns:"1fr 1fr", gap:16}}>
                    {[{lab:"Superior (ducha)", it: st.top}, {lab:"Inferior (canasta)", it: st.bottom}].map((obj, k)=>(
                      <div key={k}>
                        <div className="small" style={{fontWeight:700, marginBottom:6}}>{obj.lab}</div>
                        <div className="thumb canvas-stack">
                          <img src={obj.it.urls.croppedURL} alt={obj.lab} />
                          {overlay.heatmap && <img className="overlay" src={obj.it.urls.hmURL} alt="heatmap" />}
                          {overlay.guides && <img className="overlay" src={obj.it.urls.guideURL} alt="guides" />}
                          {overlay.edges && <img className="overlay" src={obj.it.urls.edgeURL} alt="edges" />}
                          {overlay.sectors && <img className="overlay" src={obj.it.urls.sectorURL} alt="sectors" />}
                          {overlay.ring && <img className="overlay" src={obj.it.urls.ringURL} alt="ring" />}
                          {overlay.holes && <img className="overlay" src={obj.it.urls.holesURL} alt="holes" />}
                          {overlay.bumps && <img className="overlay" src={obj.it.urls.bumpsURL} alt="bumps" />}
                          {overlay.cracks && <img className="overlay" src={obj.it.urls.cracksURL} alt="cracks" />}
                        </div>
                        <div style={{height:8}} />
                        <div className="metrics">
                          <div>ECI: <span className="mono">{obj.it.metrics.eci.toFixed(3)}</span></div>
                          <div>Extremos: <span className="mono">{(obj.it.metrics.extremes*100).toFixed(1)}%</span></div>
                          <div>σ sectores: <span className="mono">{obj.it.metrics.sectorStd.toFixed(3)}</span></div>
                          <div>Líneas (Sobel): <span className="mono">{(obj.it.metrics.edgeDensity*100).toFixed(2)}%</span></div>
                          <div>Grietas (Hessiano): <span className="mono">{(obj.it.metrics.cracksPct*100).toFixed(2)}%</span></div>
                          <div>Huecos%: <span className="mono">{(obj.it.metrics.holesDepthAreaPct*100).toFixed(2)}%</span></div>
                          <div>Bultos%: <span className="mono">{(obj.it.metrics.bumpsAreaPct*100).toFixed(2)}%</span></div>
                          <div>Anillo (prof.): <span className="mono">{obj.it.metrics.ringDepth.toFixed(3)}</span></div>
                          <div>Headspace (riesgo): <span className="mono">{(obj.it.metrics.headspace*100).toFixed(0)}%</span></div>
                          <div>Granul. BI: <span className="mono">{(obj.it.metrics.granularity?.bimodality||0).toFixed(2)}</span></div>
                        </div>
                        <div className="charts">
                          <div className="chart"><img src={obj.it.urls.profileURL} alt="perfil radial" /></div>
                          <div className="chart"><img src={obj.it.urls.histURL} alt="histograma" /></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {(st.topPrint || st.bottomPrint) && (
                    <div style={{marginTop:16}}>
                      <div style={{fontWeight:700, marginBottom:6}}>Huella en papel (opcional)</div>
                      <div className="grid" style={{gridTemplateColumns:"1fr 1fr", gap:16}}>
                        {[{lab:"Huella superior", it: st.topPrint}, {lab:"Huella inferior", it: st.bottomPrint}].map((obj, k)=>(
                          obj.it ? (
                            <div key={k}>
                              <div className="thumb canvas-stack">
                                <img src={obj.it.urls.croppedURL} alt={obj.lab} />
                                {overlay.heatmap && <img className="overlay" src={obj.it.urls.hmURL} alt="heatmap" />}
                                {overlay.sectors && <img className="overlay" src={obj.it.urls.sectorURL} alt="sectors" />}
                                {overlay.guides && <img className="overlay" src={obj.it.urls.guideURL} alt="guides" />}
                              </div>
                              <div className="metrics" style={{gridTemplateColumns:"repeat(3,1fr)"}}>
                                <div>Canales%: <span className="mono">{(obj.it.metrics.channelPct*100).toFixed(0)}%</span></div>
                                <div>σ sectores: <span className="mono">{obj.it.metrics.sectorStd.toFixed(3)}</span></div>
                                <div>Borde-medio: <span className="mono">{obj.it.metrics.radialEdgeBoost.toFixed(3)}</span></div>
                              </div>
                              <div className="charts">
                                <div className="chart"><img src={obj.it.urls.profileURL} alt="perfil radial" /></div>
                                <div className="chart"><img src={obj.it.urls.histURL} alt="histograma" /></div>
                              </div>
                            </div>
                          ) : <div key={k} className="muted small">({obj.lab} no cargada)</div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div style={{marginTop:16}}>
                    <div style={{fontWeight:700, marginBottom:6}}>Recomendaciones</div>
                    {ctxBlock(st.rec)}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  function ctxBlock(rec){
    return (
      <div>
        <ul style={{marginTop:0}}>
          {rec.lines.map((l,i)=><li key={i}>{l}</li>)}
        </ul>
        <div className="small muted" style={{margin:"6px 0"}}>Acciones sugeridas:</div>
        <ul>
          {rec.bullets.map((b,i)=><li key={i}>{b}</li>)}
        </ul>
      </div>
    );
  }
}
