import React, { useEffect, useRef, useState } from "react";

/** =================== Utilidades =================== */
const MAX_DIM = 1200;
const clamp01 = x => Math.min(1, Math.max(0, x));
const lerp = (a,b,t)=> a+(b-a)*t;
const toGray = (r,g,b)=> (0.2126*r + 0.7152*g + 0.0722*b)/255;
const round = n => Math.round(n*1000)/1000;

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
  const eci = edge - mid; // >0 = borde más claro; <0 = borde más oscuro
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
  ctx.fillStyle="rgba(0,0,0,0.5)"; ctx.fillRect(8, 8, 220, 20);
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
  const pad=8; ctx.fillStyle="rgba(0,0,0,0.45)"; ctx.fillRect(pad, pad, 260, 52);
  ctx.strokeStyle="rgba(255,255,255,0.25)"; ctx.strokeRect(pad, pad, 260, 52);
  ctx.font="12px ui-sans-serif"; ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("Mapa de sectores (desv. angular)", pad+8, pad+16);
  ctx.fillStyle="rgba(239,68,68,0.9)"; ctx.fillRect(pad+8, pad+24, 16, 12);
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("rojo: más oscuro (atasco)", pad+28, pad+34);
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

/** =================== NUEVO — Huecos / Pinholes =================== */
function drawHolesOverlay(gray, width, height, cx, cy, r){
  const can = document.createElement("canvas");
  can.width=width; can.height=height;
  const ctx=can.getContext("2d");

  let s=0,s2=0,c=0;
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){
    const dx=x-cx, dy=y-cy; if(dx*dx+dy*dy<=r*r){
      const g=gray[y*width+x]; s+=g; s2+=g*g; c++;
    }
  }
  const mean=s/Math.max(c,1), std=Math.sqrt(Math.max(s2/Math.max(c,1)-mean*mean,1e-8));
  const low=mean-1.6*std, high=mean+1.6*std;

  const id=ctx.createImageData(width,height);
  let holes=0;
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){
    const i=y*width+x; const dx=x-cx, dy=y-cy; const inside=(dx*dx+dy*dy<=r*r);
    const p=i*4; if(!inside){ id.data[p+3]=0; continue; }
    const g=gray[i];
    if (g<low){ // huecos oscuros
      id.data[p]=255; id.data[p+1]=200; id.data[p+2]=0; id.data[p+3]=220; holes++;
    } else if (g>high){ // pinholes brillantes
      id.data[p]=255; id.data[p+1]=255; id.data[p+2]=255; id.data[p+3]=200; holes++;
    } else {
      id.data[p+3]=0;
    }
  }
  ctx.putImageData(id,0,0);

  // leyenda
  ctx.fillStyle="rgba(0,0,0,0.45)"; ctx.fillRect(8,8,240,40);
  ctx.fillStyle="rgba(255,200,0,1)"; ctx.fillRect(12,14,14,14);
  ctx.fillStyle="#fff"; ctx.font="12px ui-sans-serif"; ctx.fillText("huecos oscuros", 32, 25);
  ctx.fillStyle="#fff"; ctx.fillRect(12,30,14,14);
  ctx.fillStyle="#fff"; ctx.fillText("pinholes brillantes", 32, 41);

  const pct = c? holes/c : 0;
  return { can, holesPct: pct };
}

/** =================== NUEVO — Manchas/Spots & Fingering =================== */
// Segmentación de manchas (componentes conexos) y métricas de distribución.
function analyzeSpots(gray, width, height, cx, cy, r){
  // Umbral por cuantil + sigma
  let vals=[]; vals.length=0;
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){
    const dx=x-cx, dy=y-cy; if (dx*dx+dy*dy<=r*r) vals.push(gray[y*width+x]);
  }
  vals.sort((a,b)=>a-b);
  const q20 = vals[Math.floor(vals.length*0.2)] || 0; // 20% más oscuro
  const mean = vals.reduce((a,b)=>a+b,0)/Math.max(vals.length,1);
  const std = Math.sqrt(Math.max(vals.reduce((a,b)=>a+(b-mean)*(b-mean),0)/Math.max(vals.length,1),1e-8));
  const thr = Math.min(q20, mean - 0.7*std);

  const bin = new Uint8Array(width*height);
  for (let y=0;y<height;y++) for (let x=0;x<width;x++){
    const dx=x-cx, dy=y-cy; const inside=(dx*dx+dy*dy<=r*r);
    const i=y*width+x; bin[i] = inside && (gray[i] < thr) ? 1 : 0;
  }

  // BFS labeling
  const label = new Int32Array(width*height); label.fill(-1);
  const comps = [];
  const qx=new Int32Array(width*height), qy=new Int32Array(width*height);
  let curLabel=0;
  const push=(arr,idx,val)=>arr[idx]=val;

  for (let y=1;y<height-1;y++){
    for (let x=1;x<width-1;x++){
      const i=y*width+x;
      if (!bin[i] || label[i]!==-1) continue;
      // flood
      let head=0, tail=0;
      qx[tail]=x; qy[tail]=y; tail++;
      label[i]=curLabel;
      let area=0, pxSum=0, pySum=0, perim=0, minx=x, maxx=x, miny=y, maxy=y, rsum=0, rsumsq=0;
      while(head<tail){
        const xx=qx[head], yy=qy[head]; head++;
        area++; pxSum+=xx; pySum+=yy;
        const dx0=xx-cx, dy0=yy-cy; const rad=Math.hypot(dx0,dy0)/r; rsum+=rad; rsumsq+=rad*rad;
        if (xx<minx) minx=xx; if (xx>maxx) maxx=xx; if (yy<miny) miny=yy; if (yy>maxy) maxy=yy;
        // 4 vecinos
        const neigh=[[1,0],[-1,0],[0,1],[0,-1]];
        for (let k=0;k<4;k++){
          const nx=xx+neigh[k][0], ny=yy+neigh[k][1];
          const j=ny*width+nx;
          if (!bin[j]){ perim++; continue; }
          if (label[j]===-1){ label[j]=curLabel; qx[tail]=nx; qy[tail]=ny; tail++; }
        }
      }
      const cxC=pxSum/area, cyC=pySum/area;
      const ecc = (maxx-minx+1)/(maxy-miny+1); // relación de aspecto
      const compactness = perim>0 ? (perim*perim)/(4*Math.PI*area) : 1; // >1 = alargado/irregular
      const rMean = rsum/area, rStd = Math.sqrt(Math.max(rsumsq/area - rMean*rMean, 0));

      comps.push({ area, perim, cx:cxC, cy:cyC, bbox:[minx,miny,maxx,maxy], ecc, compactness, rMean, rStd });
      curLabel++;
    }
  }

  const totalDisk = Math.PI*r*r;
  const pixArea = 1; // aproximación
  const spotsArea = comps.reduce((a,c)=>a+c.area*pixArea,0);
  const spotsCount = comps.length;
  const spotsAreaPct = totalDisk>0 ? spotsArea/totalDisk : 0;

  // Cluster index: varianza de conteo por sectores (más var = más segregación angular)
  const sectors = 24;
  const counts = new Float64Array(sectors);
  for (const c of comps){
    const a=Math.atan2(c.cy-cy, c.cx-cx); let k=Math.floor(((a+Math.PI)/(2*Math.PI))*sectors); if (k<0) k=0; if (k>=sectors) k=sectors-1;
    counts[k]+=c.area;
  }
  const meanC = counts.reduce((a,b)=>a+b,0)/sectors;
  const varC = counts.reduce((a,b)=>a+(b-meanC)*(b-meanC),0)/sectors;
  const clusterIndex = meanC>0 ? varC/(meanC*meanC+1e-6) : 0;

  // Fingering: proporción de componentes alargados/irregulares en corona media-externa
  let fingers=0, eligible=0;
  for (const c of comps){
    if (c.rMean>0.45){ // preferimos zona media-externa
      eligible++;
      if (c.compactness>1.6 || c.ecc>2.0) fingers++;
    }
  }
  const fingeringIndex = eligible>0 ? fingers/eligible : 0;

  // Overlay
  const can = document.createElement("canvas");
  can.width=width; can.height=height;
  const ctx = can.getContext("2d");
  ctx.strokeStyle="rgba(255,180,0,0.9)"; ctx.lineWidth=2;
  comps.forEach(c=>{
    ctx.beginPath();
    ctx.rect(c.bbox[0]-0.5, c.bbox[1]-0.5, (c.bbox[2]-c.bbox[0])+1, (c.bbox[3]-c.bbox[1])+1);
    ctx.stroke();
    ctx.fillStyle="rgba(255,180,0,0.25)";
    ctx.beginPath(); ctx.arc(c.cx, c.cy, 2.5, 0, Math.PI*2); ctx.fill();
  });
  // leyenda
  ctx.fillStyle="rgba(0,0,0,0.45)"; ctx.fillRect(8,8,240,40);
  ctx.fillStyle="#fff"; ctx.font="12px ui-sans-serif"; ctx.fillText("Manchas/Spots (contornos)", 12, 24);

  return { can, spotsCount, spotsAreaPct, clusterIndex, fingeringIndex };
}

/** =================== Heurísticas extra =================== */
function centerJetIndex(profile){
  // centro más oscuro que zona media => índice > umbral
  const c = avgBand(profile, 0.0, 0.18);
  const mid = avgBand(profile, 0.45, 0.65);
  // gris bajo = más oscuro; si c << mid, hay “mancha central”
  return clamp01((mid - c) / (mid + 1e-6));
}
function edgeDarkIndex(profile){
  const edge = avgBand(profile, 0.90, 0.99);
  const mid = avgBand(profile, 0.45, 0.65);
  return clamp01((mid - edge) / (mid + 1e-6)); // alto = borde más oscuro
}
function avgBand(profile, a, b){
  const n=profile.length; const i0=Math.max(0, Math.floor(a*n)); const i1=Math.min(n-1, Math.floor(b*n));
  let s=0,c=0; for (let i=i0;i<=i1;i++){ s+=profile[i]; c++; } return c? s/c : 0;
}

/** =================== Análisis (puck) =================== */
async function analyzeFromCircle(srcCanvas, cx, cy, r){
  const cropped = cropCircleFromCanvas(srcCanvas, cx, cy, r, 1.18);
  const cid = cropped.getContext("2d").getImageData(0,0,cropped.width, cropped.height);
  const cg = grayscale(cid);
  const centerX = cropped.width/2, centerY = cropped.height/2;

  const stats = polarStats(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const lv = localVariance(cg.gray, cg.width, cg.height, centerX, centerY, r, 64);
  const edges = drawEdgesOverlay(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const sectors = drawSectorsOverlay(cg.gray, cg.width, cg.height, centerX, centerY, r, 24);
  const heatmap = drawHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, lv.bw, lv.bh, cg.width, cg.height);
  const guides = (()=>{ const can=document.createElement("canvas"); can.width=cropped.width; can.height=cropped.height; const ctx=can.getContext("2d"); ctx.strokeStyle="rgba(255,255,255,0.45)"; ctx.lineWidth=1.2; const rings=[0.3,0.6,0.9]; const Cx=cropped.width/2, Cy=cropped.height/2; rings.forEach(t=>{ ctx.beginPath(); ctx.arc(Cx,Cy,r*t,0,Math.PI*2); ctx.stroke(); }); return can; })();

  const ring = detectDarkRing(stats.profile, [0.78,0.96]);
  const ringDepth = (stats.mid - ring.minV);
  const ringOverlay = drawRingOverlay(cropped.width, r, ring.pos, 0.06);

  const profileChart = drawProfileChart(Array.from(stats.profile));
  const histChart = drawHistogram(cg.gray, cg.width, cg.height, centerX, centerY, r);

  // NUEVO: huecos/pinholes + manchas/fingering
  const holes = drawHolesOverlay(cg.gray, cg.width, cg.height, centerX, centerY, r);
  const spots = analyzeSpots(cg.gray, cg.width, cg.height, centerX, centerY, r);

  // Índices de patrón
  const idxCenter = centerJetIndex(stats.profile);
  const idxEdgeDark = edgeDarkIndex(stats.profile);

  // Heurística headspace (refinada)
  const hsRisk = clamp01(
    0.55 * clamp01(ringDepth/0.06) * clamp01((ring.pos-0.88)/0.08) +
    0.25 * clamp01(edges.density*3) +
    0.20 * clamp01(spots.clusterIndex*0.7 + idxEdgeDark)
  );

  const [croppedURL, hmURL, edgeURL, sectorURL, guideURL, ringURL, profileURL, histURL, holesURL, spotsURL] = await Promise.all([
    canvasToURL(cropped), canvasToURL(heatmap), canvasToURL(edges.can),
    canvasToURL(sectors.can), canvasToURL(guides), canvasToURL(ringOverlay),
    canvasToURL(profileChart), canvasToURL(histChart), canvasToURL(holes.can), canvasToURL(spots.can)
  ]);

  return {
    dim: { w: cropped.width, h: cropped.height, r: Math.round(r) },
    urls: { croppedURL, hmURL, edgeURL, sectorURL, guideURL, ringURL, profileURL, histURL, holesURL, spotsURL },
    metrics: {
      eci: stats.eci,
      sectorStd: sectors.std,
      edgeDensity: edges.density,
      ringDepth, ringPos: ring.pos,
      headspace: hsRisk,
      spotsCount: spots.spotsCount,
      spotsAreaPct: spots.spotsAreaPct,
      clusterIndex: spots.clusterIndex,
      fingeringIndex: spots.fingeringIndex,
      idxCenter, idxEdgeDark
    },
    arrays: { sectorMeans: sectors.means, radialProfile: stats.profile },
    debug: { cx, cy, r }
  };
}

/** =================== Análisis huella en papel (opcional) =================== */
async function analyzePrintFromCircle(srcCanvas, cx, cy, r){
  const cropped = cropCircleFromCanvas(srcCanvas, cx, cy, r, 1.18);
  const id = cropped.getContext("2d").getImageData(0,0,cropped.width,cropped.height);
  const g = grayscale(id);
  // invertir: más claro = más flujo
  const inv = new Float32Array(g.gray.length);
  for (let i=0;i<inv.length;i++) inv[i] = 1.0 - g.gray[i];

  const cx0=cropped.width/2, cy0=cropped.height/2;

  const ps = polarStats(inv, g.width, g.height, cx0, cy0, r);
  const lv = localVariance(inv, g.width, g.height, cx0, cy0, r, 64);
  const heatmap = drawHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, lv.bw, lv.bh, g.width, g.height);
  const sectors = drawSectorsOverlay(inv, g.width, g.height, cx0, cy0, r, 24);
  const guides = (()=>{ const can=document.createElement("canvas"); can.width=cropped.width; can.height=cropped.height; const ctx=can.getContext("2d"); ctx.strokeStyle="rgba(255,255,255,0.45)"; ctx.lineWidth=1.2; const rings=[0.3,0.6,0.9]; rings.forEach(t=>{ ctx.beginPath(); ctx.arc(cx0,cy0,r*t,0,Math.PI*2); ctx.stroke(); }); return can; })();

  // canales por cuantil (top-25% de sectores)
  const means = sectors.means;
  const sorted = [...means].sort((a,b)=>b-a);
  const cutoff = sorted[Math.floor(means.length*0.25)];
  const chMask = means.map(v=> v >= cutoff ? 1 : 0);
  const channelPct = chMask.reduce((a,b)=>a+b,0)/chMask.length;

  // pico de flujo en el borde (anillo de flujo)
  const ringOverlay = drawRingOverlay(cropped.width, r, detectDarkRing(ps.profile.map(v=>1-v), [0.78,0.98]).pos, 0.06); // pos de máximo en inv

  const profileChart = drawProfileChart(Array.from(ps.profile));
  const histChart = drawHistogram(inv, g.width, g.height, cx0, cy0, r);

  const [croppedURL, hmURL, sectorURL, guideURL, ringURL, profileURL, histURL] = await Promise.all([
    canvasToURL(cropped), canvasToURL(heatmap), canvasToURL(sectors.can),
    canvasToURL(guides), canvasToURL(ringOverlay), canvasToURL(profileChart), canvasToURL(histChart)
  ]);

  return {
    urls:{ croppedURL, hmURL, sectorURL, guideURL, ringURL, profileURL, histURL },
    metrics:{
      channelPct,
      sectorStd: sectors.std,
      radialEdgeBoost: ps.ringAvg(0.88,0.98) - ps.ringAvg(0.55,0.7)
    },
    arrays:{ sectorMeans: sectors.means, radialProfile: ps.profile }
  };
}

/** =================== Correlaciones =================== */
function corrPearson(a,b){
  const n=Math.min(a.length,b.length); let sx=0,sy=0,sxx=0,syy=0,sxy=0;
  for (let i=0;i<n;i++){ const x=a[i], y=b[i]; sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; }
  const cov=sxy/n - (sx/n)*(sy/n); const vx=sxx/n - (sx/n)*(sx/n); const vy=syy/n - (sy/n)*(sy/n);
  const denom=Math.sqrt(Math.max(vx,1e-12)*Math.max(vy,1e-12)); return denom>0? cov/denom : 0;
}

/** =================== UI helpers =================== */
function Pill({children}){ return <span className="pill">{children}</span>; }

/** =================== App =================== */
export default function App(){
  const [sets, setSets] = useState([]);
  const [busy, setBusy] = useState(false);
  const [overlay, setOverlay] = useState({ heatmap:false, guides:true, edges:true, holes:false, spots:true, sectors:false, ring:true });
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

      // Optional prints (huella en papel)
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

      // Pair correlations
      const a = Array.from(top.arrays.sectorMeans);
      const b = Array.from(bottom.arrays.sectorMeans);
      const abCorr = (corrPearson(a,b)+1)/2;

      // Construir recomendaciones específicas
      const rec = buildRecommendations({top, bottom, abCorr, topPrint, bottomPrint});

      const setItem = {
        top, bottom, topPrint, bottomPrint, pair: { sectorCorr: abCorr },
        summary: {
          corr: Math.round(abCorr*100),
          edgeDark: round(bottom.metrics.idxEdgeDark),
          centerJet: round(bottom.metrics.idxCenter),
          spots: bottom.metrics.spotsCount
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

  function buildRecommendations({top, bottom, abCorr, topPrint, bottomPrint}){
    const R = [];
    const A = [];

    // --- Patrones clave cara inferior (salida) ---
    const b = bottom.metrics;
    const t = top.metrics;

    // 1) Borde oscurecido → headspace bajo / densidad periférica
    if (b.idxEdgeDark > 0.35 && b.ringDepth > 0.02){
      R.push("Borde más oscuro con anillo pronunciado en la cara inferior.");
      A.push("Aumenta **headspace** (menos dosis o canasta más alta) y evita compactar el borde.");
      A.push("Haz **WDT fino** hasta el borde pero sin barrer al final (no empujes finos hacia el perímetro).");
      A.push("Usa **preinfusión** suave (2–6 s) y sube presión gradualmente.");
      A.push("Considera **puck screen** superior o papel abajo si el anillo es persistente.");
    }

    // 2) Mancha central → ducha que moja más el centro (center jet)
    if (b.idxCenter > 0.30){
      R.push("Oscurecimiento/erosión en el centro (posible chorro central de la ducha).");
      A.push("Limpia/inspecciona la **ducha** y su distribución; revisa que no haya obstrucciones laterales.");
      A.push("Gira/recoloca la **pantalla/dispersion screen** y verifica homogeneidad del caudal.");
      A.push("Aumenta **preinfusión** y reduce el pico de **caudal** inicial.");
      A.push("El **puck screen** puede ayudar a repartir el agua y mitigar el chorro central.");
    }

    // 3) Manchas segregadas + fingering
    if (b.spotsCount >= 6 && (b.fingeringIndex > 0.25 || b.clusterIndex > 0.5)){
      R.push("Manchas segregadas y patrón tipo ‘fingering’ (ramas/dedos) en la cama.");
      A.push("Mejora la **humectación** inicial: preinfusión más larga y subida de presión más lenta.");
      A.push("Ajusta **molienda** (ligeramente más fina) y refuerza **WDT** profundo pero suave.");
      A.push("Reduce **estática** (RDT) para bajar grumos y finos sueltos.");
      A.push("Si persiste, prueba **papel** abajo para estabilizar la superficie de salida.");
    }

    // 4) Grietas / roturas
    if (b.edgeDensity > 0.012){
      R.push("Indicios de grietas/roturas internas.");
      A.push("Haz **tamp recto**, sin torsión ni golpecitos laterales; presión consistente.");
      A.push("Evita golpes del portafiltro al acoplar; usa agujas **0.30–0.40 mm** y menos agresivas en WDT.");
    }

    // 5) Huecos / pinholes
    if ((b.spotsAreaPct > 0.02 && b.spotsCount>0) || (t && t.spotsAreaPct > 0.02)){
      R.push("Huecos/pinholes visibles en la cama.");
      A.push("Mejora **distribución** para rellenar microvacíos; reduce **estática** (RDT).");
      A.push("Tamiza o purga para controlar finos; comprueba que no haya grumos en la tolva.");
    }

    // 6) Desacople entre caras
    if (abCorr < 0.55){
      R.push("Baja correlación entre cara superior e inferior (lo que entra ≠ lo que sale).");
      A.push("Refuerza nivelación **antes** del tamp y busca homogeneidad angular (Rao spin suave).");
      A.push("Modera el **pico de presión/caudal** al inicio para evitar caminos preferentes.");
    }

    // 7) Huella papel (si disponible)
    if (bottomPrint || topPrint){
      const bp = bottomPrint?.metrics?.channelPct ?? 0;
      if (bp > 0.3){
        R.push("La huella inferior muestra concentración de canales (>30% sectores rápidos).");
        A.push("Revisa **nivelación en perímetro** y considera reducir presión pico; papel abajo puede ayudar.");
      }
    }

    if (R.length===0) R.push("Distribución razonablemente uniforme; ajustes finos posibles.");
    if (A.length===0) A.push("Mantén WDT homogéneo, verifica headspace (2–6 mm) y evita sobre-compactar.");

    return { lines:R, bullets:A };
  }

  return (
    <div className="container">
      <div className="row" style={{justifyContent:"space-between", alignItems:"flex-start", gap:16}}>
        <div>
          <h1 style={{margin:"0 0 6px 0"}}>Puck Diagnosis</h1>
          <div className="muted">
            Sube <b>dos fotos por puck</b>: <span className="tag">Superior = lado ducha</span> <span className="tag">Inferior = lado canasta</span>.
            <span className="tag">Opcional: huella en papel (superior/inferior)</span>
          </div>
        </div>
        <div className="row">
          {["heatmap","guides","edges","holes","spots","sectors","ring"].map(key=>(
            <label key={key} className="row small" style={{gap:8}}>
              <input type="checkbox" checked={overlay[key]} onChange={e=>setOverlay(o=>({...o, [key]:e.target.checked}))}/>
              <span>{key[0].toUpperCase()+key.slice(1)}</span>
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
            <div className="small">Huella papel — Superior (opcional)</div>
            <input ref={topPrintRef} type="file" accept="image/*" />
            <div style={{marginTop:6}}><button className="btn" onClick={()=>openAdjust("topPrint")}>Ajustar recorte</button></div>
          </div>
          <div>
            <div className="small">Huella papel — Inferior (opcional)</div>
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
                  <Pill>Corr top/bot: <span className="mono">{st.summary.corr}%</span></Pill>
                  <Pill>Edge dark: <span className="mono">{st.summary.edgeDark}</span></Pill>
                  <Pill>Center jet: <span className="mono">{st.summary.centerJet}</span></Pill>
                  <Pill>Spots: <span className="mono">{st.summary.spots}</span></Pill>
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
                          {overlay.holes && <img className="overlay" src={obj.it.urls.holesURL} alt="holes" />}
                          {overlay.spots && <img className="overlay" src={obj.it.urls.spotsURL} alt="spots" />}
                          {overlay.sectors && <img className="overlay" src={obj.it.urls.sectorURL} alt="sectors" />}
                          {overlay.ring && <img className="overlay" src={obj.it.urls.ringURL} alt="ring" />}
                        </div>
                        <div style={{height:8}} />
                        <div className="metrics" style={{gridTemplateColumns:"repeat(3,1fr)"}}>
                          <div>σ sectores: <span className="mono">{obj.it.metrics.sectorStd.toFixed(3)}</span></div>
                          <div>Grietas: <span className="mono">{(obj.it.metrics.edgeDensity*100).toFixed(2)}%</span></div>
                          <div>Anillo (prof.): <span className="mono">{obj.it.metrics.ringDepth.toFixed(3)}</span></div>
                          <div>Headspace (riesgo): <span className="mono">{(obj.it.metrics.headspace*100).toFixed(0)}%</span></div>
                          <div>Spots#: <span className="mono">{obj.it.metrics.spotsCount}</span></div>
                          <div>Spots área%: <span className="mono">{(obj.it.metrics.spotsAreaPct*100).toFixed(1)}%</span></div>
                          <div>Cluster idx: <span className="mono">{obj.it.metrics.clusterIndex.toFixed(2)}</span></div>
                          <div>Fingering idx: <span className="mono">{obj.it.metrics.fingeringIndex.toFixed(2)}</span></div>
                          <div>CenterJet idx: <span className="mono">{obj.it.metrics.idxCenter.toFixed(2)}</span></div>
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
                                {overlay.ring && <img className="overlay" src={obj.it.urls.ringURL} alt="ring" />}
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
                    <ul style={{marginTop:0}}>
                      {st.rec.lines.map((l,i)=><li key={i}>{l}</li>)}
                    </ul>
                    <div className="small muted" style={{margin:"6px 0"}}>Acciones sugeridas:</div>
                    <ul>
                      {st.rec.bullets.map((b,i)=><li key={i}>{b}</li>)}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
