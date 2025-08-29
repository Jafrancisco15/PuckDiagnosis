import React, { useCallback, useState } from "react";

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
function localVariance(gray, width, height, cx, cy, r, blocks=64) {
  // create block grid that matches cropped size exactly
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
  // stats
  let s=0, s2=0, c=0;
  for (let i=0;i<mag.length;i++){ const v=mag[i]; if (v>0){ s+=v; s2+=v*v; c++; } }
  const mean = c? s/c : 0;
  const std = c? Math.sqrt(Math.max(s2/c - mean*mean, 1e-8)) : 1;
  return { mag, mean, std };
}

function drawHeatmapCanvas(varmap, wBlocks, hBlocks, vmax, bw, bh, width, height) {
  // Render directly at cropped size to ensure 1:1 with base
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
function drawSectorsOverlay(gray, width, height, cx, cy, r, sectors=24){
  // Compute sector deviations and draw legend
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
    let color = dev>0 ? `rgba(59,130,246,${0.25 + 0.5*t})` : `rgba(239,68,68,${0.25 + 0.5*t})`; // blue=claro, red=oscuro
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
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("rojo: más oscuro (densidad/atascos)", pad+28, pad+34);
  ctx.fillStyle="rgba(59,130,246,0.9)"; ctx.fillRect(pad+8, pad+38, 16, 12);
  ctx.fillStyle="rgba(255,255,255,0.85)"; ctx.fillText("azul: más claro (flujos rápidos)", pad+28, pad+48);

  return { can, means, globalMean, std };
}

function drawProfileChart(profile){
  const W=420, H=130, pad=8;
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
  const W=420, H=130, pad=8;
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

function getRecommendations({eci, extremes, sectorStd, edgeDensity, offCenter, r}){
  const lines = [];
  const bullets = [];

  lines.push(`• Δ borde-centro (ECI): ${round(eci)} — ${Math.abs(eci)>0.08 ? "alto" : "moderado/bajo"}.`);
  lines.push(`• Extremos tonales: ${round(extremes*100)}% — ${extremes>0.12 ? "muchos extremos" : "controlado"}.`);
  lines.push(`• Desv. sectores: ${round(sectorStd)} — ${sectorStd>0.03 ? "asimetrías angulares" : "uniforme"}.`);
  lines.push(`• Densidad de grietas: ${round(edgeDensity*100)}% de píxeles — ${edgeDensity>0.06 ? "elevada" : "normal"}.`);
  if (offCenter>r*0.06) lines.push(`• Desalineación: centro del puck desplazado ${Math.round(offCenter)} px — revisar distribución/tamper.`);

  // Priorizar en función de qué métrica domina
  if (Math.abs(eci) > 0.10){
    bullets.push("Refuerza la distribución en la periferia: WDT fino 10–15 s, sacudidas suaves para asentar, verifica nivelado antes del tamper.");
    bullets.push("Revisa la relación dosis/cesta: evita headspace excesivo o sobrellenado que empuja flujo hacia las paredes.");
    bullets.push("Ensayo de preinfusión 3–6 s a baja presión para empapar uniformemente antes del ramp-up.");
  }
  if (extremes > 0.15){
    bullets.push("Homogeneiza la molienda: pequeños cambios en el dial (±0.5–1 clic) y verifica presencia de grumos; aplica RDT para reducir estática.");
    bullets.push("Compactado consistente: tamper recto, presión estable; considera un nivelador previo si hay desniveles visibles.");
  }
  if (sectorStd > 0.035){
    bullets.push("Gira el portafiltro 90° entre fotos para verificar si la asimetría rota (máquina) o permanece (preparación).");
    bullets.push("Evalúa distribución radial: inserta WDT más profundo en sectores 'rojos' (oscuros) identificados.");
  }
  if (edgeDensity > 0.07){
    bullets.push("Atiende microfisuras: suaviza el grooming superficial, evita golpeteos fuertes del portafiltro tras el tamper.");
    bullets.push("Si usas paper filter, prueba colocarlo arriba/abajo para amortiguar jetting inicial.");
  }
  if (offCenter>r*0.06){
    bullets.push("Verifica que el tamper entre ajustado a la cesta y apoya perpendicular al borde.");
    bullets.push("Revisa nivel de la mesa/máquina; una ligera inclinación puede sesgar el flujo.");
  }
  if (bullets.length===0){
    bullets.push("Puck homogéneo: mantén parámetros. Explora ajustes finos de ratio/tiempo para optimizar sabor sin comprometer uniformidad.");
  }
  return { lines, bullets };
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
        const cropped = cropCircle(canvas, c.cx, c.cy, r, 1.18);

        // analysis in cropped coordinates
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

        // centroid offset (using original mask center projected into cropped coords)
        const offCenter = Math.hypot((c.cx - (cropped.width/2)), (c.cy - (cropped.height/2))); // after crop centerizing, this should be small; but we keep metric

        // charts
        const profileChart = drawProfileChart(Array.from(stats.profile));
        const histChart = drawHistogram(cg.gray, cg.width, cg.height, centerX, centerY, r);

        // blobs/urls
        const toURL = async (can)=>{
          const b = await new Promise(res=>can.toBlob(res, "image/png"));
          return URL.createObjectURL(b);
        };
        const croppedURL = await toURL(cropped);
        const hmURL = await toURL(heatmap);
        const edgeURL = await toURL(edges.can);
        const sectorURL = await toURL(sectors.can);
        const guideURL = await toURL(guides);
        const profileURL = await toURL(profileChart);
        const histURL = await toURL(histChart);

        const rec = getRecommendations({
          eci: stats.eci,
          extremes: ext.ratio,
          sectorStd: sectors.std,
          edgeDensity: edges.density,
          offCenter,
          r
        });

        arr.push({
          name: file.name,
          dim: { w: cropped.width, h: cropped.height, r: Math.round(r) },
          urls: { croppedURL, hmURL, edgeURL, sectorURL, guideURL, profileURL, histURL },
          metrics: {
            eci: stats.eci,
            extremes: ext.ratio,
            sectorStd: sectors.std,
            edgeDensity: edges.density,
            offCenter
          },
          rec
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
          <div className="muted">Recorte y centrado automáticos; análisis con overlays 1:1 sobre el puck.</div>
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
            <span>Grietas</span>
          </label>
          <label className="row small" style={{gap:8}}>
            <input type="checkbox" checked={overlay.sectors} onChange={e=>setOverlay(o=>({...o, sectors:e.target.checked}))} />
            <span>Sectores</span>
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
            <span className="tag">Heatmap</span>
            <span className="tag">Grietas (Sobel)</span>
            <span className="tag">Guías + Sectores (con leyenda)</span>
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
                  <span className="muted">r={it.dim.r}px</span>
                </div>
              )}
            </div>

            {it.error ? (
              <div className="err small" style={{marginTop:8}}>Error: {it.error}</div>
            ) : (
              <>
                <div className="thumb canvas-stack" style={{marginTop:12}}>
                  {/* Base siempre el recorte para alinear overlays 1:1 */}
                  <img src={it.urls.croppedURL} alt="puck" />
                  {/* Overlays opcionales, misma dimensión que el recorte */}
                  {overlay.heatmap && <img className="overlay" src={it.urls.hmURL} alt="heatmap" />}
                  {overlay.guides && <img className="overlay" src={it.urls.guideURL} alt="guides" />}
                  {overlay.edges && <img className="overlay" src={it.urls.edgeURL} alt="edges" />}
                  {overlay.sectors && <img className="overlay" src={it.urls.sectorURL} alt="sectors" />}
                </div>

                <div style={{height:10}} />

                <div className="metrics">
                  <div>Δ Borde-Centro (ECI): <span className="mono">{it.metrics.eci.toFixed(3)}</span></div>
                  <div>Extremos tonales: <span className="mono">{(it.metrics.extremes*100).toFixed(1)}%</span></div>
                  <div>Desv. de sectores (σ): <span className="mono">{it.metrics.sectorStd.toFixed(3)}</span></div>
                  <div>Densidad de grietas: <span className="mono">{(it.metrics.edgeDensity*100).toFixed(2)}%</span></div>
                </div>

                <div className="charts">
                  <div className="chart"><img src={it.urls.profileURL} alt="perfil radial" /></div>
                  <div className="chart"><img src={it.urls.histURL} alt="histograma" /></div>
                </div>

                <div style={{height:10}} />

                <div className="small">
                  <div style={{fontWeight:700, marginBottom:4}}>Análisis</div>
                  <ul style={{marginTop:4, paddingLeft:18}}>
                    {it.rec.lines.map((n,i)=>(<li key={i}>{n}</li>))}
                  </ul>
                </div>

                <div style={{height:10}} />

                <div className="small">
                  <div style={{fontWeight:700, marginBottom:4}}>Recomendaciones priorizadas</div>
                  <ul style={{marginTop:0, paddingLeft:18}}>
                    {it.rec.bullets.map((r,i)=>(<li key={i}>{r}</li>))}
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
        * Overlays alineados 1:1 con el recorte. Leyenda incluida en el mapa de sectores.<br/>
        Ajusta umbrales en <span className="kbd">src/App.jsx</span> según tus fotos.
      </div>
    </div>
  );
}
