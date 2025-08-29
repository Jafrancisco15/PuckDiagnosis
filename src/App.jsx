import React, { useCallback, useMemo, useRef, useState } from "react";

/**
 * Puck Diagnosis (client‑side, no dependencies)
 * - Sube varias fotos del puck (idealmente top‑down).
 * - Recorta el puck automáticamente, lo centra, remueve el fondo (transparente).
 * - Analiza patrones simples de canalización: contraste borde/centro, textura, extremos.
 * - Opcional: genera un heatmap de variación local.
 *
 * Heurística (ligera, sin OpenCV):
 * 1) Umbral adaptativo aproximado para aislar pixeles más oscuros (puck).
 * 2) Centro = centroide de esos pixeles; radio ≈ p80 de la distancia al centro.
 * 3) Máscara circular -> export PNG transparente centrada.
 * 4) Métricas: ECI (edge-center intensity Δ), Heterogeneity (var local), extremesRatio.
 * 5) Clasificación: OK | Leve | Severa.
 *
 * NOTA: Para mejores resultados, use fotos bien iluminadas, desde arriba, con contraste entre puck y fondo.
 */

const MAX_DIM = 1024; // reduce para performance en móviles

function clamp01(x) { return Math.min(1, Math.max(0, x)); }
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

// robust threshold ~ mean - k*std (puck suele ser más oscuro que fondo)
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
  // draw source such that (cx,cy) -> (size/2,size/2)
  const dx = Math.round(size/2 - cx);
  const dy = Math.round(size/2 - cy);
  ctx.drawImage(srcCanvas, dx, dy);
  ctx.restore();

  // make outside transparent explicitly
  const id = ctx.getImageData(0,0,size,size);
  const data = id.data;
  for (let y=0;y<size;y++){
    for (let x=0;x<size;x++){
      const dx2 = x-size/2, dy2=y-size/2;
      if (dx2*dx2 + dy2*dy2 > r*r){
        const p = (y*size + x)*4 + 3;
        data[p] = 0; // alpha
      }
    }
  }
  ctx.putImageData(id,0,0);
  return out;
}

function polarStats(gray, width, height, cx, cy, r) {
  // average intensity by ring (100 bins)
  const bins = 80;
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
  const eci = edge - mid; // Δ edge-center (gris 0-1)
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
  return { varmap, wBlocks, hBlocks, vmax };
}

function extremesRatio(gray, width, height, cx, cy, r) {
  // pixeles muy por debajo/encima del mean local global
  let s=0, c=0;
  for (let y=0;y<height;y++){
    for (let x=0;x<width;x++){
      const dx=x-cx, dy=y-cy;
      if (dx*dx+dy*dy<=r*r){ s += gray[y*width+x]; c++; }
    }
  }
  const mean = s/Math.max(c,1);
  // std aproximado dentro del disco
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

function makeHeatmapCanvas(varmap, wBlocks, hBlocks, vmax, block=16) {
  const W = wBlocks*block, H = hBlocks*block;
  const can = document.createElement("canvas");
  can.width = W; can.height = H;
  const ctx = can.getContext("2d");
  const id = ctx.createImageData(W,H);
  function colormap(t) {
    // simple blue->green->yellow->red (0..1)
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

function classify(eci, hetero, extremes) {
  // Umbrales empíricos (ajustables con tus datos)
  const eciAbs = Math.abs(eci);
  let level = "OK";
  let notes = [];
  if (eciAbs > 0.08) notes.push("Contraste borde‑centro notable (posible canalización en bordes).");
  if (hetero > 0.055) notes.push("Textura heterogénea (flujos irregulares).");
  if (extremes > 0.12) notes.push("Muchos extremos tonales (huecos/fisuras o zonas sobre‑extraídas).");

  const score = (eciAbs/0.08)*0.4 + (hetero/0.055)*0.35 + (extremes/0.12)*0.25;
  if (score > 2.2) level = "Severa";
  else if (score > 1.0) level = "Leve";

  const recs = [];
  if (eciAbs > 0.08) recs.push("Mejorar distribución en periferia (WDT más fino, chequeo de dosis/basket).");
  if (hetero > 0.055) recs.push("Revisar consistencia de molienda y compactado (tamper recto, fuerza estable).");
  if (extremes > 0.12) recs.push("Reducir channelling por jetting: ajustar calibración del molino, evaluar preinfusión/ratio.");
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
  const [overlayHeatmap, setOverlayHeatmap] = useState(false);
  const [busy, setBusy] = useState(false);

  const onFiles = useCallback(async (files)=>{
    setBusy(true);
    const arr = [];
    for (const file of files){
      try {
        const img = await fileToImageBitmap(file);
        const { canvas, cx, w, h } = (()=>{
          const d = drawToCanvas(img, MAX_DIM);
          return { canvas: d.canvas, w: d.w, h: d.h };
        })();
        const id = getImageData(canvas);
        const g = grayscale(id);
        const m = darkMask(g, 0.35);
        const c = centroid(m.mask, m.width, m.height);
        const r = robustRadius(m.mask, m.width, m.height, c.cx, c.cy);
        const cropped = cropCircle(img, canvas, c.cx, c.cy, r, 1.18);

        // recompute gray within the cropped frame for analysis alignment
        const cid = cropped.getContext("2d").getImageData(0,0,cropped.width, cropped.height);
        const cg = grayscale(cid);
        const stats = polarStats(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r);
        const lv = localVariance(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r, 14);
        const ext = extremesRatio(cg.gray, cg.width, cg.height, cropped.width/2, cropped.height/2, r);
        const cls = classify(stats.eci, lv.vmax, ext);

        // heatmap canvas sized to cropped
        const hm = makeHeatmapCanvas(lv.varmap, lv.wBlocks, lv.hBlocks, lv.vmax, 14);
        // draw overlay cropped mask shape on top of heatmap (circle alpha outside)
        const hmCtx = hm.getContext("2d");
        hmCtx.globalCompositeOperation = "destination-in";
        hmCtx.beginPath();
        hmCtx.arc(cropped.width/2, cropped.height/2, r, 0, Math.PI*2);
        hmCtx.fill();

        const blob = await new Promise(res=>cropped.toBlob(res, "image/png"));
        const dataURL = URL.createObjectURL(blob);

        const hmBlob = await new Promise(res=>hm.toBlob(res, "image/png"));
        const hmURL = URL.createObjectURL(hmBlob);

        arr.push({
          name: file.name,
          original: img,
          croppedURL: dataURL,
          heatmapURL: hmURL,
          size: { w: cropped.width, h: cropped.height, r: Math.round(r) },
          metrics: {
            eci: stats.eci,
            extremes: ext,
          },
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
          <div className="muted">Sube fotos del puck (vista superior). La app recorta, centra y analiza canalización.</div>
        </div>
        <div className="row">
          <label className="row" style={{gap:8}}>
            <input type="checkbox" checked={overlayHeatmap} onChange={e=>setOverlayHeatmap(e.target.checked)} />
            <span className="small">Mostrar heatmap de variación</span>
          </label>
          <button className="btn" disabled={busy} onClick={()=>document.getElementById("file-input").click()}>
            {busy? "Procesando..." : "Seleccionar imágenes"}
          </button>
          <input id="file-input" type="file" accept="image/*" multiple style={{display:"none"}} onChange={handleBrowse} />
        </div>
      </div>

      <div style={{height:16}} />

      <div className="card">
        <div className="drop"
             onDrop={handleDrop}
             onDragOver={(e)=>e.preventDefault()}>
          <div style={{fontWeight:700, fontSize:18, marginBottom:6}}>Arrastra y suelta imágenes aquí</div>
          <div className="muted">o haz clic en <span className="kbd">Seleccionar imágenes</span>. Formatos: PNG/JPG.</div>
          <div style={{height:8}} />
          <div className="row">
            <span className="tag">Recorte automático</span>
            <span className="tag">Centrado</span>
            <span className="tag">Fondo transparente</span>
            <span className="tag">Análisis de canalización</span>
            <span className="tag">Heatmap opcional</span>
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
                  <span className="muted">r={it.size.r}px</span>
                  <span className="tag">{it.level==="OK" ? "OK" : it.level==="Leve" ? "Leve" : "Severa"}</span>
                </div>
              )}
            </div>

            {it.error ? (
              <div className="err small" style={{marginTop:8}}>Error: {it.error}</div>
            ) : (
              <>
                <div className="thumb" style={{marginTop:12}}>
                  <img src={overlayHeatmap ? it.heatmapURL : it.croppedURL} alt="puck" />
                </div>

                <div style={{height:10}} />

                <div className="metrics">
                  <div>Δ Borde‑Centro (ECI): <span className="mono">{it.metrics.eci.toFixed(3)}</span></div>
                  <div>Extremos tonales: <span className="mono">{(it.metrics.extremes*100).toFixed(1)}%</span></div>
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
                    const resp = await fetch(it.croppedURL);
                    const blob = await resp.blob();
                    downloadBlob(`puck-${idx+1}.png`, blob);
                  }}>Descargar recorte</button>
                  <button className="btn" onClick={async ()=>{
                    const resp = await fetch(it.heatmapURL);
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
        * Heurística ligera sin OpenCV. Ajusta los umbrales en <span className="kbd">App.jsx</span> si lo deseas.
        Para producción, puedes comparar con un pipeline OpenCV (Hough + CLAHE + connected components) según tus datasets.
      </div>
    </div>
  );
}
