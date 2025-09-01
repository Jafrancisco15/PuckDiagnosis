import React, { useEffect, useRef, useState } from "react";

/* ================= Utilidades base ================= */
const MAX_DIM = 1200;
const clamp01 = x => Math.max(0, Math.min(1, x));
const toGray = (r,g,b)=> (0.2126*r + 0.7152*g + 0.0722*b)/255;

async function fileToImageBitmap(file){
  const url = URL.createObjectURL(file);
  const img = await createImageBitmap(await fetch(url).then(r=>r.blob()));
  URL.revokeObjectURL(url);
  return img;
}
function drawToCanvas(img, maxDim = MAX_DIM){
  const s = Math.min(1, maxDim/Math.max(img.width, img.height));
  const w = Math.round(img.width*s), h = Math.round(img.height*s);
  const c = document.createElement("canvas"); c.width=w; c.height=h;
  c.getContext("2d").drawImage(img,0,0,w,h);
  return c;
}
function getImageData(canvas){ return canvas.getContext("2d").getImageData(0,0,canvas.width,canvas.height); }
function grayscale(imgData){
  const { data, width:w, height:h } = imgData;
  const g = new Float32Array(w*h); let s=0,s2=0;
  for (let i=0,j=0;i<data.length;i+=4,j++){ const v=toGray(data[i],data[i+1],data[i+2]); g[j]=v; s+=v; s2+=v*v; }
  const n=w*h, mean=s/n, std=Math.sqrt(Math.max(s2/n - mean*mean,1e-8));
  return { gray:g, width:w, height:h, mean, std };
}
function canvasToURL(can){ return new Promise(res=>can.toBlob(b=>res(URL.createObjectURL(b)),"image/png")); }

/* ================= Filtros/Blur ================= */
function boxBlur1D(src,w,h,r,horiz){
  const out=new Float32Array(src.length); const R=Math.max(1,Math.floor(r));
  if(horiz){
    for(let y=0;y<h;y++){
      let S=0; for(let x=-R;x<=R;x++) S+=src[y*w+Math.min(w-1,Math.max(0,x))];
      for(let x=0;x<w;x++){
        out[y*w+x]=S/(2*R+1);
        const x0=x-R, x1=x+R+1;
        S += src[y*w+Math.min(w-1,x1)] - src[y*w+Math.max(0,x0)];
      }
    }
  }else{
    for(let x=0;x<w;x++){
      let S=0; for(let y=-R;y<=R;y++) S+=src[Math.min(h-1,Math.max(0,y))*w+x];
      for(let y=0;y<h;y++){
        out[y*w+x]=S/(2*R+1);
        const y0=y-R, y1=y+R+1;
        S += src[Math.min(h-1,y1)*w+x] - src[Math.max(0,y0)*w+x];
      }
    }
  }
  return out;
}
function blur(src,w,h,r,it=2){ let a=src; for(let i=0;i<it;i++){ a=boxBlur1D(a,w,h,r,true); a=boxBlur1D(a,w,h,r,false); } return a; }

/* ================= Normalización e iluminación ================= */
function normalizeIllum(gray,w,h){
  // “retinex” sencillo: divide por fondo (blur grande) y re-escala
  const bg = blur(gray,w,h,18,2);
  const out=new Float32Array(gray.length); let min=1e9, max=-1e9;
  for(let i=0;i<gray.length;i++){
    const v = gray[i]/(bg[i]+1e-6);
    const g = Math.pow(clamp01(v), 0.85); // gamma
    out[i]=g; if(g<min)min=g; if(g>max)max=g;
  }
  const k = 1/(max-min+1e-9);
  for(let i=0;i<out.length;i++) out[i]=(out[i]-min)*k;
  return out;
}

/* ================= Gradiente / Sobel ================= */
function sobel(gray,w,h){
  const kx=[-1,0,1,-2,0,2,-1,0,1], ky=[-1,-2,-1,0,0,0,1,2,1];
  const mag=new Float32Array(w*h);
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
    let gx=0,gy=0, t=0;
    for(let j=-1;j<=1;j++) for(let i=-1;i<=1;i++){
      const p=(y+j)*w+(x+i), val=gray[p];
      gx += val * kx[t]; gy += val * ky[t]; t++;
    }
    mag[y*w+x]=Math.hypot(gx,gy);
  }
  let m=0,s2=0,c=0; for(let i=0;i<mag.length;i++){ const v=mag[i]; if(v>0){ m+=v; s2+=v*v; c++; } }
  const mean=c?m/c:0, std=c?Math.sqrt(Math.max(s2/c-mean*mean,1e-8)):1;
  return { mag, mean, std };
}

/* ================= Morfología en grises (cuadrado) ================= */
function dilateGray(src,w,h,r){
  const out=new Float32Array(src.length); const R=Math.max(1,Math.floor(r));
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    let mx=-1e9;
    for(let j=-R;j<=R;j++) for(let i=-R;i<=R;i++){
      const xx=Math.min(w-1,Math.max(0,x+i)), yy=Math.min(h-1,Math.max(0,y+j));
      mx=Math.max(mx, src[yy*w+xx]);
    }
    out[y*w+x]=mx;
  }
  return out;
}
function erodeGray(src,w,h,r){
  const out=new Float32Array(src.length); const R=Math.max(1,Math.floor(r));
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    let mn=1e9;
    for(let j=-R;j<=R;j++) for(let i=-R;i<=R;i++){
      const xx=Math.min(w-1,Math.max(0,x+i)), yy=Math.min(h-1,Math.max(0,y+j));
      mn=Math.min(mn, src[yy*w+xx]);
    }
    out[y*w+x]=mn;
  }
  return out;
}
function closingGray(src,w,h,r){ return erodeGray(dilateGray(src,w,h,r),w,h,r); }

/* ================= Componentes / medidas de forma ================= */
function componentsFromMask(mask,w,h){
  const seen=new Uint8Array(w*h), comps=[];
  const Qx=new Int32Array(w*h), Qy=new Int32Array(w*h);
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const i=y*w+x; if(mask[i] && !seen[i]){
      let head=0, tail=0; Qx[tail]=x; Qy[tail]=y; tail++; seen[i]=1;
      const px=[]; let minx=x,maxx=x,miny=y,maxy=y, area=0, bx=0,by=0;
      while(head<tail){
        const cx=Qx[head], cy=Qy[head]; head++;
        const p=cy*w+cx; px.push(p); area++; bx+=cx; by+=cy;
        minx=Math.min(minx,cx); maxx=Math.max(maxx,cx);
        miny=Math.min(miny,cy); maxy=Math.max(maxy,cy);
        for(let j=-1;j<=1;j++) for(let i2=-1;i2<=1;i2++){
          const nx=cx+i2, ny=cy+j;
          if(nx>=0&&ny>=0&&nx<w&&ny<h){
            const q=ny*w+nx; if(mask[q] && !seen[q]){ seen[q]=1; Qx[tail]=nx; Qy[tail]=ny; tail++; }
          }
        }
      }
      const cx = bx/area, cy = by/area;
      // covarianza para elongación
      let cxx=0, cyy=0, cxy=0, perim=0;
      for(const p of px){
        const x0 = p%w, y0=(p-x0)/w;
        const dx=x0-cx, dy=y0-cy;
        cxx+=dx*dx; cyy+=dy*dy; cxy+=dx*dy;
        // perímetro aproximado: vecinos fuera
        let neigh=0;
        for(let j=-1;j<=1;j++) for(let i2=-1;i2<=1;i2++){
          const nx=x0+i2, ny=y0+j;
          if(nx>=0&&ny>=0&&nx<w&&ny<h){ if(mask[ny*w+nx]) neigh++; }
        }
        if(neigh<8) perim++;
      }
      const tr=cxx+cyy, det=cxx*cyy - cxy*cxy;
      const tmp=Math.sqrt(Math.max(tr*tr-4*det,0));
      const l1=(tr+tmp)/(2*area), l2=(tr-tmp)/(2*area);
      comps.push({pixels:px, area, minx,maxx,miny,maxy, cx,cy, perim, elong: Math.sqrt((l1+1e-9)/(l2+1e-9))});
    }
  }
  return comps;
}
function maskToOverlay(mask,w,h,[R,G,B,A]=[235,64,52,160]){
  const can=document.createElement("canvas"); can.width=w; can.height=h;
  const ctx=can.getContext("2d"); const id=ctx.createImageData(w,h);
  for(let i=0;i<w*h;i++) if(mask[i]){ const p=i*4; id.data[p]=R; id.data[p+1]=G; id.data[p+2]=B; id.data[p+3]=A; }
  ctx.putImageData(id,0,0); return can;
}

/* ================= Detección de grietas ================= */
function detectCracksOverlay(norm,w,h,cx,cy,r, params){
  const { mag, mean, std } = sobel(norm,w,h);
  // Black-hat gris para realzar líneas oscuras finas
  const closed = closingGray(norm,w,h, params.seCrack);
  const bh = new Float32Array(norm.length);
  for(let i=0;i<bh.length;i++) bh[i]=Math.max(0, closed[i]-norm[i]);

  // combinar black-hat + gradiente
  // normaliza ambos
  let bmax=0,gmax=0; for(let i=0;i<bh.length;i++){ if(bh[i]>bmax) bmax=bh[i]; if(mag[i]>gmax) gmax=mag[i]; }
  const comb=new Float32Array(norm.length);
  for(let i=0;i<comb.length;i++){
    const a=bmax? bh[i]/bmax : 0;
    const g=gmax? mag[i]/gmax : 0;
    comb[i]=a*Math.pow(g,0.9);
  }

  // umbral adaptativo
  let m=0,s2=0; for(let i=0;i<comb.length;i++){ m+=comb[i]; s2+=comb[i]*comb[i]; }
  const mu=m/comb.length, sigma=Math.sqrt(Math.max(s2/comb.length - mu*mu, 1e-8));
  const thr = mu + params.kCrack*sigma;

  // máscara dentro del disco (evita el borde del 10%)
  const mask=new Uint8Array(w*h); const r2=r*r, rIn=(r*0.90)**2;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const dx=x-cx, dy=y-cy; const d=dx*dx+dy*dy;
    if(d<=r2 && d>=rIn && comb[y*w+x]>thr) mask[y*w+x]=1; // primero borde interno para cerrar grietas periféricas
  }
  // también dentro del disco completo
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const dx=x-cx, dy=y-cy; const d=dx*dx+dy*dy;
    if(d<rIn && comb[y*w+x]>thr) mask[y*w+x]=1;
  }

  const comps = componentsFromMask(mask,w,h);
  const keep=new Uint8Array(w*h); let area=0,cnt=0;
  const diskArea = Math.PI*r*r;
  comps.forEach(c=>{
    const rel=c.area/diskArea;
    // cracks: alargamiento alto, tamaño razonable
    if(c.elong>=params.minElong && rel>params.minCrackArea && rel<params.maxCrackArea){
      cnt++; for(const p of c.pixels){ if(!keep[p]){ keep[p]=1; area++; } }
    }
  });

  return { can: maskToOverlay(keep,w,h,[235,64,52,160]), count:cnt, areaRatio: area/Math.max(1,diskArea) };
}

/* ================= Detección de huecos ================= */
function detectPitsOverlay(norm,w,h,cx,cy,r, params){
  // DoG (blur pequeño vs grande) sobre imagen normalizada invertida
  const inv = new Float32Array(norm.length); for(let i=0;i<inv.length;i++) inv[i]=1-norm[i];
  const bSmall = blur(inv,w,h, params.sePitSmall, 2);
  const bLarge = blur(inv,w,h, params.sePitLarge, 2);
  const dog = new Float32Array(inv.length);
  for(let i=0;i<dog.length;i++) dog[i]=Math.max(0, bSmall[i]-bLarge[i]); // blobs locales

  let m=0,s2=0; for(let i=0;i<dog.length;i++){ m+=dog[i]; s2+=dog[i]*dog[i]; }
  const mu=m/dog.length, sigma=Math.sqrt(Math.max(s2/dog.length - mu*mu, 1e-8));
  const thr = mu + params.kPit*sigma;

  const mask=new Uint8Array(w*h);
  const r2=r*r, rEdge=(r*0.92)**2; // evita confundir borde
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const dx=x-cx, dy=y-cy; const d=dx*dx+dy*dy;
    if(d<rEdge && dog[y*w+x]>thr) mask[y*w+x]=1;
  }

  const comps = componentsFromMask(mask,w,h);
  const keep=new Uint8Array(w*h); let area=0,cnt=0;
  const diskArea = Math.PI*r*r;
  comps.forEach(c=>{
    const rel=c.area/diskArea;
    const circ = (4*Math.PI*c.area)/Math.max(1, c.perim*c.perim); // 0..1
    // pits: redondos, tamaño medio
    if(c.elong<=params.maxElongPit && circ>=params.minCirc && rel>params.minPitArea && rel<params.maxPitArea){
      cnt++; for(const p of c.pixels){ if(!keep[p]){ keep[p]=1; area++; } }
    }
  });

  return { can: maskToOverlay(keep,w,h,[59,130,246,150]), count:cnt, areaRatio: area/Math.max(1,diskArea) };
}

/* ================= Recorte circular ================= */
function autoCircle(canvas){
  const id=getImageData(canvas), {gray,width:w,height:h,mean,std}=grayscale(id);
  // Umbral oscuro para aislar disco
  const thr=mean - 0.35*std; const m=new Uint8Array(w*h);
  for(let i=0;i<gray.length;i++) m[i]= gray[i]<thr ? 1:0;
  // centro aproximado
  let sx=0,sy=0,c=0; for(let y=0;y<h;y++) for(let x=0;x<w;x++){ const i=y*w+x; if(m[i]){sx+=x; sy+=y; c++;} }
  const cx=c? sx/c : w/2, cy=c? sy/c : h/2;
  // radio robusto (p80)
  const dists=[]; for(let y=0;y<h;y++) for(let x=0;x<w;x++){ const i=y*w+x; if(m[i]) dists.push(Math.hypot(x-cx,y-cy)); }
  dists.sort((a,b)=>a-b); const r = dists.length? dists[Math.floor(dists.length*0.8)] : Math.min(w,h)*0.45;
  return { cx, cy, r, g: {gray,width:w,height:h} };
}
function cropCircleFromCanvas(src,cx,cy,r,pad=1.18){
  const size=Math.max(64, Math.round(2*r*pad));
  const out=document.createElement("canvas"); out.width=size; out.height=size;
  const ctx=out.getContext("2d"), C=size/2;
  ctx.save(); ctx.beginPath(); ctx.arc(C,C,r,0,Math.PI*2); ctx.clip();
  ctx.drawImage(src, Math.round(C-cx), Math.round(C-cy));
  ctx.restore();
  // alpha fuera del disco
  const id=ctx.getImageData(0,0,size,size); const d=id.data;
  for(let y=0;y<size;y++) for(let x=0;x<size;x++){
    const dx=x-C, dy=y-C; if(dx*dx+dy*dy>r*r){ d[(y*size+x)*4+3]=0; }
  }
  ctx.putImageData(id,0,0);
  return out;
}

/* ================== Análisis principal ================== */
async function analyzePuck(srcCanvas, userCircle, params){
  // 1) localizar/cortar
  const det = userCircle || autoCircle(srcCanvas);
  const crop = cropCircleFromCanvas(srcCanvas, det.cx, det.cy, det.r, 1.18);
  const { data, width:w, height:h } = crop.getContext("2d").getImageData(0,0,crop.width,crop.height);
  const gray = new Float32Array(w*h); for(let i=0,j=0;i<data.length;i+=4,j++) gray[j]=toGray(data[i],data[i+1],data[i+2]);

  // 2) normalización
  const norm = normalizeIllum(gray,w,h);
  const C=w/2, R=det.r;

  // 3) detecciones
  const cracks = detectCracksOverlay(norm,w,h,C,C,R, params);
  const pits   = detectPitsOverlay(norm,w,h,C,C,R, params);

  // 4) overlays
  const [cracksURL, pitsURL] = await Promise.all([canvasToURL(cracks.can), canvasToURL(pits.can)]);
  return {
    size:{w,h,r:Math.round(R)}, cropURL: await canvasToURL(crop),
    cracks:{count:cracks.count, area:cracks.areaRatio, url:cracksURL},
    pits:{count:pits.count, area:pits.areaRatio, url:pitsURL}
  };
}

/* ================= UI ================= */
function Toggle({label,checked,onChange}){ return (
  <label style={{display:"flex",alignItems:"center",gap:8}}>
    <input type="checkbox" checked={checked} onChange={e=>onChange(e.target.checked)} />
    <span>{label}</span>
  </label>
); }
function Slider({label,min,max,step,value,onChange,fmt=(v)=>v}){
  return (
    <div style={{display:"grid",gridTemplateColumns:"110px 1fr 70px",alignItems:"center",gap:10}}>
      <div className="small">{label}</div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(parseFloat(e.target.value))}/>
      <div className="mono small" style={{textAlign:"right"}}>{fmt(value)}</div>
    </div>
  );
}

/* ================= App ================= */
export default function App(){
  const topRef = useRef(null);
  const botRef = useRef(null);

  const [overlay, setOverlay] = useState({ cracks:true, pits:true });
  const [params, setParams] = useState({
    // cracks
    seCrack: 3,       // struct elem (pix) para black-hat en grietas
    kCrack: 1.05,     // sensibilidad a sigma
    minElong: 2.2,    // elongación mínima
    minCrackArea: 0.00005,
    maxCrackArea: 0.03,
    // pits
    sePitSmall: 2.5,
    sePitLarge: 7.5,
    kPit: 1.0,
    maxElongPit: 1.9,
    minCirc: 0.45,
    minPitArea: 0.00025,
    maxPitArea: 0.045
  });

  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);

  async function run(){
    const fTop = topRef.current?.files?.[0];
    const fBot = botRef.current?.files?.[0];
    if(!fTop || !fBot){ alert("Sube dos fotos: Superior (lado ducha) e Inferior (lado canasta)."); return; }
    setBusy(true);
    try{
      const imgTop = await fileToImageBitmap(fTop);
      const imgBot = await fileToImageBitmap(fBot);
      const canTop = drawToCanvas(imgTop, MAX_DIM);
      const canBot = drawToCanvas(imgBot, MAX_DIM);
      const top = await analyzePuck(canTop, null, params);
      const bot = await analyzePuck(canBot, null, params);
      setResults({ top, bot });
    } finally { setBusy(false); }
  }

  return (
    <div style={{padding:16, fontFamily:"ui-sans-serif, system-ui"}}>
      <h2 style={{margin:"0 0 8px 0"}}>Puck Diagnosis — Cracks & Pits (v2)</h2>

      <div className="card" style={{padding:12, display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(260px,1fr))", gap:12}}>
        <div>
          <div className="small">Superior (lado ducha)</div>
          <input ref={topRef} type="file" accept="image/*" />
        </div>
        <div>
          <div className="small">Inferior (lado canasta)</div>
          <input ref={botRef} type="file" accept="image/*" />
        </div>
        <div style={{alignSelf:"end"}}><button className="btn" onClick={run} disabled={busy}>{busy? "Analizando..." : "Analizar"}</button></div>
      </div>

      <div style={{height:10}}/>

      <div className="card" style={{padding:12, display:"grid", gridTemplateColumns:"1fr 1fr", gap:18}}>
        <div>
          <h4 style={{margin:"4px 0"}}>Overlays</h4>
          <div style={{display:"flex",gap:16}}>
            <Toggle label="Cracks" checked={overlay.cracks} onChange={v=>setOverlay(o=>({...o,cracks:v}))}/>
            <Toggle label="Pits" checked={overlay.pits} onChange={v=>setOverlay(o=>({...o,pits:v}))}/>
          </div>
        </div>
        <div>
          <h4 style={{margin:"4px 0"}}>Parámetros (ajusta sensibilidad)</h4>
          <div style={{display:"grid",gap:6}}>
            <Slider label="kCrack (σ)" min={0.6} max={1.6} step={0.01} value={params.kCrack} onChange={v=>setParams(p=>({...p,kCrack:v}))}/>
            <Slider label="SE grieta (px)" min={2} max={6} step={0.5} value={params.seCrack} onChange={v=>setParams(p=>({...p,seCrack:v}))}/>
            <Slider label="Min elongación" min={1.5} max={4} step={0.1} value={params.minElong} onChange={v=>setParams(p=>({...p,minElong:v}))}/>
            <Slider label="kPit (σ)" min={0.6} max={1.6} step={0.01} value={params.kPit} onChange={v=>setParams(p=>({...p,kPit:v}))}/>
            <Slider label="DoG pequeño" min={1.5} max={4.5} step={0.1} value={params.sePitSmall} onChange={v=>setParams(p=>({...p,sePitSmall:v}))}/>
            <Slider label="DoG grande" min={5} max={12} step={0.5} value={params.sePitLarge} onChange={v=>setParams(p=>({...p,sePitLarge:v}))}/>
          </div>
        </div>
      </div>

      {results && (
        <div style={{marginTop:16, display:"grid", gridTemplateColumns:"1fr 1fr", gap:16}}>
          {["top","bot"].map((side,i)=>(
            <div key={side} className="card" style={{padding:12}}>
              <div style={{fontWeight:700, marginBottom:6}}>{side==="top"?"Superior (ducha)":"Inferior (canasta)"}</div>
              <div className="thumb" style={{position:"relative", width:"100%", paddingTop:"100%", background:"#101827", borderRadius:8, overflow:"hidden"}}>
                <img alt="crop" src={results[side].cropURL} style={{position:"absolute", inset:0, width:"100%", height:"100%", objectFit:"contain"}}/>
                {overlay.cracks && <img alt="cracks" src={results[side].cracks.url} style={{position:"absolute", inset:0, width:"100%", height:"100%", objectFit:"contain"}}/>}
                {overlay.pits &&   <img alt="pits"   src={results[side].pits.url}   style={{position:"absolute", inset:0, width:"100%", height:"100%", objectFit:"contain"}}/>}
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8,marginTop:8,fontSize:12}}>
                <div>Cracks: <b>{results[side].cracks.count}</b></div>
                <div>Área cracks: <b>{(results[side].cracks.area*100).toFixed(2)}%</b></div>
                <div>Pits: <b>{results[side].pits.count}</b></div>
                <div>Área pits: <b>{(results[side].pits.area*100).toFixed(2)}%</b></div>
                <div>Radio: <b>{results[side].size.r}px</b></div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* estilos mínimos */}
      <style>{`
        .card{background:#0b1220;border:1px solid #1f2a44;border-radius:10px}
        .btn{background:#2563eb;color:#fff;border:none;border-radius:8px;padding:8px 12px;cursor:pointer}
        .btn:disabled{opacity:0.6}
        .small{color:#9aa4b2;font-size:12px}
        .mono{font-family:ui-monospace, SFMono-Regular, Menlo, monospace}
      `}</style>
    </div>
  );
}
