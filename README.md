# Puck Diagnosis

App ligera (React + Vite) para analizar canalización en pucks de espresso a partir de fotos.
- Recorta y centra automáticamente el puck y remueve el fondo (transparente).
- Calcula métricas simples (Δ borde‑centro, heterogeneidad, extremos) y clasifica (OK/Leve/Severa).
- Genera heatmap opcional de variación local.
- 100% client‑side.

## Desarrollo
```bash
npm i
npm run dev
```

## Build y despliegue (Vercel)
```bash
npm run build
# sube el repo a GitHub y conéctalo a Vercel (framework: Vite)
```

## Estructura
- `src/App.jsx` — lógica de procesamiento (sin dependencias externas).
- `index.html` — estilos mínimos embebidos.
