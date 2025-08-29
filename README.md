# Puck Diagnosis v0.2

App React (Vite) para análisis visual de pucks con **overlays opcionales**:
- Recorta y centra el puck, **fondo transparente**.
- Métricas: Δ borde-centro (ECI), extremos tonales.
- **Heatmap** de variación local (bloques).
- **Líneas guía** (anillos y radios).
- **Resaltado de grietas/canales** (Sobel + umbral adaptativo).
- **Mapa de sectores** (desviación angular).
- Gráficos: **perfil radial** y **histograma**.

## Uso
```bash
npm i
npm run dev
```

## Despliegue (Vercel)
Framework: Vite · Build: `npm run build` · Output: `dist/`

## Ajustes clave (src/App.jsx)
- Detección del puck: `darkMask(..., 0.35)`
- Grietas: umbral en `drawEdgesOverlay` (`mean + 1.1*std`)
- Sectores: `sectorDeviation(..., sectors=24)` y escala `1.2*std`
- Heatmap: tamaño de bloque en `localVariance(..., 14)`
