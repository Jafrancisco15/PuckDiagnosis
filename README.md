# Puck Diagnosis v0.3

Mejoras clave respecto a v0.2:
- **Overlays 1:1**: todos (heatmap, guías, grietas, sectores) tienen el **mismo tamaño** que el recorte → se alinean perfectos.
- **Leyenda** en el **mapa de sectores** (rojo = más oscuro/atascos, azul = más claro/flujo rápido).
- **Recomendaciones analíticas** y priorizadas con explicación de métricas: ECI, extremos, desviación angular, densidad de grietas y desalineación.
- **Histogramas/perfil** redimensionados y contenidos en tarjetas separadas.

## Uso
```bash
npm i
npm run dev
```

## Deploy (Vercel)
Framework: **Vite** · Build: `npm run build` · Output: `dist/`

## Ajustes
- Detección puck: `darkMask(..., 0.35)`
- Grietas: `drawEdgesOverlay` (umbral `mean + 1.1*std`)
- Sectores: `drawSectorsOverlay(..., 24)` + leyenda
- Heatmap: `localVariance(..., 64)` (bloques) → se pinta a tamaño del recorte


## v0.4 — Pares Frontal/Trasera
- Formulario para crear **sets** con 2 imágenes por puck (frontal + trasera).
- Opción **rotar trasera 180°** para alinear orientación.
- Métrica **correlación sectorial** entre caras y recomendaciones combinadas.
