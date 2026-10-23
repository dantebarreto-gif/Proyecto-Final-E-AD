// Genera report/Presentacion_Final.pptx a partir de los resultados guardados.
//
//   npm install pptxgenjs react react-dom react-icons sharp
//   node scripts/build_slides.js
//
// Las cifras se leen de results/demo/summary_10M.json y results/bench/*.csv,
// así que basta con volver a correr el script tras repetir los experimentos.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.join(__dirname, "..");
const R = (...p) => path.join(ROOT, ...p);
const OUT = R("report", "Presentacion_Final.pptx");

// ---------- datos ----------
function csv(name) {
  const p = R("results", "bench", `${name}.csv`);
  if (!fs.existsSync(p)) return [];
  const [head, ...rows] = fs.readFileSync(p, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => {
    const v = r.split(",");
    const o = {};
    cols.forEach((c, i) => (o[c] = isNaN(+v[i]) || v[i] === "" ? v[i] : +v[i]));
    return o;
  });
}
const S = JSON.parse(fs.readFileSync(R("results", "demo", "summary_10M.json"), "utf8"));
const base = csv("base_structures");
const mh = csv("minhash_vs_bruteforce");
const ann = csv("ann_vs_exact");
const pr = csv("pagerank_vs_degree");
const com = csv("communities");
const slopes = csv("slopes");
const BASE = ["LinkedQueue", "MergeSort", "BinaryHeap", "HashTable", "LRUCache", "BPlusTree", "KDTree", "KNN fuerza bruta", "BloomFilter"];

const find = (rows, f) => rows.find(f) || {};
const fmt = (x, d = 0) => Number(x).toLocaleString("es-PE", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => `${fmt(100 * x, d)} %`;
const slope = (st, op) => find(slopes, (r) => r.Estructura === st && r["Operación"] === op).Pendiente;

const kd10 = find(base, (r) => r.structure === "KDTree" && r.op === "knn10(100 consultas)" && r.n === 1e7);
const bf10 = find(base, (r) => r.structure === "KNN fuerza bruta" && r.op === "knn10(100 consultas)" && r.n === 1e7);
const bpS = find(base, (r) => r.structure === "BPlusTree" && r.op === "search(1000)" && r.n === 1e7);
const bpBulk = find(base, (r) => r.structure === "BPlusTree" && r.op === "bulk_load" && r.n === 1e7);
const hash10 = find(base, (r) => r.structure === "HashTable" && r.n === 1e7);
const lru10 = find(base, (r) => r.structure === "LRUCache" && r.n === 1e7);
const mhRow = find(mh, (r) => r.n === 1e6 && r.bands === 32).n ? find(mh, (r) => r.n === 1e6 && r.bands === 32)
  : find(mh, (r) => r.n === 1e5 && r.bands === 32);
const annIvf = ann.filter((r) => r.method === "IVF").sort((a, b) => b.n - a.n || a.nprobe - b.nprobe);
const ivfTop = annIvf.find((r) => r.nprobe === 8) || annIvf[0] || {};
const annExact = find(ann, (r) => r.method === "Fuerza bruta (exacto)" && r.n === ivfTop.n);
const hnsw = find(ann, (r) => r.method === "HNSW");
const prBig = pr.filter((r) => r.method === "PageRank").sort((a, b) => b.edges - a.edges)[0] || {};
const louBig = com.filter((r) => r.method === "Louvain").sort((a, b) => b.edges - a.edges)[0] || {};
const lpaSame = find(com, (r) => r.method === "Label Propagation" && r.edges === louBig.edges);

// ---------- tema ----------
const THEME = {
  name: "Stream Comercio",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F3B", lt1: "FFFFFF", dk2: "3A3F5C", lt2: "F5F3EE",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "6250D6", accent5: "E87BA4", accent6: "C9C4B8",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Pipeline de data streams para comercio electrónico";
pres.author = "Garcia, Barreto, Chulluncuy, Hilario";
pres.subject = "Proyecto Final EDA 2026-2, UTEC";
const C = pres.SchemeColor;
const W = 13.333;

pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.9, fontSize: 40, bold: true,
      color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.2, w: 11.7, h: 2.2, fontSize: 16,
      color: C.accent6, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 28, bold: true,
      color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Proyecto Final EDA 2026-2 · Escenario B: comercio electrónico", options: {
      x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});
pres.defineSlideMaster({
  title: "CIERRE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 28, bold: true,
      color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- utilidades ----------
async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// insignia de etapa: el motivo visual que se repite en todo el deck
function badge(slide, label, x, y, color, size = 0.62) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: size, h: size, rectRadius: 0.12, fill: { color },
    line: { color, width: 0 }, objectName: `insignia ${label}` });
  slide.addText(label, { x, y, w: size, h: size, align: "center", valign: "middle", fontSize: size > 0.55 ? 16 : 12,
    bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: `texto insignia ${label}` });
}

function card(slide, { x, y, w, h, tag, color, title, lines, stat, statLabel }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: C.background2, width: 0 }, objectName: `tarjeta ${tag}` });
  badge(slide, tag, x + 0.3, y + 0.3, color);
  slide.addText(title, { x: x + 1.1, y: y + 0.28, w: w - 1.3, h: 0.66, fontSize: 18, bold: true, color: C.text1,
    valign: "middle", margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
  slide.addText(lines.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lines.length - 1 } })), {
    x: x + 0.3, y: y + 1.1, w: w - 0.6, h: h - (stat ? 2.65 : 1.35), fontSize: 15, color: C.text2, valign: "top",
    margin: 0, paraSpaceAfter: 10, isTextBox: true });
  if (stat) {
    slide.addText(stat, { x: x + 0.3, y: y + h - 1.45, w: w - 0.6, h: 0.75, fontSize: 30, bold: true, color,
      margin: 0, valign: "bottom", isTextBox: true, fontFace: THEME.headFontFace });
    slide.addText(statLabel, { x: x + 0.3, y: y + h - 0.68, w: w - 0.6, h: 0.45, fontSize: 11.5, color: C.text2,
      margin: 0, valign: "top", isTextBox: true });
  }
}

function statBlock(slide, x, y, w, value, label, color) {
  slide.addText(value, { x, y, w, h: 0.8, fontSize: 36, bold: true, color, margin: 0, valign: "bottom",
    isTextBox: true, fontFace: THEME.headFontFace });
  slide.addText(label, { x, y: y + 0.82, w, h: 0.55, fontSize: 12.5, color: C.text2, margin: 0, valign: "top",
    isTextBox: true });
}

function imageFit(file, maxW, maxH) {
  const meta = require("child_process").execSync(`python3 -c "from PIL import Image;print(*Image.open('${file}').size)"`)
    .toString().trim().split(" ").map(Number);
  const r = Math.min(maxW / meta[0], maxH / meta[1]);
  return { w: meta[0] * r, h: meta[1] * r };
}

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt" };

async function main() {
  const ic = {
    queue: await icon(fa.FaStream, HEX.accent1), search: await icon(fa.FaBolt, HEX.accent1),
    shield: await icon(fa.FaShieldAlt, HEX.accent2), graph: await icon(fa.FaProjectDiagram, HEX.accent4),
    users: await icon(fa.FaUsers, HEX.accent3),
  };

  // 1. Portada
  pres.addSection({ title: "Introducción" });
  let s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Introducción" });
  s.addText("Diez estructuras de datos en cadena para el stream de una tienda en línea", { placeholder: "title" });
  s.addText([
    { text: "Proyecto Final · Estructuras de Datos y Algoritmos 2026-2 · UTEC", options: { breakLine: true } },
    { text: "Sebastian Garcia · Dante Barreto · Clinton Chulluncuy · Ronald Hilario", options: { breakLine: true } },
    { text: "29 de octubre de 2026", options: {} },
  ], { placeholder: "body" });
  s.addNotes("Presenta: Dante. Buenas tardes. Construimos un pipeline de 10 etapas que procesa 10 millones de eventos de una tienda en línea. Cada etapa usa la estructura de datos adecuada y su salida alimenta a la siguiente. En 15 minutos veremos el problema, la arquitectura, cada familia de estructuras, la escalabilidad y una demo.");

  // 2. Problema
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Introducción" });
  s.addText("Cuatro necesidades del negocio, cada una con su estructura", { placeholder: "title" });
  const needs = [
    [ic.queue, "Atender primero lo urgente", "Compras express antes que vistas: cola FIFO y heap de prioridad"],
    [ic.search, "Responder en microsegundos", "Productos y usuarios por id, rangos de precio: hash, LRU y B+Tree en disco"],
    [ic.shield, "Detectar fraude en segundos", "Lista negra y ráfagas por tarjeta con memoria fija: Bloom y Count-Min"],
    [ic.graph, "Recomendar y descubrir", "Productos parecidos y comprados juntos: MinHash, IVF, PageRank y Louvain"],
  ];
  needs.forEach(([img, t, d], i) => {
    const x = 0.6 + i * 3.08;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 2.85, h: 3.6, rectRadius: 0.08, fill: { color: C.background2 },
      line: { color: C.background2, width: 0 }, objectName: `necesidad ${i + 1}` });
    s.addImage({ data: img, x: x + 0.3, y: 1.9, w: 0.6, h: 0.6, altText: t });
    s.addText(t, { x: x + 0.3, y: 2.7, w: 2.3, h: 0.85, fontSize: 17, bold: true, color: C.text1, margin: 0,
      valign: "top", isTextBox: true, fontFace: THEME.headFontFace });
    s.addText(d, { x: x + 0.3, y: 3.6, w: 2.3, h: 1.4, fontSize: 13, color: C.text2, margin: 0, valign: "top",
      isTextBox: true });
  });
  s.addText([
    { text: "La condición del enunciado: ", options: { bold: true, color: C.text1 } },
    { text: "no son módulos sueltos. La salida de cada etapa es la entrada real de la siguiente, y cada estructura aproximada se mide contra su versión exacta.", options: { color: C.text2 } },
  ], { x: 0.6, y: 5.55, w: 12.1, h: 0.9, fontSize: 15, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("Presenta: Dante. El escenario B es comercio electrónico. Hay cuatro necesidades y cada una pide una estructura distinta. Lo importante es que no son módulos sueltos: un evento entra por la cola y su recorrido termina en una recomendación. Además, cada estructura aproximada se compara con la exacta.");

  // 3. Datos
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Introducción" });
  s.addText("Datos: 10M eventos reales de REES46 más el componente de stream", { placeholder: "title" });
  statBlock(s, 0.6, 1.5, 3.8, fmt(S.eventos_recibidos), "eventos reales del 1 al 8 de octubre de 2019", C.accent1);
  statBlock(s, 4.7, 1.5, 3.8, fmt(S.S3.productos), `productos y ${fmt(S.S3.usuarios)} usuarios`, C.accent3);
  statBlock(s, 8.8, 1.5, 3.9, fmt(S.S1.duplicados_eliminados), "duplicados inyectados (1,5 %), todos eliminados", C.accent2);
  s.addText([
    { text: "REES46 (Kaggle): ", options: { bold: true, color: C.text1 } },
    { text: "DuckDB lee 2019-Oct.csv (5,7 GB) en streaming y toma los primeros 10M eventos en orden temporal: event_time, event_type, product_id, category_code, brand, price, user_id, user_session. El 95,6 % son vistas.", options: { color: C.text2, breakLine: true } },
    { text: "Componente sintético con semilla fija: ", options: { bold: true, color: C.text1 } },
    { text: "dispositivo, IP, tarjeta, usuario premium, tipo de envío, consultas, duplicados reenviados y ráfagas de fraude de 25 a 120 intentos en menos de 5 minutos.", options: { color: C.text2, breakLine: true } },
    { text: "Experimentos de escalabilidad: ", options: { bold: true, color: C.text1 } },
    { text: "usan entradas generadas para controlar n y conocer la respuesta exacta; el pipeline completo corre sobre los datos reales.", options: { color: C.text2 } },
  ], { x: 0.6, y: 3.3, w: 12.1, h: 3.4, fontSize: 18, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 16 });
  s.addNotes("Presenta: Dante. Usamos los primeros 10 millones de eventos reales de REES46, del 1 al 8 de octubre de 2019. Encima agregamos lo que el dataset no trae y que el enunciado pide simular: dispositivo, tarjeta, tipo de envío, duplicados y ráfagas de fraude. Las etiquetas de fraude y duplicado solo se usan para medir precisión y recall, nunca para decidir.");

  // 4. Arquitectura
  pres.addSection({ title: "Arquitectura" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Arquitectura" });
  s.addText("Arquitectura: un evento recorre las 10 etapas en orden", { placeholder: "title" });
  const pipe = R("docs", "pipeline.png");
  const pf = imageFit(pipe, 11.4, 5.2);
  s.addImage({ path: pipe, x: (W - pf.w) / 2, y: 1.45, w: pf.w, h: pf.h,
    altText: "Diagrama del pipeline: S0 cola, S1 limpieza, S2 heap, S3 hash, S6 filtrado, luego S4, S5, S7, S8 y S9" });
  s.addText("Lotes de 5.000 eventos. Lo incremental corre por evento; KD-Tree, MinHash, SVD + IVF, PageRank y Louvain se refrescan cada 2M eventos.",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 12.5, color: C.text2, margin: 0, isTextBox: true, italic: true });
  s.addNotes("Presenta: Dante. Este es el flujo. El filtrado S6 corre justo después de resolver ids en S3, de modo que lo bloqueado no contamina índices, similitud ni grafo. Las estructuras costosas se reconstruyen por lotes cada 2 millones de eventos, como una arquitectura lambda.");

  // 5. S0-S2
  pres.addSection({ title: "Etapas" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Etapas" });
  s.addText("S0 a S2: entrar, limpiar y priorizar", { placeholder: "title" });
  const cw = 3.9;
  card(s, { x: 0.6, y: 1.5, w: cw, h: 5.2, tag: "S0", color: C.accent1, title: "Cola FIFO enlazada",
    lines: ["Buffer entre el lector y el consumidor", "Capacidad máxima: si se llena, descarta y cuenta (backpressure)", "enqueue y dequeue O(1)"],
    stat: `pend. ${fmt(slope("LinkedQueue", "enqueue+dequeue"), 2)}`, statLabel: "en log-log, esperado 1 para O(n)" });
  card(s, { x: 0.6 + cw + 0.2, y: 1.5, w: cw, h: 5.2, tag: "S1", color: C.accent1, title: "Merge sort + búsqueda binaria",
    lines: ["Valida y normaliza cada evento", "Busca cada id en la ventana ordenada de los últimos 500K", "O(b log W) por lote"],
    stat: fmt(S.S1.duplicados_eliminados), statLabel: `duplicados eliminados, ${fmt(S.S1.duplicados_reales_detectados)} reales` });
  card(s, { x: 0.6 + 2 * (cw + 0.2), y: 1.5, w: cw, h: 5.2, tag: "S2", color: C.accent1, title: "Heap binario",
    lines: ["Express, premium, estándar, programada, carrito, vista", "Desempate por orden de llegada", "push y pop O(log n)"],
    stat: fmt(S.S2["compra express"]), statLabel: "compras express atendidas primero" });
  s.addNotes("Presenta: Sebastian. S0 es una cola enlazada con capacidad, que da backpressure. S1 elimina duplicados buscando cada id en una ventana ordenada con búsqueda binaria, sin guardar todos los ids. S2 es un heap: las compras express salen primero. Las pendientes medidas están cerca de lo esperado.");

  // 6. S3-S5
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Etapas" });
  s.addText("S3 a S5: consultar rápido, en memoria y en disco", { placeholder: "title" });
  card(s, { x: 0.6, y: 1.5, w: cw, h: 5.2, tag: "S3", color: C.accent3, title: "Hash, LRU y matriz dispersa",
    lines: ["Sondeo lineal con carga ≤ 0,7", `LRU de fichas: ${fmt(100 * +String(lru10.extra).split("=")[1], 1)} % de aciertos`, `Matriz usuario × producto: ${fmt(S.S3.interacciones_nnz)} celdas`],
    stat: `${fmt(hash10.per_op_us, 1)} µs`,
    statLabel: `por operación con 10M claves (${fmt(+String(hash10.extra).split("=")[1], 2)} sondeos)` });
  card(s, { x: 0.6 + cw + 0.2, y: 1.5, w: cw, h: 5.2, tag: "S4", color: C.accent3, title: "B+Tree en disco",
    lines: ["Páginas de 4 KB, 255 claves por hoja", "Buffer pool LRU con write-back", `Carga masiva de 10M claves en ${fmt(bpBulk.mean_s, 1)} s`],
    stat: `${fmt(bpS.per_op_us, 0)} µs`, statLabel: "por búsqueda con 10M claves, altura 3" });
  card(s, { x: 0.6 + 2 * (cw + 0.2), y: 1.5, w: cw, h: 5.2, tag: "S5", color: C.accent3, title: "KD-Tree 3D",
    lines: ["Precio, popularidad y conversión", "Corte por mediana en el eje más extendido", `Fuerza bruta: ${fmt(bf10.per_op_us / 1000, 0)} ms por consulta`],
    stat: `${fmt(bf10.per_op_us / kd10.per_op_us, 0)}x`, statLabel: `más rápido que fuerza bruta (${fmt(kd10.per_op_us / 1000, 2)} ms, n = 10M)` });
  s.addNotes("Presenta: Dante. S3 resuelve ids en una tabla hash propia y cachea con LRU. La matriz dispersa usuario por producto alimenta a S7. S4 es un B+Tree real en disco con páginas de 4 KB: con 10 millones de claves tiene altura 3, así que una búsqueda lee 3 páginas. S5 es un KD-Tree: con 10 millones de puntos responde KNN en menos de un milisegundo, contra casi 400 ms de fuerza bruta.");

  // 7. S6
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Etapas" });
  s.addText("S6 Filtrado: fraude detectado con unos pocos kilobytes", { placeholder: "title" });
  badge(s, "S6", 0.6, 1.55, C.accent2);
  s.addText([
    { text: "Bloom Filter: ", options: { bold: true, color: C.text1 } },
    { text: "lista negra de tarjetas y dispositivos, p = 1 %, k = 7. Todo positivo se verifica contra el conjunto exacto antes de bloquear.", options: { color: C.text2, breakLine: true } },
    { text: "Count-Min Sketch: ", options: { bold: true, color: C.text1 } },
    { text: "intentos por tarjeta y por dispositivo en una ventana de 10 minutos (anillo de 10 sub-sketches). Alerta con más de 12 o 20 intentos.", options: { color: C.text2 } },
  ], { x: 1.45, y: 1.5, w: 5.6, h: 2.6, fontSize: 15, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 12 });
  statBlock(s, 0.6, 4.3, 3.1, pct(S.S6.precision, 1), "precisión: lo demás son cuentas reales con ráfagas de compra", C.accent2);
  statBlock(s, 3.9, 4.3, 3.1, pct(S.S6.recall, 1), "recall: las primeras compras de cada ráfaga pasan", C.accent2);
  statBlock(s, 0.6, 5.65, 3.1, `${fmt(S.S6.bloom_memory_bytes / 1024, 1)} KB`, "memoria del Bloom", C.text1);
  statBlock(s, 3.9, 5.65, 3.1, fmt(S.S6.bloom_false_positives), "falsos positivos, descartados por la verificación", C.text1);
  const bloom = csv("bloom_vs_exact");
  const bm = bloom.filter((r) => r.structure === "BloomFilter" && r.p === 0.01).sort((a, b) => a.n - b.n);
  const hm = bloom.filter((r) => r.structure === "HashSet exacto").sort((a, b) => a.n - b.n);
  if (bm.length && hm.length) {
    const ns = bm.map((r) => r.n).filter((n) => hm.some((h) => h.n === n));
    const mb = (rows) => ns.map((n) => +(rows.find((r) => r.n === n).memory_B / 1e6).toPrecision(3));
    s.addChart(pres.charts.BAR, [
      { name: "Bloom p = 1 %", labels: ns.map((n) => fmt(n)), values: mb(bm) },
      { name: "HashSet exacto", labels: ns.map((n) => fmt(n)), values: mb(hm) },
    ], { x: 7.2, y: 1.5, w: 5.5, h: 5.1, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent2, HEX.accent6],
      showTitle: true, title: "Memoria de la lista negra (MB, escala log)", titleFontSize: 13, titleColor: HEX.dk1,
      valAxisLogScaleBase: 10, showLegend: true, legendPos: "b", legendFontSize: 11, catAxisTitle: "elementos",
      showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: "E4E3DF", size: 0.5 },
      catGridLine: { style: "none" }, ...chartText });
  }
  s.addNotes("Presenta: Clinton. El Bloom guarda la lista negra en pocos kilobytes y no tiene falsos negativos. Como un falso positivo bloquearía una compra legítima, cada positivo se verifica contra el conjunto exacto. El Count-Min cuenta intentos por tarjeta en una ventana deslizante de 10 minutos; nunca subestima, así que no oculta una ráfaga. El recall es 91 % porque los primeros intentos de cada ráfaga pasan antes de llegar al umbral. La precisión es 93 % porque en los datos reales hay unas 300 cuentas con hasta 110 carritos o compras en 10 minutos: parecen bots o revendedores, y en producción irían a revisión manual. Con datos sintéticos la precisión fue 99,998 %.");

  // 8. S7
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Etapas" });
  s.addText("S7 Similitud: candidatos en milisegundos sin comparar todo", { placeholder: "title" });
  badge(s, "S7", 0.6, 1.55, C.accent4);
  s.addText([
    { text: "MinHash + LSH: ", options: { bold: true, color: C.text1 } },
    { text: "cada producto es el conjunto de usuarios que lo compraron o lo pusieron en el carrito. 128 permutaciones en 32 bandas de 4 filas (umbral de Jaccard ≈ 0,42).", options: { color: C.text2, breakLine: true } },
    { text: "IVF sobre embeddings SVD: ", options: { bold: true, color: C.text1 } },
    { text: "32 dimensiones de la matriz usuario × producto; k-means en celdas y búsqueda solo en las nprobe más cercanas. HNSW implementado como referencia.", options: { color: C.text2 } },
  ], { x: 1.45, y: 1.5, w: 5.6, h: 2.8, fontSize: 15, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 12 });
  if (mhRow.n) {
    statBlock(s, 0.6, 4.2, 3.1, `${fmt(mhRow.speedup, 0)}x`, `LSH frente a fuerza bruta con ${fmt(mhRow.n / 1e6)}M conjuntos`, C.accent4);
    statBlock(s, 3.9, 4.2, 3.1, fmt(mhRow.recall_at10_J05, 2), "recall@10 de vecinos con Jaccard ≥ 0,5", C.accent4);
  }
  if (ivfTop.n) {
    statBlock(s, 0.6, 5.5, 3.1, `${fmt(annExact.query_ms / ivfTop.query_ms, 0)}x`, `IVF frente a exacto con ${fmt(ivfTop.n / 1e6)}M vectores`, C.text1);
    statBlock(s, 3.9, 5.5, 3.1, fmt(ivfTop.recall_at10, 2), "recall@10 del IVF", C.text1);
  }
  const mhRows = mh.filter((r) => r.bands === 32).sort((a, b) => a.n - b.n);
  if (mhRows.length) {
    s.addChart(pres.charts.BAR, [
      { name: "MinHash + LSH", labels: mhRows.map((r) => fmt(r.n)), values: mhRows.map((r) => +r.query_ms.toFixed(2)) },
      { name: "Fuerza bruta", labels: mhRows.map((r) => fmt(r.n)), values: mhRows.map((r) => +r.bruteforce_query_ms.toFixed(2)) },
    ], { x: 7.2, y: 1.5, w: 5.5, h: 5.1, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent4, HEX.accent2],
      showTitle: true, title: "Tiempo por consulta Top-10 (ms, escala log)", titleFontSize: 13, titleColor: HEX.dk1,
      valAxisLogScaleBase: 10, showLegend: true, legendPos: "b", legendFontSize: 11, catAxisTitle: "conjuntos indexados",
      showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: "E4E3DF", size: 0.5 },
      catGridLine: { style: "none" }, ...chartText });
  }
  s.addNotes("Presenta: Ronald. MinHash estima Jaccard entre conjuntos de compradores, y LSH por bandas evita comparar contra todos: una consulta son dos búsquedas binarias sobre un arreglo ordenado. La ganancia crece con n porque fuerza bruta es lineal. Para vectores usamos IVF sobre embeddings SVD; HNSW quedó como referencia por su costo de construcción en Python.");

  // 9. S8
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Etapas" });
  s.addText("S8 Grafo de co-compra: quién influye y qué se compra junto", { placeholder: "title" });
  badge(s, "S8", 0.6, 1.55, C.accent5);
  s.addText([
    { text: "PageRank ", options: { bold: true, color: C.text1 } },
    { text: "por iteración de potencias sobre CSR (α = 0,85), validado contra NetworkX. Premia estar conectado con productos centrales, no solo tener muchas co-compras.", options: { color: C.text2, breakLine: true } },
    { text: "Louvain frente a Label Propagation: ", options: { bold: true, color: C.text1 } },
    { text: "Louvain maximiza la modularidad; LPA es casi lineal pero sin función objetivo y menos estable entre semillas.", options: { color: C.text2 } },
  ], { x: 1.45, y: 1.5, w: 5.6, h: 2.8, fontSize: 15, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 12 });
  statBlock(s, 0.6, 4.2, 3.1, fmt(S.S8.nodes_with_edges), `productos y ${fmt(S.S8.edges)} aristas en la corrida de 10M`, C.accent5);
  statBlock(s, 3.9, 4.2, 3.1, `${fmt(S.S8.pagerank_s, 2)} s`, `PageRank, ${S.S8.pagerank_iters} iteraciones`, C.accent5);
  if (prBig.edges) {
    statBlock(s, 0.6, 5.5, 3.1, `${fmt(prBig.pagerank_s ?? prBig.time_s, 1)} s`, `PageRank con ${fmt(prBig.edges / 1e6, 1)}M aristas`, C.text1);
    statBlock(s, 3.9, 5.5, 3.1, fmt(prBig.spearman_vs_degree, 2), "correlación de Spearman con el grado", C.text1);
  }
  const comRows = [["REES46 real (10M eventos)", S.S8.louvain_Q, S.S8.lpa_Q]];
  if (louBig.edges) comRows.push([`Grafo plantado ${fmt(louBig.edges / 1e6, 1)}M aristas`, louBig.modularity, lpaSame.modularity]);
  s.addChart(pres.charts.BAR, [
    { name: "Louvain", labels: comRows.map((r) => r[0]), values: comRows.map((r) => +(+r[1]).toFixed(3)) },
    { name: "Label Propagation", labels: comRows.map((r) => r[0]), values: comRows.map((r) => +(+r[2]).toFixed(3)) },
  ], { x: 7.2, y: 1.5, w: 5.5, h: 5.1, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent5, HEX.accent6],
    showTitle: true, title: "Modularidad Q (más alto es mejor)", titleFontSize: 13, titleColor: HEX.dk1,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFormatCode: "0.000",
    valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "b", legendFontSize: 11,
    catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" }, ...chartText });
  s.addNotes("Presenta: Clinton. El grafo une productos que aparecen en la misma orden. PageRank converge en menos de un segundo. En los datos reales el grafo es disperso porque pocas sesiones compran más de un producto, y ahí la ventaja de Louvain sobre Label Propagation crece. En comunidades, Louvain obtiene mayor modularidad y es más estable entre semillas que Label Propagation, a cambio de más tiempo.");

  // 10. Escalabilidad
  pres.addSection({ title: "Resultados" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  s.addText("Escalabilidad: las pendientes medidas siguen la teoría", { placeholder: "title" });
  const fb = R("report", "figures", "fig_base_scaling.png");
  const ff = imageFit(fb, 7.6, 5.2);
  s.addImage({ path: fb, x: 0.6, y: 1.45, w: ff.w, h: ff.h, altText: "Tiempo frente a n en log-log para las estructuras base, de 1K a 10M" });
  const rows = [["Estructura", "Pend.", "Esperado"]].concat(slopes.filter((r) => BASE.includes(r.Estructura)).map((r) => [`${r.Estructura} ${r["Operación"]}`, fmt(r.Pendiente, 2), r.Esperado]));
  s.addTable(rows.map((r, i) => r.map((t) => ({ text: String(t), options: i === 0 ? { bold: true, color: C.text1, fill: { color: C.background2 } } : { color: C.text2 } }))),
    { x: 8.45, y: 1.45, w: 4.3, colW: [2.35, 0.65, 1.3], fontSize: 9.5, border: { type: "solid", pt: 0.5, color: "E4E3DF" },
      margin: 0.04, rowH: 0.3 });
  s.addText("1K a 10M, 100 repeticiones hasta 10K, 10 en 100K, 3 en 1M y 1 en 10M por tiempo de cómputo.",
    { x: 0.6, y: 6.6, w: 7.6, h: 0.35, fontSize: 11.5, italic: true, color: C.text2, margin: 0, isTextBox: true });
  s.addNotes("Presenta: Ronald. Medimos de mil a diez millones en escala log-log. Una pendiente cercana a 1 es lineal; merge sort y heap quedan algo arriba por el factor log n. La búsqueda en el B+Tree y el KNN del KD-Tree casi no crecen con n, mientras fuerza bruta es lineal. Las repeticiones bajan con n por tiempo de cómputo, y el informe lo justifica.");

  // 11. Demo
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  s.addText("Demo: el recorrido de una compra express por las etapas", { placeholder: "title" });
  const t = S.traza_evento;
  const steps = [t.evento].concat(t.pasos.map((p) => p.replace(/^\s+/, "")));
  s.addText(steps.map((p, i) => ({ text: p, options: { breakLine: i < steps.length - 1 } })), {
    x: 0.6, y: 1.5, w: 6.4, h: 5.1, fontSize: 11, fontFace: "Courier New", color: C.text1, margin: 0.18,
    fill: { color: C.background2 }, valign: "top", isTextBox: true, paraSpaceAfter: 5 });
  const shot = R("docs", "capturas", "panel_demo.png");
  if (fs.existsSync(shot)) {
    const f = imageFit(shot, 5.5, 5.1);
    s.addImage({ path: shot, x: 7.2, y: 1.5, w: f.w, h: f.h, altText: "Panel Streamlit con métricas y recorrido del evento" });
  }
  s.addNotes("Presenta: Sebastian. Demo en vivo de 3 a 4 minutos: python scripts/run_pipeline.py --data data/synthetic_200000 y luego el panel con streamlit run src/eda/service/dashboard.py. A la izquierda está la traza literal de una compra de la corrida de 10M: entra a la cola, pasa la limpieza, sale tercera del heap, se resuelve en la tabla hash, pasa el filtro, se indexa en el B+Tree y termina en recomendaciones que combinan grafo, IVF y KD-Tree.");

  // 12. Corrida de 10M
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  s.addText(`La corrida completa: ${fmt(S.eventos_recibidos / 1e6, 1)}M eventos reales en ${fmt(S.segundos / 60, 0)} minutos`, { placeholder: "title" });
  const st = Object.entries(S.tiempos_por_etapa_s).filter(([k]) => k !== "setup").sort((a, b) => a[1] - b[1]);
  s.addChart(pres.charts.BAR, [{ name: "segundos", labels: st.map((x) => x[0]), values: st.map((x) => +x[1].toFixed(1)) }], {
    x: 0.6, y: 1.45, w: 7.4, h: 5.3, barDir: "bar", barGapWidthPct: 40, chartColors: [HEX.accent1],
    showTitle: true, title: "Tiempo total por etapa (s)", titleFontSize: 13, titleColor: HEX.dk1, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFormatCode: "#,##0",
    catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10.5, valAxisLabelFontSize: 10,
    valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" }, ...chartText });
  statBlock(s, 8.4, 1.5, 4.3, `${fmt(S.eventos_por_segundo)} ev/s`, "de punta a punta en Python, 1 proceso", C.accent1);
  statBlock(s, 8.4, 2.95, 4.3, fmt(S.S4.ordenes_indexadas), `órdenes en el B+Tree en disco, altura ${S.S4.altura_indice_ordenes}`, C.text1);
  statBlock(s, 8.4, 4.4, 4.3, fmt(S.S6.blocked), "eventos de fraude bloqueados antes de llegar a S4", C.text1);
  s.addText("El refresco de S7 (SVD + k-means sobre la matriz usuario × producto) domina el tiempo: es el primer candidato a paralelizar.",
    { x: 8.4, y: 5.85, w: 4.3, h: 0.9, fontSize: 12.5, italic: true, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("Presenta: Ronald. Corrimos los 10 millones de eventos reales de punta a punta. El gráfico muestra dónde se va el tiempo: la ingesta y el refresco de similitud son lo más caro; las estructuras de consulta como el B+Tree o el KD-Tree cuestan segundos. Esto nos dice qué optimizar primero.");

  // 13. Conclusiones
  pres.addSection({ title: "Cierre" });
  s = pres.addSlide({ masterName: "CIERRE", sectionTitle: "Cierre" });
  s.addText("Conclusiones", { placeholder: "title" });
  const concl = [
    ["La integración funciona", "Un mismo evento atraviesa las 10 etapas y cada salida es la entrada de la siguiente, comprobado con una prueba automática."],
    ["Lo aproximado paga", "Bloom y Count-Min detectan fraude con kilobytes; LSH e IVF responden órdenes de magnitud más rápido con recall alto."],
    ["Las pendientes confirman la teoría", "Lineal donde se espera O(n), casi plano en B+Tree y KD-Tree, lineal en fuerza bruta."],
    ["Siguiente paso", "Procesar octubre y noviembre completos (110M eventos), calibrar el umbral por usuario y paralelizar S7."],
  ];
  concl.forEach(([h, d], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 2.55;
    badge(s, String(i + 1), x, y, [C.accent1, C.accent2, C.accent3, C.accent4][i], 0.55);
    s.addText(h, { x: x + 0.8, y: y - 0.05, w: 5.0, h: 0.65, fontSize: 19, bold: true, color: C.background1, margin: 0,
      valign: "middle", isTextBox: true, fontFace: THEME.headFontFace });
    s.addText(d, { x: x + 0.8, y: y + 0.65, w: 5.0, h: 1.4, fontSize: 14, color: C.accent6, margin: 0, valign: "top",
      isTextBox: true });
  });
  s.addText("github.com/dantebarreto-gif/Proyecto-Final-E-AD", { x: 0.6, y: 6.8, w: 8, h: 0.4, fontSize: 12,
    color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("Presenta: Clinton. Cerramos con cuatro ideas: la integración es observable de punta a punta, las estructuras probabilísticas ganan en memoria y tiempo con error controlado, las mediciones confirman la complejidad teórica, y el siguiente paso es procesar los dos meses completos y paralelizar el refresco de similitud. Gracias, quedamos atentos a sus preguntas.");

  await pres.writeFile({ fileName: OUT });
  // escribe los colores del tema en el .pptx (opcional: módulo apply_theme.js de la skill pptx)
  if (process.env.APPLY_THEME_JS) await require(process.env.APPLY_THEME_JS).applyTheme(OUT, THEME);
  console.log("diapositivas:", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
