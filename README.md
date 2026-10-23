# Proyecto Final EDA 2026-2: pipeline de data streams para comercio electrónico

**UTEC · Maestría en Ciencia de Datos e IA · Estructuras de Datos y Algoritmos · Prof. Heider Sánchez**

Pipeline de 10 etapas que procesa un stream de eventos de una tienda en línea (escenario B) con estructuras de datos implementadas desde cero: cada etapa usa la estructura adecuada y su salida alimenta a la siguiente. Un evento entra por una cola FIFO, se limpia, se prioriza, se resuelve en tablas hash, pasa un filtro de fraude probabilístico y termina actualizando índices, similitud, el grafo de co-compra y el servicio de recomendaciones.

![Pipeline](docs/pipeline.png)

## Entregables

| Entregable | Dónde |
|---|---|
| Código fuente de las estructuras y el pipeline | [`src/eda/`](src/eda) |
| Informe PDF (8 a 12 páginas) | [`report/Proyecto_Final_Grupo2.pdf`](report/Proyecto_Final_Grupo2.pdf), fuente LaTeX en [`report/informe.tex`](report/informe.tex) |
| Diapositivas | [`report/Presentacion_Final.pptx`](report/Presentacion_Final.pptx) |
| Notebook de análisis | [`notebooks/analisis_resultados.ipynb`](notebooks/analisis_resultados.ipynb) |
| Diagrama del pipeline | [`docs/pipeline.svg`](docs/pipeline.svg) |
| Resultados de experimentos (CSV) y figuras | [`results/bench/`](results/bench), [`report/figures/`](report/figures) |
| Resumen de la corrida con 10M eventos | [`results/demo/summary_10M.json`](results/demo/summary_10M.json) |

## Instalación y uso

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt

# 1. Datos: REES46 real (ver abajo cómo descargarlo) o el generador con el mismo esquema
python scripts/prepare_data.py --source rees46 --events 10000000   # 10M eventos reales
python scripts/prepare_data.py --events 200000                     # sintético, ~3 s, sin descarga

# 2. Pipeline de punta a punta: imprime el recorrido de un evento por las 10 etapas
python scripts/run_pipeline.py --data data/synthetic_200000

# 3. Panel (etapa 9): demo en vivo, autocompletado, recomendaciones y rangos de precio
streamlit run src/eda/service/dashboard.py -- --data data/synthetic_200000

# 4. Pruebas (cada estructura contra su versión exacta) y experimentos
pytest
python bench/run_benchmarks.py --quick     # 1K a 100K, minutos
python bench/run_benchmarks.py             # 1K a 10M, horas en Python puro
python bench/plots.py                      # figuras log-log en report/figures/

# 5. Informe PDF (LaTeX, requiere pdflatex)
python scripts/build_report.py --group Grupo2

# 6. Diapositivas (Node.js): lee las cifras de results/
npm install pptxgenjs react react-dom react-icons sharp
node scripts/build_slides.js
```

### Usar el dataset real REES46

1. Descargar *eCommerce behavior data from multi category store* de Kaggle (4,6 GB comprimido) y dejar `2019-Oct.csv` en `data/raw/`:
   `python -c "import kagglehub; print(kagglehub.dataset_download('mkechinov/ecommerce-behavior-data-from-multi-category-store'))"`
   o `kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -p data/raw --unzip`
2. `python scripts/prepare_data.py --source rees46 --events 10000000`

DuckDB lee los CSV en streaming y toma los primeros 10M eventos en orden temporal; el catálogo se deriva de los mismos eventos. Encima se agrega el componente sintético que el dataset no trae (dispositivo, tarjeta, tipo de envío, premium, consultas, duplicados y ráfagas de fraude) con semilla fija, como recomienda el enunciado.

## Arquitectura

| Etapa | Estructura (archivo) | Qué hace | Qué entrega |
|---|---|---|---|
| S0 Ingesta | Cola FIFO enlazada ([`linked_queue.py`](src/eda/structures/linked_queue.py)) | Buffer con capacidad entre el lector y el consumidor por lotes | Lotes de eventos |
| S1 Preprocesamiento | Merge sort + búsqueda binaria ([`sorting.py`](src/eda/structures/sorting.py)) | Valida, normaliza, elimina duplicados contra una ventana de 500K ids y ordena por tiempo | Eventos limpios |
| S2 Prioridad | Heap binario ([`heap.py`](src/eda/structures/heap.py)) | Compras express, premium, estándar, programadas, carritos, vistas | Eventos en orden de urgencia |
| S3 Almacenamiento | Tabla hash, LRU, matriz dispersa ([`hash_table.py`](src/eda/structures/hash_table.py), [`lru_cache.py`](src/eda/structures/lru_cache.py), [`sparse_matrix.py`](src/eda/structures/sparse_matrix.py)) | Catálogo y usuarios por id, caché de fichas y recomendaciones, matriz usuario × producto | Índices densos e interacciones |
| S4 Índice en disco | B+Tree de páginas de 4 KB ([`bplus_tree.py`](src/eda/structures/bplus_tree.py)) | Índice de precio (carga masiva) e índice temporal de órdenes (inserciones) | Consultas por rango |
| S5 Multidimensional | KD-Tree + KNN exacto ([`kd_tree.py`](src/eda/structures/kd_tree.py)) | Vecinos por precio, popularidad y conversión | Productos parecidos por atributos |
| S6 Filtrado [FIJA] | Bloom Filter + Count-Min Sketch ([`bloom_filter.py`](src/eda/structures/bloom_filter.py), [`count_min_sketch.py`](src/eda/structures/count_min_sketch.py)) | Lista negra (verificada contra el conjunto exacto) e intentos por tarjeta/dispositivo en 10 min | Alertas y eventos legítimos |
| S7 Similitud [FIJA] | MinHash + LSH, IVF, HNSW ([`minhash_lsh.py`](src/eda/structures/minhash_lsh.py), [`ann.py`](src/eda/structures/ann.py)) | Productos con compradores parecidos; vecinos en embeddings SVD | Productos relacionados |
| S8 Grafo [FIJA] | PageRank, Louvain, Label Propagation ([`graph.py`](src/eda/structures/graph.py)) | Grafo de co-compra por orden; influencia y comunidades | Ranking y comunidades |
| S9 Servicio | Trie, Top-K ([`trie.py`](src/eda/structures/trie.py)), panel ([`dashboard.py`](src/eda/service/dashboard.py)) | Autocompletado, más vendidos, recomendación combinada, alertas | Respuesta al usuario |

El orquestador está en [`src/eda/pipeline/`](src/eda/pipeline). S6 corre justo después de S3 para que lo bloqueado no contamine índices, similitud ni grafo. Las librerías externas (NetworkX, SciPy) solo se usan en los tests para validar resultados.

## Resultados principales

Corrida de punta a punta con **10.223.169 eventos reales de REES46** (1 al 8 de octubre de 2019, más el componente de stream) en 4 vCPU con Python 3.13. El resumen está en [`results/demo/summary_10M.json`](results/demo/summary_10M.json) y el log en [`results/demo/run_10M_rees46.log`](results/demo/run_10M_rees46.log):

| Métrica | Valor |
|---|---|
| Datos | 120.565 productos, 1.072.341 usuarios; 95,6 % de los eventos son vistas |
| Tiempo total | 1.246 s (8.204 eventos/s); el refresco de S7 es el 55 % |
| Duplicados eliminados (S1) | 151.081 de 151.081 inyectados |
| Fraude (S6) | 70.682 bloqueados, precisión 92,9 %, recall 91,1 %; Bloom de 3,4 KB, 7 falsos positivos descartados |
| B+Tree en disco (S4) | 178.145 órdenes, altura 3 |
| Grafo de co-compra (S8) | 6.901 productos, 14.035 aristas; Louvain Q = 0,626 frente a LPA Q = 0,394 |

Experimentos de 1K a 10M (detalle y figuras en el informe):

| Comparación | Resultado |
|---|---|
| KD-Tree frente a KNN por fuerza bruta (10M puntos) | 0,73 ms frente a 393 ms por consulta (537x) |
| B+Tree, búsqueda con 10M claves | 44 µs, 3 páginas leídas |
| Bloom frente a tabla hash exacta (10M) | 12 MB frente a 268 MB; FP medido 1,01 % con p = 1 % |
| Count-Min frente a conteo exacto (10M) | 53 KB frente a 16,8 MB; error siempre bajo ε·N |
| MinHash + LSH frente a Jaccard exacto (1M conjuntos) | 0,76 ms frente a 225 ms (295x), recall@10 = 1,0 |
| IVF frente a KNN exacto (10M vectores, d = 32) | 0,46 ms frente a 104 ms (226x), recall@10 = 0,95 |

Las pendientes log-log medidas están en [`results/bench/slopes.csv`](results/bench/slopes.csv).

**Sobre los datos.** La precisión del filtro baja frente a la corrida sintética (99,998 %) porque en REES46 hay 309 cuentas reales con más de 12 eventos de carrito o compra en 10 minutos (hasta 110); el Count-Min las marca igual que a las ráfagas inyectadas y se reportan como anomalías de bots o revendedores. El grafo es pequeño porque pocas sesiones compran más de un producto. Los experimentos de escalabilidad usan entradas generadas para controlar n y conocer la respuesta exacta. La corrida sintética anterior se conserva en `results/demo/summary_10M_sintetico.json`.

## Roles y contribución

Cada integrante es responsable principal de al menos dos etapas o una familia de algoritmos, la expone en la presentación y responde por ella en la sustentación.

| Integrante | Responsable principal de | Familias del bloque avanzado |
|---|---|---|
| Sebastian Rodrigo Garcia Villacorta | S0 Ingesta, S1 Preprocesamiento, S2 Prioridad, S9 Servicio y panel | — |
| Dante Guillermo Barreto Daza | S3 Almacenamiento (hash, LRU, matriz dispersa), S4 B+Tree, S5 KD-Tree | — |
| Clinton Michael Chulluncuy Reynoso | S6 Filtrado, S8 Grafo | Filtrado probabilístico, ranking de influencia, comunidades |
| Ronald Ramiro Hilario Orihuela | S7 Similitud, experimentos | Similitud de conjuntos, ANN vectorial |

La base del código se generó con asistencia de Claude Code (ver el historial de commits). Cada integrante revisa, prueba y amplía las etapas a su cargo en commits propios, de modo que el historial del repositorio refleje la contribución real de cada uno.

## Estructura del repositorio

```text
src/eda/structures/   estructuras de datos (una por archivo)
src/eda/data/         generador sintético con esquema REES46 y cargador del dataset real
src/eda/pipeline/     etapas y orquestador
src/eda/service/      panel Streamlit
scripts/              preparar datos, correr el pipeline, generar el informe
bench/                experimentos y figuras
tests/                pruebas unitarias e integración
notebooks/            análisis de resultados
report/               informe, figuras y diapositivas
results/              CSV de experimentos y resumen de la corrida de 10M
```
