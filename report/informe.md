# Estructuras de Datos Avanzadas Orquestadas para un Pipeline Eficiente de Procesamiento de Data Streams

<div class="meta">UTEC · Maestría en Ciencia de Datos e IA · Estructuras de Datos y Algoritmos 2026-2 · Prof. Heider Sánchez<br>
Integrantes: Sebastian Rodrigo Garcia Villacorta, Dante Guillermo Barreto Daza, Clinton Michael Chulluncuy Reynoso, Ronald Ramiro Hilario Orihuela<br>
Escenario B: comercio electrónico · Repositorio: github.com/dantebarreto-gif/Proyecto-Final-E-AD</div>

## Resumen

{{RESUMEN}}

## 1. Contexto y objetivo

Una tienda en línea recibe un flujo continuo de vistas, carritos y compras. Sobre ese flujo hay que priorizar órdenes urgentes, consultar productos con baja latencia, detectar fraude en segundos, recomendar productos relacionados y descubrir qué productos se compran juntos. Cada problema tiene una estructura de datos adecuada; el objetivo del proyecto es encadenarlas de modo que la salida de cada etapa sea la entrada real de la siguiente, y medir cuánto ganan las estructuras aproximadas frente a sus alternativas exactas.

**Datos.** Se usa el esquema del dataset REES46 *eCommerce behavior data from multi category store* (Kaggle; ~285M eventos, columnas `event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session`). El repositorio incluye el cargador del CSV real (DuckDB, lectura en streaming de los primeros N eventos ordenados por tiempo) y un generador con el mismo esquema y semilla fija que reproduce el embudo vista → carrito → compra, la popularidad Zipf de productos y sesiones centradas en grupos de categorías afines (lo que crea estructura de co-compra). Como pide el enunciado, el componente de stream es sintético en ambos casos: identificador de evento, dispositivo, IP, tarjeta, usuario premium, tipo de envío (express, estándar, programado), consultas de búsqueda, **1,5 % de duplicados** reenviados segundos después y **ráfagas de fraude** (25 a 120 intentos en menos de 5 minutos desde una tarjeta y un dispositivo). La mitad de las tarjetas fraudulentas se asume ya reportada (lista negra). Las corridas de este informe usan el generador porque el entorno de ejecución no tenía acceso a Kaggle; con los CSV en `data/raw/` el mismo comando corre sobre los datos reales.

## 2. Arquitectura del pipeline

{{ARQUITECTURA}}

## 3. Etapas 0 a 5: estructuras base

| Etapa | Estructura | Función en el pipeline | Complejidad | Alternativa descartada |
|---|---|---|---|---|
| 0 Ingesta | Cola FIFO con lista enlazada y capacidad máxima | Buffer entre el lector del stream y el consumidor por lotes; descarta y cuenta si se llena (backpressure) | enqueue/dequeue O(1) | Lista de Python: `pop(0)` es O(n) |
| 1 Preprocesamiento | Merge sort estable + búsqueda binaria vectorizada | Valida y normaliza; ordena el lote por id para eliminar duplicados adyacentes y busca cada id en la ventana ordenada de los últimos 500K ids; reordena por tiempo | O(b log b) por lote; O(b log W) contra la ventana | Conjunto de todos los ids: memoria sin límite |
| 2 Prioridad | Heap binario (min-heap) con desempate por llegada | Procesa primero compras express, luego premium, estándar, programadas, carritos y vistas; también ordena las alertas de fraude por severidad | push/pop O(log n), heapify O(n) | Lista ordenada: inserción O(n) |
| 3 Almacenamiento | Tabla hash con direccionamiento abierto (sondeo lineal, carga ≤ 0,7, lápidas), caché LRU (hash + lista doble), matriz dispersa DOK → CSR | Resuelve `product_id` y `user_id` a índices densos; cachea fichas y recomendaciones; acumula la matriz usuario × producto (vista 1, carrito 3, compra 5) | O(1) amortizado; CSR con nnz valores | Matriz densa: 200K × 200K floats no cabe |
| 4 Índice en disco | B+Tree con páginas de 4 KB (255 claves por hoja), hojas encadenadas y buffer pool LRU con write-back | Índice de precio del catálogo (carga masiva ordenada) e índice temporal de órdenes legítimas (inserciones); consultas por rango | búsqueda O(log_B n) páginas; rango O(log_B n + k/B) | Árbol binario: altura log₂ n y una página por nodo |
| 5 Multidimensional | KD-Tree (corte por mediana en el eje de mayor extensión, hojas de 16) y KNN exacto | Vecinos por atributos (log precio, log vistas, tasa de conversión) y consultas por caja | construcción O(n log n); KNN ≈ O(log n) en d bajo | Fuerza bruta O(n) por consulta (línea base) |

La clave del índice de precio combina precio en centavos e índice de producto (`precio·100 << 24 | id`), única y ordenada por precio. Con 10M claves el B+Tree tiene altura 3, así que una búsqueda lee como máximo 3 páginas.

## 4. Bloque de investigación avanzada (etapas 6 a 8)

### 4.1 Filtrado probabilístico: Bloom Filter y Count-Min Sketch

**Bloom Filter** (Bloom, 1970). Arreglo de *m* bits y *k* hashes: insertar pone *k* bits en 1; consultar responde "posiblemente presente" si los *k* están en 1. No hay falsos negativos y la tasa de falsos positivos es p ≈ (1 − e^(−kn/m))^k, mínima con m = −n ln p / (ln 2)² y k = (m/n) ln 2: para p = 1 % son 9,6 bits por elemento y k = 7, sin importar el tamaño de la clave. Las *k* posiciones se obtienen con doble hashing g_i = h₁ + i·h₂ mod m (Kirsch y Mitzenmacher, 2006) sobre splitmix64, y los bits van empaquetados en un arreglo `uint8`. Tiempo O(k) por operación. **Uso:** lista negra de tarjetas y dispositivos; como un positivo puede ser falso y bloquear una compra es una decisión crítica, todo positivo se verifica contra el conjunto exacto antes de bloquear. Límite: no admite borrado (sí con contadores, Counting Bloom) y la tasa de FP crece si se supera la capacidad.

**Count-Min Sketch** (Cormode y Muthukrishnan, 2005). Matriz de d × w contadores con un hash por fila; la estimación es el mínimo de las d celdas. Nunca subestima y, con w = ⌈e/ε⌉ y d = ⌈ln(1/δ)⌉, cumple f̂ ≤ f + εN con probabilidad 1 − δ. Memoria O(w·d) independiente del número de claves. **Uso:** clave = tarjeta y, por separado, dispositivo; **ventana** = 10 minutos en anillo de 10 sub-sketches de 1 minuto (al avanzar se limpia el más viejo); **umbral** = más de 12 intentos de compra o carrito por tarjeta, o 20 por dispositivo, en la ventana. La sobreestimación solo puede adelantar una alerta, nunca ocultarla.

### 4.2 Similitud de conjuntos: MinHash + LSH

Para una permutación aleatoria h, P[min h(A) = min h(B)] = J(A, B) (Broder, 1997): la fracción de coincidencias entre dos firmas de P minhashes estima Jaccard con error estándar √(J(1−J)/P). Con LSH por bandas (Indyk y Motwani, 1998) la firma se corta en b bandas de r filas y dos conjuntos son candidatos si coinciden en una banda completa: P[candidato] = 1 − (1 − J^r)^b, una curva en S con umbral ≈ (1/b)^(1/r). Se eligió MinHash (y no SimHash) porque el objeto natural es un **conjunto**: los usuarios que pusieron cada producto en el carrito o lo compraron. Con P = 128 y b = 32, r = 4, el umbral es 0,42. Las 32 bandas se indexan en **un solo arreglo ordenado** de pares (hash de banda mezclado con el número de banda, id), y una consulta son dos búsquedas binarias vectorizadas (la estructura de la etapa 1). Construcción O(P·nnz); consulta O(b log(nb) + candidatos).

### 4.3 ANN vectorial: IVF (y HNSW como referencia)

Los productos se representan con embeddings de 32 dimensiones obtenidos por **SVD truncada aleatorizada** (Halko et al., 2011) de la matriz usuario × producto en CSR, implementada solo con productos matriz-vector. **IVF** (índice de archivo invertido, Jégou et al., 2011): k-means (k-means++ y Lloyd) divide el espacio en nlist = 4√n celdas (máx. 4096); cada vector se guarda en la lista de su centroide y una consulta recorre solo las nprobe listas más cercanas: O(nlist·d + nprobe·(n/nlist)·d). **HNSW** (Malkov y Yashunin, 2020): grafo navegable en capas con nivel geométrico por nodo, descenso greedy en las capas altas y búsqueda best-first con ef candidatos en la capa 0, con la heurística de vecinos diversos del paper; O(log n) esperado. Se eligió IVF para el pipeline porque su construcción y consulta son operaciones de matriz (escala a 10M en Python), y HNSW se implementó para comparar el compromiso recall/latencia. La línea base es KNN exacto por fuerza bruta (O(n·d)); el KD-Tree de la etapa 5 sirve de contraste: en d = 32 deja de podar.

### 4.4 Ranking de influencia: PageRank

Grafo de co-compra no dirigido y ponderado: nodos = productos, peso(u, v) = número de órdenes (compras de una misma sesión) que contienen a ambos. PageRank (Brin y Page, 1998) es el vector estacionario de una caminata aleatoria que sigue aristas con probabilidad proporcional al peso y salta a un nodo aleatorio con probabilidad 1 − α (α = 0,85): r ← α(Pᵀr + masa colgante·v) + (1 − α)v. Se implementa con iteración de potencias sobre la CSR (O(m) por iteración, converge en norma L1 < 10⁻⁹) y se valida contra NetworkX. La versión personalizada (v concentrado en los productos del usuario) da recomendaciones locales. Línea base: grado ponderado. La diferencia es que PageRank premia estar conectado con productos a su vez centrales, no solo tener muchas co-compras.

### 4.5 Comunidades: Louvain frente a Label Propagation

**Louvain** (Blondel et al., 2008) maximiza la modularidad Q = Σ_c [L_c/2m − (d_c/2m)²] en dos fases repetidas: mover cada nodo a la comunidad vecina con mayor ganancia ΔQ ∝ k_i,c − Σ_tot·k_i/2m, y luego colapsar cada comunidad en un nodo. Casi O(m log n). **Label Propagation** (Raghavan et al., 2007): cada nodo adopta la etiqueta de mayor peso entre sus vecinos hasta que no hay cambios; casi lineal pero sin función objetivo, por lo que es menos estable. La estabilidad se mide con la información mutua normalizada (NMI) entre corridas con distinta semilla. Leiden (Traag et al., 2019) corrige comunidades mal conectadas de Louvain; queda como trabajo futuro.

## 5. Diseño de la integración

{{INTEGRACION}}

## 6. Resultados experimentales

{{RESULTADOS}}

## 7. Conclusiones

{{CONCLUSIONES}}

## Referencias

<div class="refs">

Bloom, B. H. (1970). Space/time trade-offs in hash coding with allowable errors. *Communications of the ACM*, 13(7), 422–426. https://doi.org/10.1145/362686.362692

Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.*, P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search engine. *Computer Networks and ISDN Systems*, 30(1–7), 107–117. https://doi.org/10.1016/S0169-7552(98)00110-X

Broder, A. Z. (1997). On the resemblance and containment of documents. *Compression and Complexity of Sequences*, 21–29. https://doi.org/10.1109/SEQUEN.1997.666900

Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: The Count-Min sketch and its applications. *Journal of Algorithms*, 55(1), 58–75. https://doi.org/10.1016/j.jalgor.2003.12.001

Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288.

Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. *STOC '98*, 604–613. https://doi.org/10.1145/276698.276876

Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. *IEEE TPAMI*, 33(1), 117–128. https://doi.org/10.1109/TPAMI.2010.57

Kirsch, A., & Mitzenmacher, M. (2006). Less hashing, same performance: Building a better Bloom filter. *ESA 2006*, 456–467.

Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE TPAMI*, 42(4), 824–836. https://doi.org/10.1109/TPAMI.2018.2889473

Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect community structures in large-scale networks. *Physical Review E*, 76(3), 036106. https://doi.org/10.1103/PhysRevE.76.036106

Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: Guaranteeing well-connected communities. *Scientific Reports*, 9, 5233. https://doi.org/10.1038/s41598-019-41695-z

Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. · Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets* (3rd ed.). Cambridge University Press. · Samet, H. (2006). *Foundations of Multidimensional and Metric Data Structures*. Morgan Kaufmann.

</div>
