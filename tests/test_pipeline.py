"""Prueba de integración: un stream pequeño atraviesa las 10 etapas."""
import os

from eda.data.rees46 import prepare
from eda.pipeline.pipeline import StreamPipeline


def test_pipeline_end_to_end(tmp_path):
    ev, cat = prepare("synthetic", 30_000, str(tmp_path / "d"))
    p = StreamPipeline(ev, cat, workdir=str(tmp_path / "w"), batch_size=2_000, refresh_every=10_000)
    s = p.run(verbose=False)
    p.close()
    assert s["eventos_recibidos"] > 30_000
    assert s["S1"]["duplicados_eliminados"] == s["S1"]["duplicados_reales_detectados"] > 0
    assert s["S6"]["blocked"] > 0 and s["S6"]["precision"] > 0.9
    assert s["S4"]["ordenes_indexadas"] > 0
    assert s["top_ventas"] and s["top_pagerank"] and s["alertas_prioritarias"]
    t = s["traza_evento"]
    stages = [step.split("]")[0] for step in t["pasos"]]
    assert stages == ["[S0", "[S1", "[S2", "[S3", "[S6", "[S4", "[S5", "[S7", "[S8"]
    assert t["recomendaciones"]
    assert os.path.exists(tmp_path / "w" / "price_index.bpt")
