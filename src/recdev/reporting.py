"""Tabelas e gráficos derivados dos mesmos relatórios JSON."""

import csv
from pathlib import Path

import numpy as np


def experiment_rows(experiment):
    for report in experiment["candidates"] + experiment["final_test"]:
        diagnostics = [fit["diagnostics"] for fit in report["fits"]]
        k = report["config"]["n_components"]
        yield dict(
            dataset=report["dataset"], partition=report["partition"], method=report["method"],
            requested_components=k, effective_components=diagnostics[0]["output_dimension"],
            whiten=report["config"]["whiten"], dtype=report["config"]["dtype"],
            accuracy=report["accuracy_mean"], hits=report["hits"], n_queries=report["n_queries"],
            retained_variance=float(np.mean([d["pca"]["retained_variance"] for d in diagnostics])) if k else "",
            reconstruction_error=(sum(f["diagnostics"]["query_reconstruction_error"] * f["n_queries"]
                                      for f in report["folds"]) / report["n_queries"]) if k else "",
            gallery_bytes=float(np.mean([d["gallery_bytes"] for d in diagnostics])),
            projection_bytes=float(np.mean([d["projection_bytes"] for d in diagnostics])),
            total_feature_bytes=float(np.mean([d["total_feature_bytes"] for d in diagnostics])),
            projection_ms=sum(f["timings"]["query_projection_ms"] for f in report["folds"]),
            search_ms=sum(f["timings"]["search_ms"] for f in report["folds"]),
            pca_fit_ms=sum(fit["timings"]["pca_fit_ms"] for fit in report["fits"]),
            svd_reused=any(fit["timings"]["svd_reused"] for fit in report["fits"]),
        )


def export_experiment(experiment, directory: Path, plots=False):
    rows = list(experiment_rows(experiment))
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if not plots:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage = "validation" if experiment["selections"] else "test"
    for metric, title in (("accuracy", "Acurácia"), ("retained_variance", "Variância retida"),
                          ("reconstruction_error", "Erro relativo de reconstrução"),
                          ("total_feature_bytes", "Galeria + projeção (bytes)"),
                          ("search_ms", "Busca (ms; lote de consultas)")):
        fig, ax = plt.subplots()
        for method in ("pca", "embedding-pca"):
            points = [row for row in rows if row["partition"] == stage and row["method"] == method]
            if points:
                ax.plot([p["effective_components"] for p in points], [p[metric] for p in points], "o-", label=method)
        if metric == "accuracy":
            for method in ("pixels", "embedding"):
                points = [r for r in rows if r["partition"] == stage and r["method"] == method]
                if points:
                    ax.axhline(points[0][metric], linestyle="--", label=method)
            ax.set_ylim(0, 1.05)
        ax.set(xlabel="Componentes k", ylabel=title, title=f"{experiment['dataset']} — {stage}")
        if ax.lines:
            ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / f"{metric}-vs-k.png", dpi=140)
        plt.close(fig)
    for method in ("pca", "embedding-pca"):
        reports = [r for r in experiment["candidates"] if r["method"] == method]
        if not reports:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for fit in reports[0]["fits"]:
            diagnostics = fit["diagnostics"]["pca"]
            singular = diagnostics["singular_values"]
            axes[0].semilogy(range(1, len(singular) + 1), np.maximum(singular, np.finfo(float).tiny), label=fit["split"])
            axes[1].plot(range(1, len(singular) + 1), np.cumsum(diagnostics["explained_variance_ratio"]), label=fit["split"])
        axes[0].set(xlabel="Índice", ylabel="Valor singular", title="Espectro da galeria")
        axes[1].set(xlabel="Componentes", ylabel="Variância acumulada", title="Energia da galeria")
        axes[0].legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(directory / f"spectrum-{method}.png", dpi=140)
        plt.close(fig)
