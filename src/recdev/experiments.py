"""Comparação pareada e seleção de k exclusivamente na validação do medium."""

from .config import MethodConfig
from .evaluation import evaluate
from .protocols import make_splits

DEFAULT_METHODS = ("pixels", "pca", "embedding", "embedding-pca")


def run_experiment(manifest, stores, components=None, methods=DEFAULT_METHODS,
                   whiten=False, dtype="float64"):
    splits = make_splits(manifest)
    has_validation = all(split.validation for split in splits)
    components = sorted(set(components or ([8, 16, 32, 64] if has_validation else [1, 2, 3, 4])))
    if not components or any(k < 1 for k in components):
        raise ValueError("Componentes devem ser inteiros positivos")
    candidates, selections, final, skipped = [], [], [], []
    # Nesta primeira fase, medium acessa somente galeria e validação.
    for method in methods:
        is_pca = method in ("pca", "embedding-pca")
        configurations = [MethodConfig(method, k if is_pca else None,
                                       whiten if is_pca else False, dtype)
                          for k in (components if is_pca else [None])]
        valid = []
        store = stores["embedding" if method.startswith("embedding") else "pixels"]
        for config in configurations:
            try:
                report = evaluate(manifest, store, config, "validation" if has_validation else "test", splits)
            except ValueError as error:
                if "excede o posto numérico" not in str(error):
                    raise
                skipped.append(dict(config=config.to_dict(), reason=str(error)))
                continue
            candidates.append(report.summary())
            valid.append((config, report.accuracy))
        if not valid:
            raise ValueError(f"Nenhuma configuração válida para {method}; reduza os componentes")
        if has_validation:
            # Desempate determinístico favorece a menor dimensão; teste não participa.
            chosen, accuracy = min(valid, key=lambda item: (-item[1], item[0].n_components or 0))
            selections.append(dict(method=method, config=chosen.to_dict(), validation_accuracy=accuracy,
                                   rule="max validation accuracy; ties: smallest k"))
    # Somente depois de TODAS as escolhas, avalia as configurações fixadas no teste.
    if has_validation:
        for selection in selections:
            config = MethodConfig(**selection["config"])
            store = stores["embedding" if config.uses_embeddings else "pixels"]
            final.append(evaluate(manifest, store, config, "test", splits).summary())
    return dict(
        dataset=manifest.dataset,
        mode="validation-selection-heldout-test" if has_validation else "descriptive-fixed-grid",
        note=("Teste reservado; galeria não é reajustada com a validação." if has_validation else
              "Curvas descritivas; não há validação independente nem seleção automática de vencedor."),
        candidates=candidates, selections=selections, final_test=final, skipped=skipped,
        feature_extraction={key: store.summary() for key, store in stores.items()},
    )
