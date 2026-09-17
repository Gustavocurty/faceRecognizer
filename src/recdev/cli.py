"""RecDev: audit, evaluate, experiment/compare, enroll e identify."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import environment, new_run, write_json
from .config import DATASETS, METHODS, MethodConfig
from .evaluation import calibrate_threshold_from_folds, evaluate, fit_split
from .experiments import DEFAULT_METHODS, run_experiment
from .features import FeatureStore
from .manifest import audit, build_manifest, check_subset
from .pipeline import RecognitionModel
from .preprocessing import load_pixels
from .protocols import PROTOCOL_VERSION, make_splits
from .reporting import export_experiment

ROOT = Path(__file__).resolve().parents[2]


def config_from(args):
    return MethodConfig(args.method or "embedding", args.components, args.whiten, args.dtype,
                        getattr(args, "threshold", None))


def feature_store(args, manifest, embeddings):
    return FeatureStore(args.root, manifest, "embedding" if embeddings else "pixels",
                        args.device, args.batch_size)


def run_metadata(args, manifest):
    return dict(schema=2, command=args.command, dataset=manifest.dataset,
                manifest_fingerprint=manifest.fingerprint, protocol=PROTOCOL_VERSION,
                environment=environment(args.root), threads=args.threads,
                splits=[split.summary() for split in make_splits(manifest)])


def print_report(report):
    print(f"{report['dataset']} / {report['method']} / {report['partition']}: "
          f"{report['hits']}/{report['n_queries']} = {report['accuracy_mean']:.2%}")
    for condition, values in report["per_condition"].items():
        print(f"  {condition}: {values['hits']}/{values['n_queries']} = {values['accuracy']:.2%}")


def cmd_audit(args):
    import json

    manifest = build_manifest(args.dataset, args.root)
    result = audit(manifest)
    result["protocol"] = PROTOCOL_VERSION
    result["splits"] = [split.summary() for split in make_splits(manifest)]
    other = {"very-easy": "easy", "easy": "very-easy", "medium": "hard"}[args.dataset]
    if (args.root / other).is_dir():
        result["overlap"] = {"dataset": other, "byte_identical_images": len(check_subset(manifest, build_manifest(other, args.root)))}
    path = args.root / "artifacts" / "manifests" / f"{args.dataset}.csv"
    manifest.to_csv(path)
    write_json(path.with_suffix(".json"), result)
    print(json.dumps({key: value for key, value in result.items() if key != "splits"}, indent=2, ensure_ascii=False))
    print(f"Manifesto e auditoria: {path}")
    return 0


def cmd_evaluate(args):
    manifest = build_manifest(args.dataset, args.root)
    config = config_from(args)
    if args.reject and (not config.uses_embeddings or args.dataset != "medium" or args.partition != "test"):
        raise ValueError("--reject requer embeddings, medium e --partition test (calibração na validação)")
    if args.reject and args.threshold is not None:
        raise ValueError("Use --reject ou --threshold, não ambos")
    store = feature_store(args, manifest, config.uses_embeddings)
    metadata = run_metadata(args, manifest)
    if args.reject:
        validation = evaluate(manifest, store, config, "validation")
        threshold = calibrate_threshold_from_folds(validation.folds, args.quantile)
        config = replace(config, threshold=threshold)
        metadata["rejection"] = dict(threshold=threshold, quantile=args.quantile,
                                     calibration_partition="validation",
                                     note="Somente conhecidos: não mede aceitação de desconhecidos.",
                                     validation=validation.summary())
    report = evaluate(manifest, store, config, args.partition)
    metadata.update(report=report.summary(), feature_extraction=store.summary())
    directory = new_run(args.root, args.dataset, f"evaluate-{args.partition}-{config.slug}")
    write_json(directory / "report.json", metadata)
    print_report(report.summary())
    print(f"Relatório: {directory / 'report.json'}")
    return 0


def cmd_experiment(args):
    if args.plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise ValueError('Gráficos requerem: pip install -e ".[plots]"') from error
    manifest = build_manifest(args.dataset, args.root)
    stores = {}
    for method in args.methods:
        space = "embedding" if method.startswith("embedding") else "pixels"
        if space not in stores:
            stores[space] = feature_store(args, manifest, space == "embedding")
    experiment = run_experiment(manifest, stores, args.components, args.methods, args.whiten, args.dtype)
    metadata = run_metadata(args, manifest)
    metadata["experiment"] = experiment
    directory = new_run(args.root, args.dataset, "experiment")
    write_json(directory / "report.json", metadata)
    export_experiment(experiment, directory, args.plots)
    if experiment["selections"]:
        for chosen in experiment["selections"]:
            print(f"Validação: {chosen['method']}, k={chosen['config']['n_components']}, "
                  f"acurácia={chosen['validation_accuracy']:.2%}")
    for report in experiment["final_test"] or experiment["candidates"]:
        if report["config"]["n_components"]:
            print(f"k={report['config']['n_components']}")
        print_report(report)
    for skipped in experiment["skipped"]:
        print(f"Configuração ignorada: {skipped['reason']}")
    print(f"JSON, CSV e gráficos solicitados: {directory}")
    return 0


def create_gallery(args, dataset):
    manifest = build_manifest(dataset, args.root)
    split = make_splits(manifest)[0]
    config = config_from(args)
    store = feature_store(args, manifest, config.uses_embeddings)
    model = fit_split(split, store, config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = getattr(args, "output", None) or args.root / "artifacts" / "galleries" / dataset / f"{stamp}-{config.slug}.npz"
    model.save(path, dataset, {r.identity_index: r.identity_original for r in split.gallery}, split.gallery)
    print(f"Galeria salva: {path}")
    return RecognitionModel.load(path)


def cmd_enroll(args):
    create_gallery(args, args.dataset)
    return 0


def cmd_identify(args):
    if args.model:
        if args.method or args.components is not None or args.whiten:
            raise ValueError("--model já contém método, componentes e whitening")
        model = RecognitionModel.load(args.model)
        if args.threshold is not None:
            model.config = replace(model.config, threshold=args.threshold)
            model.classifier.threshold = args.threshold
    else:
        model = create_gallery(args, args.gallery)
    if args.detect and not model.config.uses_embeddings:
        raise ValueError("--detect requer método embedding")
    if model.config.uses_embeddings:
        from .embeddings import FaceEmbedder

        features = FaceEmbedder(device=args.device, batch_size=args.batch_size).embed_image(args.image, args.detect)
    else:
        features = load_pixels(args.image)
    prediction = model.predict(features[None, :])[0]
    identities = model.metadata["identity_map"]
    print(f"Identidade: {identities.get(str(prediction.label), 'desconhecida')}")
    print(f"Score: {prediction.similarity:.6f}")
    print(f"Melhor candidato: {identities[str(prediction.ranking[0])]}")
    print(f"Segundo colocado: {identities[str(prediction.runner_up)]}")
    print(f"Margem: {prediction.similarity - prediction.runner_up_similarity:.6f}")
    return 0


def _method_args(parser, default="embedding"):
    parser.add_argument("--method", choices=METHODS, default=default)
    parser.add_argument("--components", type=int, help="k; omitido = posto numérico completo")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threshold", type=float, help="limiar manual de cosseno [-1, 1]")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="recdev", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=ROOT, help="pasta que contém os datasets")
    common.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--threads", type=int, default=4, help="limite de threads BLAS/PyTorch")
    sub = parser.add_subparsers(dest="command", required=True)

    audit_parser = sub.add_parser("audit", parents=[common], help="manifesto, hashes e splits")
    audit_parser.add_argument("--dataset", required=True, choices=DATASETS)
    audit_parser.set_defaults(func=cmd_audit)

    eval_parser = sub.add_parser("evaluate", parents=[common], help="avalia uma configuração fixa")
    eval_parser.add_argument("--dataset", required=True, choices=DATASETS)
    _method_args(eval_parser)
    eval_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    eval_parser.add_argument("--reject", action="store_true", help="calibra somente na validação do medium")
    eval_parser.add_argument("--quantile", type=float, default=0.05)
    eval_parser.set_defaults(func=cmd_evaluate)

    experiment_parser = sub.add_parser("experiment", aliases=["compare"], parents=[common], help="varredura de k + comparação")
    experiment_parser.add_argument("--dataset", required=True, choices=DATASETS)
    experiment_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    experiment_parser.add_argument("--components", nargs="+", type=int)
    experiment_parser.add_argument("--whiten", action="store_true")
    experiment_parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    experiment_parser.add_argument("--plots", action="store_true")
    experiment_parser.set_defaults(func=cmd_experiment)

    enroll_parser = sub.add_parser("enroll", parents=[common], help="salva a galeria do primeiro split")
    enroll_parser.add_argument("--dataset", required=True, choices=DATASETS)
    enroll_parser.add_argument("--output", type=Path)
    _method_args(enroll_parser)
    enroll_parser.set_defaults(func=cmd_enroll)

    identify_parser = sub.add_parser("identify", parents=[common], help="identifica usando galeria persistida")
    gallery = identify_parser.add_mutually_exclusive_group(required=True)
    gallery.add_argument("--model", type=Path, help="galeria .npz produzida por enroll")
    gallery.add_argument("--gallery", choices=DATASETS, help="cria e salva galeria antes de identificar")
    identify_parser.add_argument("--image", required=True, type=Path)
    identify_parser.add_argument("--detect", action="store_true", help="MTCNN: detecta e recorta o maior rosto")
    _method_args(identify_parser, default=None)
    identify_parser.set_defaults(func=cmd_identify)

    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads e --batch-size devem ser positivos")
    args.root = args.root.resolve()
    if not args.root.is_dir():
        parser.error(f"Pasta raiz inexistente: {args.root}")
    try:
        from threadpoolctl import threadpool_limits

        # Evita sobresubscrição de BLAS em matrizes pequenas e inferência CPU.
        if args.command not in ("audit",):
            import torch

            torch.set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            return args.func(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
