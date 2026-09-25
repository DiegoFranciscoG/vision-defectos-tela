"""Command line entry point: `fabric-inspection <command>`."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from fabric_inspection.config import get_settings


def _reproduce(args: argparse.Namespace) -> None:
    from fabric_inspection.ml.classifier import ClassifierConfig  # noqa: PLC0415 - heavy import
    from fabric_inspection.ml.pipeline import PipelineConfig, run_pipeline  # noqa: PLC0415

    config = PipelineConfig(
        seed=args.seed,
        classifier=ClassifierConfig(epochs=args.epochs, seed=args.seed),
        patchcore_backbones=tuple(args.backbones),
    )
    metrics = run_pipeline(config)
    served = metrics["serving"]["served_detector_test"]
    print(json.dumps({"selected": metrics["selected_backbone"], "served_test": served}, indent=2))


def _download(_: argparse.Namespace) -> None:
    from fabric_inspection.data.download import download_dataset  # noqa: PLC0415

    print(download_dataset(Path("data/raw")))


def _init_db(_: argparse.Namespace) -> None:
    from fabric_inspection.db.migrate import upgrade_database  # noqa: PLC0415

    upgrade_database(get_settings().database_url)
    print("Database migrated to head")


def _seed_demo(_: argparse.Namespace) -> None:
    from fabric_inspection.db.session import (  # noqa: PLC0415
        build_engine,
        build_session_factory,
        session_scope,
    )
    from fabric_inspection.registry import ModelRegistry  # noqa: PLC0415
    from fabric_inspection.service.demo_seed import seed_demo  # noqa: PLC0415

    settings = get_settings()
    registry = ModelRegistry.load(settings.model_registry)
    engine = build_engine(settings.database_url)
    with session_scope(build_session_factory(engine)) as session:
        summary = seed_demo(session, registry, settings)
    engine.dispose()
    print(summary or "Demo data already present")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "fabric_inspection.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
        server_header=False,
    )


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(prog="fabric-inspection")
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    reproduce = commands.add_parser("reproduce", help="download, train, evaluate and export")
    reproduce.add_argument("--seed", type=int, default=42)
    reproduce.add_argument("--epochs", type=int, default=12)
    reproduce.add_argument(
        "--backbones", nargs="+", default=["resnet18", "wide_resnet50_2"], metavar="NAME"
    )
    reproduce.set_defaults(handler=_reproduce)
    commands.add_parser("download", help="download and verify the dataset").set_defaults(
        handler=_download
    )
    commands.add_parser("init-db", help="apply database migrations").set_defaults(handler=_init_db)
    commands.add_parser("seed-demo", help="insert fictitious demo data").set_defaults(
        handler=_seed_demo
    )
    serve = commands.add_parser("serve", help="run the inference API")
    serve.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    serve.add_argument(
        "--forwarded-allow-ips",
        default=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        help="Proxies trusted for X-Forwarded-For ('*' only behind a managed proxy like Render)",
    )
    serve.set_defaults(handler=_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.handler(args)


if __name__ == "__main__":
    main()
