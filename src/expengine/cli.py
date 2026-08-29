import argparse

from expengine import pipeline
from expengine.data import loader

DESCRIPTION = "Analysis engine for a large scale randomised advertising experiment"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expengine", description=DESCRIPTION)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cache", help="build the processed parquet cache from the raw file")
    commands.add_parser("verify", help="reproduce the published facts about the raw file")
    run_command = commands.add_parser("run", help="run every stage and write the artifacts")
    run_command.add_argument("--skip-figures", action="store_true", help="do not write figures")
    run_command.add_argument("--refit", action="store_true", help="refit the control variate")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "cache":
        pipeline.log(f"cache written to {loader.build_cache()}")
        return 0
    if arguments.command == "verify":
        loader.build_cache()
        table = loader.check_facts(loader.summarise(loader.load_analysis_frame()))
        print(table.to_string(index=False))
        failures = table.loc[~table["matches"], "fact"].tolist()
        if failures:
            print(f"mismatched facts: {failures}")
            return 1
        return 0
    pipeline.run(skip_figures=arguments.skip_figures, refit=arguments.refit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
