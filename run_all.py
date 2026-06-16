import argparse
import os
import subprocess
import sys


def run(cmd, label):
    print(f"\n{'#'*60}\n  {label}\n{'#'*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Failed: {label}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
    parser.add_argument("--arch",    default=None)
    parser.add_argument("--phase",   default="all",
                        choices=["all","data","baseline","phase1","phase2","phase3","evaluate","plot"])
    args = parser.parse_args()

    py   = sys.executable
    base = ["--config", args.config, "--dataset", args.dataset]
    if args.arch:
        base += ["--arch", args.arch]

    if args.phase in ("all", "data"):
        dl = [py, "scripts/download_cifar_c.py", "--data_dir", "./data"]
        if args.dataset == "cifar100":
            dl += ["--cifar100"]
        run(dl, "Downloading CIFAR-*-C")

    for phase in ("baseline", "phase1", "phase2", "phase3"):
        if args.phase in ("all", phase):
            run([py, "train.py"] + base + ["--phase", phase], f"Training: {phase}")

    if args.phase in ("all", "evaluate"):
        run([py, "evaluate.py"] + base, "Evaluating all models")

    if args.phase in ("all", "plot"):
        run([py, "results/plot_results.py"] + base[:4], "Plotting results")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
