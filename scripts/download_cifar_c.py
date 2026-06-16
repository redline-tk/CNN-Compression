import argparse
import os
import tarfile
import urllib.request

DOWNLOADS = {
    "cifar10c":  {"url": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar",  "fname": "CIFAR-10-C.tar",  "subdir": "CIFAR-10-C"},
    "cifar100c": {"url": "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar", "fname": "CIFAR-100-C.tar", "subdir": "CIFAR-100-C"},
}


def _progress(count, block_size, total_size):
    print(f"\r  {min(count * block_size * 100 / total_size, 100):.1f}%", end="", flush=True)


def download_and_extract(key, data_dir):
    info     = DOWNLOADS[key]
    target   = os.path.join(data_dir, info["subdir"])
    if os.path.isdir(target):
        print(f"  {info['subdir']} already exists, skipping.")
        return
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, info["fname"])
    print(f"Downloading {info['fname']} ...")
    urllib.request.urlretrieve(info["url"], tar_path, reporthook=_progress)
    print()
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_dir)
    os.remove(tar_path)
    print(f"  Done -> {target}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--cifar100", action="store_true")
    args = parser.parse_args()
    download_and_extract("cifar10c", args.data_dir)
    if args.cifar100:
        download_and_extract("cifar100c", args.data_dir)


if __name__ == "__main__":
    main()
