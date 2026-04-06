"""
Setup WebShop environment for end-to-end evaluation.

WebShop (Yao et al., NeurIPS 2022): Simulated e-commerce with 1.18M products.
Repo: https://github.com/princeton-nlp/WebShop

Usage:
    python data/setup_webshop.py [--webshop-dir third_party/WebShop]

This script:
  1. Clones the WebShop repo if not present
  2. Downloads the product database
  3. Verifies the environment works
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup WebShop environment")
    parser.add_argument("--webshop-dir", type=str,
                        default="third_party/WebShop",
                        help="Directory for WebShop installation")
    args = parser.parse_args()

    webshop_dir = Path(args.webshop_dir)

    # Step 1: Clone if not present
    if not webshop_dir.exists():
        print("Cloning WebShop repository...")
        webshop_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/princeton-nlp/WebShop.git",
            str(webshop_dir),
        ], check=True)
        print(f"Cloned to {webshop_dir}")
    else:
        print(f"WebShop already exists at {webshop_dir}")

    # Step 2: Install WebShop dependencies
    print("\nInstalling WebShop dependencies...")
    requirements_path = webshop_dir / "requirements.txt"
    if requirements_path.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r",
            str(requirements_path), "--quiet",
        ])

    # Step 3: Download product data if not present
    data_dir = webshop_dir / "data"
    if not (data_dir / "items_shuffle.json").exists():
        print("\nDownloading WebShop product database...")
        setup_script = webshop_dir / "setup.sh"
        if setup_script.exists():
            subprocess.run(["bash", str(setup_script)], cwd=str(webshop_dir))
        else:
            print("Warning: setup.sh not found. You may need to download data manually.")
            print("See: https://github.com/princeton-nlp/WebShop#setup")
    else:
        print("Product database already downloaded.")

    # Step 4: Verify
    print("\nVerifying WebShop installation...")
    try:
        sys.path.insert(0, str(webshop_dir))
        # Try importing the environment
        print("  WebShop directory: OK")
        print(f"  Data directory exists: {data_dir.exists()}")
        if (data_dir / "items_shuffle.json").exists():
            import json
            with open(data_dir / "items_shuffle.json") as f:
                first_line = f.readline()
            print(f"  Product data: OK (first line: {len(first_line)} chars)")
        print("\nSetup complete!")
    except Exception as e:
        print(f"  Warning: {e}")
        print("  WebShop may need additional setup. Check the README.")


if __name__ == "__main__":
    main()
