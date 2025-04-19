import os
import sys
import subprocess
import importlib.util

# Check if required packages are installed
required_packages = ['requests', 'urllib3']

def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

def install_requirements():
    for package in required_packages:
        if not is_package_installed(package):
            print(f"Installing {package}...")
            install_package(package)
            print(f"{package} installed successfully.")
        else:
            print(f"{package} is already installed.")

if __name__ == "__main__":
    install_requirements()
