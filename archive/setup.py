import importlib
import sys

def check_and_install_libraries():
    # List of required libraries with their package names
    requirements = {
        "os": "os",
        "sys": "sys",
        "datetime": "datetime",
        "logging": "logging",
        "subprocess": "subprocess",
        "json": "json",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "torch": "torch",
        "torchvision": "torchvision",
        "pytorch_msssim": "pytorch-msssim",
        "sklearn": "scikit-learn"
    }

    # Custom libraries
    custom_modules = [
        "libGalaxyZooDataset",
        "libDDPM",
        "libGPU_torch_utils",
        "libAutoEncoder"
    ]

    missing_packages = []

    # Check Python packages
    for module, package in requirements.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_packages.append(package)

    # Check custom libraries
    for module in custom_modules:
        if not importlib.util.find_spec(module):
            print(f"Custom library '{module}' is not found. Ensure it's in the project directory or PYTHONPATH.")
    
    # Suggest installation
    if missing_packages:
        print("\nMissing packages:")
        print("\n".join(missing_packages))
        print("\nYou can install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("All libraries are installed and ready!")

# Run the check
check_and_install_libraries()
