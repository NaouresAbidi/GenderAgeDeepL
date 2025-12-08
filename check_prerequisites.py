#!/usr/bin/env python3
"""
Prerequisites checker for Age & Gender Prediction project
"""
import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.10+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name} {version} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False

def main():
    print("🔍 Checking Prerequisites for Age & Gender Prediction Project\n")
    
    all_good = True
    
    # Check Python
    all_good &= check_python_version()
    
    # Check required packages
    packages = [
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('flask', 'flask'),
        ('pillow', 'PIL'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas')
    ]
    
    print("\n📦 Checking Python Packages:")
    for pkg_name, import_name in packages:
        all_good &= check_package(pkg_name, import_name)
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 All prerequisites met! You can run the project.")
    else:
        print("⚠️  Some prerequisites missing. Install them first.")
        print("\nTo install missing packages:")
        print("pip install tensorflow numpy opencv-python flask pillow matplotlib pandas")

if __name__ == "__main__":
    main()