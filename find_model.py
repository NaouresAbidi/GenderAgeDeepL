#!/usr/bin/env python3
"""Find available model files"""

import os
import glob

print("🔍 Looking for model files...")
print("=" * 40)

# Look for .h5 files in current directory
h5_files = glob.glob("*.h5")
if h5_files:
    print("✅ Found .h5 files:")
    for file in h5_files:
        size = os.path.getsize(file) / (1024*1024)  # MB
        print(f"  📁 {file} ({size:.1f} MB)")
else:
    print("❌ No .h5 files found in current directory")

# Look in subdirectories
print("\n🔍 Looking in subdirectories...")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".h5"):
            full_path = os.path.join(root, file)
            size = os.path.getsize(full_path) / (1024*1024)  # MB
            print(f"  📁 {full_path} ({size:.1f} MB)")

print("\n💡 To use a specific model, set the MODEL_PATH environment variable:")
print("   export MODEL_PATH='your_model_file.h5'")
print("   # or on Windows:")
print("   set MODEL_PATH=your_model_file.h5")