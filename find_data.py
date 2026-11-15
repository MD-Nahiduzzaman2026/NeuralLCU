import os
import shutil
import glob

target_name = "training_setA.csv"
found_file = None

print("Searching for your downloaded CSV file...")

# 1. Check current folder
csvs = glob.glob("*.csv")
if csvs:
    found_file = csvs[0]

# 2. Check Downloads folder
if not found_file:
    dl_path = os.path.join(os.path.expanduser("~"), "Downloads")
    for f in os.listdir(dl_path):
        if f.endswith(".csv") and ("sepsis" in f.lower() or "prediction" in f.lower()):
            found_file = os.path.join(dl_path, f)
            break

# 3. Check Kaggle's hidden cache folder
if not found_file:
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub")
    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            for f in files:
                if f.endswith(".csv"):
                    found_file = os.path.join(root, f)
                    break
            if found_file: break

if found_file:
    shutil.copy(found_file, target_name)
    print(f"\nSUCCESS! Found '{os.path.basename(found_file)}' and copied it.")
    print(f"It is now named '{target_name}' in your project folder.")
else:
    print("\nERROR: Could not find the file automatically.")
