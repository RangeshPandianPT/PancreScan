import os
import hashlib
from tqdm import tqdm
from collections import defaultdict

def get_file_hash(filepath):
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def scan_directory(directory):
    """Scans a directory and returns a dictionary of {hash: [filepaths]}."""
    file_hashes = defaultdict(list)
    print(f"Scanning {directory}...")
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                file_hash = get_file_hash(filepath)
                file_hashes[file_hash].append(filepath)
    return file_hashes

def main():
    TRAIN_DIR = "DATASET/train"
    TEST_DIR = "DATASET/test"
    
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Error: DATASET/train or DATASET/test not found.")
        return

    print("--- Checking for Data Leakage ---")
    
    train_hashes = scan_directory(TRAIN_DIR)
    test_hashes = scan_directory(TEST_DIR)
    
    leakage_count = 0
    duplicate_files = []
    
    # Check 1: Exact Hash Match
    for hash_val, test_paths in test_hashes.items():
        if hash_val in train_hashes:
            leakage_count += len(test_paths)
            train_paths = train_hashes[hash_val]
            for test_path in test_paths:
                duplicate_files.append((test_path, train_paths, "Exact Hash Match"))

    # Check 2: Filename Match (potential weak leakage or augmentation)
    train_filenames = {}
    for hash_val, paths in train_hashes.items():
        for path in paths:
            train_filenames[os.path.basename(path)] = path

    filename_matches = 0
    for hash_val, paths in test_hashes.items():
        for path in paths:
            basename = os.path.basename(path)
            if basename in train_filenames and hash_val not in train_hashes:
                # If hash matches, it's already counted above. Only count if hash differs but name same.
                filename_matches += 1
                duplicate_files.append((path, [train_filenames[basename]], "Filename Match (Different Hash)"))

    output_lines = []
    output_lines.append("--- Data Leakage Report ---")
    output_lines.append(f"Total unique images in Train: {len(train_hashes)}")
    output_lines.append(f"Total unique images in Test: {len(test_hashes)}")
    output_lines.append(f"Exact Hash Duplicates: {leakage_count}")
    output_lines.append(f"Filename Matches (Different Hash): {filename_matches}")
    
    if duplicate_files:
        output_lines.append("\nDetailed List:")
        for test_path, train_paths, reason in duplicate_files:
            output_lines.append(f"Test: {test_path}")
            output_lines.append(f"   Matches: {train_paths} ({reason})")
    
    with open("leakage_report.txt", "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"Report saved to leakage_report.txt. Found {leakage_count} hash matches and {filename_matches} filename matches.")


if __name__ == "__main__":
    main()
