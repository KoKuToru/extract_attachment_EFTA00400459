import os
import hashlib
from PIL import Image
from glob import glob
from tqdm import tqdm

def hash_image(filepath):
    """Calculates the SHA256 hash of an image's pixel data."""
    try:
        # Open the image, convert to 'RGB' to standardize format
        # and get raw pixel data for hashing.
        img = Image.open(filepath).convert('RGB')
        # Get raw pixel data as bytes
        img_bytes = img.tobytes()
        hasher = hashlib.sha256()
        hasher.update(img_bytes)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def deduplicate_letters(letters_done_dir='letters_done', letters_dir='letters'):
    print(f"Deduplicating images from '{letters_dir}' against '{letters_done_dir}'...")

    # Collect hashes from letters_done_dir
    letters_done_hashes = set()
    letters_done_files = glob(os.path.join(letters_done_dir, '*.png'))
    print(f"Hashing {len(letters_done_files)} files in '{letters_done_dir}'...")
    for filepath in tqdm(letters_done_files):
        img_hash = hash_image(filepath)
        if img_hash:
            letters_done_hashes.add(img_hash)
    print(f"Collected {len(letters_done_hashes)} unique hashes from '{letters_done_dir}'.")

    # Deduplicate files in letters_dir
    letters_files = glob(os.path.join(letters_dir, '*.png'))
    print(f"Processing {len(letters_files)} files in '{letters_dir}' for duplicates...")
    removed_count = 0
    for filepath in tqdm(letters_files):
        img_hash = hash_image(filepath)
        if img_hash and img_hash in letters_done_hashes:
            print(f"Removing duplicate: {filepath}")
            os.remove(filepath)
            removed_count += 1
    print(f"Removed {removed_count} duplicate files from '{letters_dir}'.")

if __name__ == '__main__':
    deduplicate_letters()
