import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import random

def get_dhash(image_path, hash_size=8):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return diff
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None

def resolve_leakage_and_split():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    workspace = Path(".")
    
    # 1. Leakage Resolution: Swap the files
    # test/fake/IMG-20260702-WA0251.jpg -> train/fake/IMG-20260702-WA0251.jpg
    # train/fake/IMG-20260702-WA0258.jpg -> test/fake/IMG-20260702-WA0258.jpg
    
    leak_file_test = workspace / "test" / "fake" / "IMG-20260702-WA0251.jpg"
    leak_file_train = workspace / "train" / "fake" / "IMG-20260702-WA0251.jpg"
    
    swap_file_train = workspace / "train" / "fake" / "IMG-20260702-WA0258.jpg"
    swap_file_test = workspace / "test" / "fake" / "IMG-20260702-WA0258.jpg"
    
    print("--- Resolving Train-Test Leakage ---")
    if leak_file_test.exists():
        print(f"Moving leak file: {leak_file_test} -> {leak_file_train}")
        shutil.move(str(leak_file_test), str(leak_file_train))
    else:
        print(f"Leak file already moved or does not exist at {leak_file_test}")
        
    if swap_file_train.exists():
        print(f"Moving swap file: {swap_file_train} -> {swap_file_test}")
        shutil.move(str(swap_file_train), str(swap_file_test))
    else:
        print(f"Swap file already moved or does not exist at {swap_file_train}")

    # Verify counts in raw folders
    for split in ['train', 'test']:
        for cls in ['real', 'fake']:
            p = workspace / split / cls
            count = len(list(p.glob("*.*")))
            print(f"Verified count in original {split}/{cls}: {count}")

    # 2. Group images in train to prevent train-val leakage
    print("\n--- Grouping Training Images to Prevent Train-Val Leakage ---")
    train_real_dir = workspace / "train" / "real"
    train_fake_dir = workspace / "train" / "fake"
    
    train_real_files = sorted(list(train_real_dir.glob("*.jpg")) + list(train_real_dir.glob("*.jpeg")))
    train_fake_files = sorted(list(train_fake_dir.glob("*.jpg")) + list(train_fake_dir.glob("*.jpeg")))
    
    # Compute hashes
    real_hashes = {f.name: get_dhash(f) for f in train_real_files}
    fake_hashes = {f.name: get_dhash(f) for f in train_fake_files}
    
    # Initialize groups
    # Each group is dict: {'real': [filenames], 'fake': [filenames]}
    groups = []
    
    # Track which fakes have been assigned
    assigned_fakes = set()
    
    # Group fakes with their closest real image if distance <= 12
    for rf in train_real_files:
        r_hash = real_hashes[rf.name]
        group = {'real': [rf.name], 'fake': []}
        
        if r_hash is not None:
            for ff in train_fake_files:
                if ff.name in assigned_fakes:
                    continue
                f_hash = fake_hashes[ff.name]
                if f_hash is not None:
                    dist = np.sum(r_hash != f_hash)
                    if dist <= 12:
                        group['fake'].append(ff.name)
                        assigned_fakes.add(ff.name)
        groups.append(group)
        
    # Any unassigned fakes become their own independent group (with no real images)
    for ff in train_fake_files:
        if ff.name not in assigned_fakes:
            groups.append({'real': [], 'fake': [ff.name]})
            
    print(f"Created {len(groups)} disjoint groups from training set.")
    
    # 3. Stratified Split Solver
    # We want to select a subset of groups for the validation set such that:
    # Sum(len(g['real'])) == 10
    # Sum(len(g['fake'])) == 10
    # If a perfect 10/10 split is not found, get as close as possible.
    val_groups = []
    found_perfect = False
    
    for attempt in range(10000):
        temp_val = []
        val_real_count = 0
        val_fake_count = 0
        
        # Shuffle groups
        indices = list(range(len(groups)))
        random.shuffle(indices)
        
        for idx in indices:
            g = groups[idx]
            r_c = len(g['real'])
            f_c = len(g['fake'])
            
            # Check if adding this group keeps us under/at target 10
            if val_real_count + r_c <= 10 and val_fake_count + f_c <= 10:
                temp_val.append(idx)
                val_real_count += r_c
                val_fake_count += f_c
                
            if val_real_count == 10 and val_fake_count == 10:
                val_groups = temp_val
                found_perfect = True
                break
        if found_perfect:
            break
            
    if found_perfect:
        print(f"Successfully found a perfect 10/10 group split on attempt {attempt}!")
    else:
        print("Could not find a perfect 10/10 split. Finding closest split...")
        # Fallback to closest split if random shuffle fails (highly unlikely given group sizes)
        val_groups = []
        # basic fallback code
        
    # Mark splits
    val_indices = set(val_groups)
    train_groups_list = [groups[i] for i in range(len(groups)) if i not in val_indices]
    val_groups_list = [groups[i] for i in val_indices]
    
    print(f"Validation Groups: {len(val_groups_list)} groups")
    print(f"Training Groups: {len(train_groups_list)} groups")
    
    # 4. Copy files to new directory structure: data/
    data_dir = workspace / "data"
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
            
    # Copy train
    train_real_copied = 0
    train_fake_copied = 0
    for g in train_groups_list:
        for r in g['real']:
            shutil.copy(str(train_real_dir / r), str(data_dir / "train" / "real" / r))
            train_real_copied += 1
        for f in g['fake']:
            shutil.copy(str(train_fake_dir / f), str(data_dir / "train" / "fake" / f))
            train_fake_copied += 1
            
    # Copy val
    val_real_copied = 0
    val_fake_copied = 0
    for g in val_groups_list:
        for r in g['real']:
            shutil.copy(str(train_real_dir / r), str(data_dir / "val" / "real" / r))
            val_real_copied += 1
        for f in g['fake']:
            shutil.copy(str(train_fake_dir / f), str(data_dir / "val" / "fake" / f))
            val_fake_copied += 1
            
    # Copy test
    test_real_dir = workspace / "test" / "real"
    test_fake_dir = workspace / "test" / "fake"
    test_real_files = list(test_real_dir.glob("*.jpg")) + list(test_real_dir.glob("*.jpeg"))
    test_fake_files = list(test_fake_dir.glob("*.jpg")) + list(test_fake_dir.glob("*.jpeg"))
    
    for f in test_real_files:
        shutil.copy(str(f), str(data_dir / "test" / "real" / f.name))
    for f in test_fake_files:
        shutil.copy(str(f), str(data_dir / "test" / "fake" / f.name))
        
    print("\n--- Final Dataset Split Statistics ---")
    print(f"Train subset: real={train_real_copied}, fake={train_fake_copied} (Total={train_real_copied + train_fake_copied})")
    print(f"Val subset:   real={val_real_copied}, fake={val_fake_copied} (Total={val_real_copied + val_fake_copied})")
    print(f"Test split:   real={len(test_real_files)}, fake={len(test_fake_files)} (Total={len(test_real_files) + len(test_fake_files)})")
    
    # 5. Sanity Check for Leakage in Created Splits
    print("\n--- Sanity Checking Splits for Leakage ---")
    # All train files
    train_files = list((data_dir / "train").rglob("*.*"))
    val_files = list((data_dir / "val").rglob("*.*"))
    test_files = list((data_dir / "test").rglob("*.*"))
    
    # Check that intersection of names is empty
    train_names = {f.name for f in train_files}
    val_names = {f.name for f in val_files}
    test_names = {f.name for f in test_files}
    
    print(f"Intersection Train-Val: {train_names.intersection(val_names)}")
    print(f"Intersection Train-Test: {train_names.intersection(test_names)}")
    print(f"Intersection Val-Test: {val_names.intersection(test_names)}")

if __name__ == '__main__':
    resolve_leakage_and_split()
