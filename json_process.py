```
author: Hanzhe Liang
modified: 2025/6/4
```
import os
import json
from pathlib import Path

def generate_json_metadata(data_dir):
    """Generate train.json and test.json for MulSen_AD_processed dataset"""
    
    data_path = Path(data_dir)
    train_data = []
    test_data = []
    
    # Get all category directories
    categories = [d for d in data_path.iterdir() if d.is_dir()]
    
    for category_dir in categories:
        category_name = category_dir.name
        
        # Process train folder
        train_dir = category_dir / "train"
        if train_dir.exists():
            for pcd_file in train_dir.glob("*.pcd"):
                filename = f"{category_name}/train/{pcd_file.name}"
                
                train_entry = {
                    "filename": filename,
                    "label": 0,
                    "label_name": "good",
                    "clsname": category_name
                }
                train_data.append(train_entry)
        
        # Process test folder
        test_dir = category_dir / "test"
        gt_dir = category_dir / "GT"
        
        if test_dir.exists():
            for pcd_file in test_dir.glob("*.pcd"):
                filename = f"{category_name}/test/{pcd_file.name}"
                base_name = pcd_file.stem  # filename without extension
                
                # Determine label based on filename
                if "_good" in base_name:
                    label = 0
                    label_name = "good"
                    test_entry = {
                        "filename": filename,
                        "label": label,
                        "label_name": label_name,
                        "clsname": category_name
                    }
                elif "_bad" in base_name:
                    label = 1
                    label_name = "defective"
                    
                    # Check if corresponding GT file exists
                    gt_file = gt_dir / f"{base_name}.txt"
                    if gt_file.exists():
                        maskname = f"{category_name}/GT/{base_name}.txt"
                        test_entry = {
                            "filename": filename,
                            "label": label,
                            "label_name": label_name,
                            "maskname": maskname,
                            "clsname": category_name
                        }
                    else:
                        # GT file doesn't exist, skip maskname
                        test_entry = {
                            "filename": filename,
                            "label": label,
                            "label_name": label_name,
                            "clsname": category_name
                        }
                        print(f"Warning: GT file not found for {filename}")
                else:
                    print(f"Warning: Unknown file type {pcd_file.name}, skipping...")
                    continue
                
                test_data.append(test_entry)
    
    # Write train.json
    train_json_path = data_path / "train.json"
    with open(train_json_path, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    
    # Write test.json
    test_json_path = data_path / "test.json"
    with open(test_json_path, 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated train.json with {len(train_data)} entries")
    print(f"Generated test.json with {len(test_data)} entries")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Categories: {len(categories)}")
    
    train_by_category = {}
    test_good_by_category = {}
    test_bad_by_category = {}
    
    for entry in train_data:
        cls = entry["clsname"]
        train_by_category[cls] = train_by_category.get(cls, 0) + 1
    
    for entry in test_data:
        cls = entry["clsname"]
        if entry["label"] == 0:
            test_good_by_category[cls] = test_good_by_category.get(cls, 0) + 1
        else:
            test_bad_by_category[cls] = test_bad_by_category.get(cls, 0) + 1
    
    print("\nPer category breakdown:")
    all_categories = set(train_by_category.keys()) | set(test_good_by_category.keys()) | set(test_bad_by_category.keys())
    for cls in sorted(all_categories):
        train_count = train_by_category.get(cls, 0)
        test_good_count = test_good_by_category.get(cls, 0)
        test_bad_count = test_bad_by_category.get(cls, 0)
        print(f"  {cls}: train={train_count}, test_good={test_good_count}, test_bad={test_bad_count}")

def preview_generated_files(data_dir, num_examples=3):
    """Preview generated JSON files"""
    data_path = Path(data_dir)
    
    print("\n" + "="*50)
    print("PREVIEW OF GENERATED FILES")
    print("="*50)
    
    # Preview train.json
    train_json_path = data_path / "train.json"
    if train_json_path.exists():
        print(f"\nFirst {num_examples} entries in train.json:")
        with open(train_json_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break
                entry = json.loads(line)
                print(f"  {json.dumps(entry, indent=2)}")
    
    # Preview test.json
    test_json_path = data_path / "test.json"
    if test_json_path.exists():
        print(f"\nFirst {num_examples} entries in test.json:")
        with open(test_json_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break
                entry = json.loads(line)
                print(f"  {json.dumps(entry, indent=2)}")

if __name__ == "__main__":
    # Change this to your MulSen_AD_processed folder path
    data_directory = "MulSen_AD_process"
    
    if not os.path.exists(data_directory):
        print(f"Error: Directory {data_directory} not found!")
        print("Please make sure the MulSen_AD_processed folder exists in the current directory.")
    else:
        generate_json_metadata(data_directory)
        preview_generated_files(data_directory)
        print(f"\nJSON files generated successfully in {data_directory}/")
