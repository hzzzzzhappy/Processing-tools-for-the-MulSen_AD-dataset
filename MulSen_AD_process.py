"""
author: Hanzhe Liang
modified: 2025/06/04
"""
import os
import glob
import shutil
import time
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.neighbors import KDTree

def move_pointcloud_contents():
    for item in Path(".").iterdir():
        if item.is_dir():
            pointcloud_dir = item / "Pointcloud"
            if pointcloud_dir.exists():
                for pc_item in pointcloud_dir.iterdir():
                    destination = item / pc_item.name
                    if not destination.exists():
                        shutil.move(str(pc_item), str(destination))
                try:
                    pointcloud_dir.rmdir()
                except OSError:
                    pass

def reorganize_files():
    for class_dir in Path(".").iterdir():
        if not class_dir.is_dir():
            continue
            
        test_dir = class_dir / "test"
        gt_dir = class_dir / "GT"
        
        if not test_dir.exists():
            continue
            
        counter = 0
        
        for subfolder in test_dir.iterdir():
            if subfolder.is_dir() and subfolder.name not in ["good", "color"]:
                gt_subfolder = gt_dir / subfolder.name if gt_dir.exists() else None
                
                test_files = list(subfolder.glob("*"))
                gt_files = list(gt_subfolder.glob("*")) if gt_subfolder and gt_subfolder.exists() else []
                
                max_files = max(len(test_files), len(gt_files))
                
                for i in range(max_files):
                    new_name = f"{counter}_bad"
                    
                    if i < len(test_files):
                        test_ext = test_files[i].suffix
                        shutil.move(str(test_files[i]), str(test_dir / f"{new_name}{test_ext}"))
                    
                    if i < len(gt_files):
                        gt_ext = gt_files[i].suffix
                        shutil.move(str(gt_files[i]), str(gt_dir / f"{new_name}{gt_ext}"))
                    
                    counter += 1
                
                if subfolder.exists():
                    try:
                        subfolder.rmdir()
                    except OSError:
                        pass
                if gt_subfolder and gt_subfolder.exists():
                    try:
                        gt_subfolder.rmdir()
                    except OSError:
                        pass
        
        good_folder = test_dir / "good"
        if good_folder.exists():
            for good_file in good_folder.glob("*"):
                if good_file.is_file():
                    new_name = f"{counter}_good{good_file.suffix}"
                    shutil.move(str(good_file), str(test_dir / new_name))
                    counter += 1
            
            try:
                good_folder.rmdir()
            except OSError:
                pass
        
        color_folder = test_dir / "color"
        if color_folder.exists():
            shutil.rmtree(color_folder)

def norm_pcd(point_cloud):
    center = np.average(point_cloud, axis=0)
    return point_cloud - np.expand_dims(center, axis=0)

def mark_stl_with_anomalies(stl_vertices, txt_points, tolerance=1000):
    labels = np.zeros(len(stl_vertices), dtype=int)
    if len(txt_points) == 0:
        return labels
    tree = KDTree(stl_vertices)
    for txt_point in txt_points:
        dist, idx = tree.query(txt_point.reshape(1, -1))
        if dist[0] < tolerance:
            labels[idx[0]] = 1
    return labels

def load_stl_vertices(stl_path):
    try:
        mesh_stl = o3d.io.read_triangle_mesh(stl_path)
        mesh_stl = mesh_stl.remove_duplicated_vertices()
        return np.asarray(mesh_stl.vertices)
    except:
        return None

def load_gt_points(gt_path):
    try:
        if os.path.exists(gt_path):
            pcd = np.genfromtxt(gt_path, delimiter=",")
            if pcd.ndim == 1:
                pcd = pcd.reshape(1, -1)
            return pcd[:, :3]
        return np.array([])
    except:
        return np.array([])

def save_labels_to_txt(vertices, labels, output_path):
    normalized_vertices = norm_pcd(vertices)
    data = np.column_stack([normalized_vertices, labels])
    np.savetxt(output_path, data, delimiter=',', fmt='%.6f,%.6f,%.6f,%d')

def process_category_labels(category_path):
    category_name = os.path.basename(category_path)
    print(f"Processing category: {category_name}")
    
    test_path = os.path.join(category_path, 'test')
    gt_input_path = os.path.join(category_path, 'GT')
    gt_output_path = os.path.join(category_path, 'gt')
    
    if not os.path.exists(test_path):
        print(f"  Warning: test folder not found in {category_name}")
        return
    
    if os.path.exists(gt_output_path):
        shutil.rmtree(gt_output_path)
        print(f"  Cleaned existing gt folder in {category_name}")
    os.makedirs(gt_output_path, exist_ok=True)
    
    stl_files = glob.glob(os.path.join(test_path, "*.stl"))
    print(f"  Found {len(stl_files)} STL files in {category_name}")
    
    for stl_file in stl_files:
        base_name = os.path.splitext(os.path.basename(stl_file))[0]
        vertices = load_stl_vertices(stl_file)
        if vertices is None:
            print(f"    Failed to load: {base_name}")
            continue
        
        if '_good' in base_name:
            labels = np.zeros(len(vertices), dtype=int)
        elif '_bad' in base_name:
            gt_file = os.path.join(gt_input_path, f"{base_name}.txt")
            gt_points = load_gt_points(gt_file)
            labels = mark_stl_with_anomalies(vertices, gt_points) if len(gt_points) > 0 else np.zeros(len(vertices), dtype=int)
        else:
            print(f"    Skipped unknown type: {base_name}")
            continue
        
        output_file = os.path.join(gt_output_path, f"{base_name}.txt")
        save_labels_to_txt(vertices, labels, output_file)
        print(f"    Processed: {base_name} ({len(vertices)} vertices)")

def generate_gt_labels():
    current_dir = os.getcwd()
    categories = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
    
    for category in categories:
        category_path = os.path.join(current_dir, category)
        process_category_labels(category_path)
    
    print("Completed!")

def save_txt_to_pcd(txt_path, pcd_path):
    try:
        data = np.genfromtxt(txt_path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        points = data[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(pcd_path, pcd)
        return True
    except:
        return False

def save_stl_to_pcd(stl_path, pcd_path):
    try:
        vertices = load_stl_vertices(stl_path)
        if vertices is None:
            return False
        
        normalized_vertices = norm_pcd(vertices)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(normalized_vertices)
        o3d.io.write_point_cloud(pcd_path, pcd)
        return True
    except:
        return False

def process_category_conversion(category_path):
    category_name = os.path.basename(category_path)
    print(f"Processing category: {category_name}")
    
    gt_path = os.path.join(category_path, 'gt')
    train_path = os.path.join(category_path, 'train')
    test_output_path = os.path.join(category_path, 'Test')
    train_output_path = os.path.join(category_path, 'Train')
    
    if os.path.exists(gt_path):
        if os.path.exists(test_output_path):
            shutil.rmtree(test_output_path)
            print(f"  Cleaned existing Test folder in {category_name}")
        os.makedirs(test_output_path, exist_ok=True)
        
        txt_files = glob.glob(os.path.join(gt_path, "*.txt"))
        print(f"  Found {len(txt_files)} TXT files in gt folder")
        
        for txt_file in txt_files:
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            pcd_file = os.path.join(test_output_path, f"{base_name}.pcd")
            
            if save_txt_to_pcd(txt_file, pcd_file):
                print(f"    Converted: {base_name}.txt -> {base_name}.pcd")
            else:
                print(f"    Failed to convert: {base_name}.txt")
    else:
        print(f"  No gt folder found in {category_name}")
    
    if os.path.exists(train_path):
        if os.path.exists(train_output_path):
            shutil.rmtree(train_output_path)
            print(f"  Cleaned existing Train folder in {category_name}")
        os.makedirs(train_output_path, exist_ok=True)
        
        stl_files = glob.glob(os.path.join(train_path, "*.stl"))
        print(f"  Found {len(stl_files)} STL files in train folder")
        
        for stl_file in stl_files:
            base_name = os.path.splitext(os.path.basename(stl_file))[0]
            pcd_file = os.path.join(train_output_path, f"{base_name}.pcd")
            
            if save_stl_to_pcd(stl_file, pcd_file):
                print(f"    Converted: {base_name}.stl -> {base_name}.pcd")
            else:
                print(f"    Failed to convert: {base_name}.stl")
    else:
        print(f"  No train folder found in {category_name}")

def convert_to_pcd():
    current_dir = os.getcwd()
    categories = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
    
    print(f"Found {len(categories)} categories: {categories}")
    
    for category in categories:
        category_path = os.path.join(current_dir, category)
        process_category_conversion(category_path)
    
    print("All processing completed!")

def copy_and_rename_folders(source_category, target_category):
    folders_to_copy = {
        'Train': 'train',
        'Test': 'test', 
        'gt': 'GT'
    }
    
    copied_count = 0
    for source_folder, target_folder in folders_to_copy.items():
        source_path = os.path.join(source_category, source_folder)
        target_path = os.path.join(target_category, target_folder)
        
        if os.path.exists(source_path):
            shutil.copytree(source_path, target_path)
            print(f"    Copied: {source_folder} -> {target_folder}")
            copied_count += 1
        else:
            print(f"    Warning: {source_folder} not found")
    
    return copied_count

def remove_good_files_from_gt(target_dir):
    categories = [d for d in os.listdir(target_dir) 
                 if os.path.isdir(os.path.join(target_dir, d))]
    
    for category in categories:
        gt_path = os.path.join(target_dir, category, 'GT')
        if os.path.exists(gt_path):
            files = os.listdir(gt_path)
            good_files = [f for f in files if 'good' in f]
            
            for good_file in good_files:
                file_path = os.path.join(gt_path, good_file)
                os.remove(file_path)
                print(f"    Removed: {category}/GT/{good_file}")
            
            if good_files:
                print(f"  Removed {len(good_files)} good files from {category}/GT")

def create_final_dataset():
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'MulSen_AD_processed')
    parent_dir = os.path.dirname(current_dir)
    
    categories = [d for d in os.listdir(current_dir) 
                 if os.path.isdir(os.path.join(current_dir, d)) and d != 'MulSen_AD_processed']
    
    print(f"Found {len(categories)} categories: {categories}")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"Cleaned existing MulSen_AD_processed folder")
    
    os.makedirs(target_dir)
    print(f"Created MulSen_AD_processed folder")
    
    for category in categories:
        source_category_path = os.path.join(current_dir, category)
        target_category_path = os.path.join(target_dir, category)
        
        print(f"\nProcessing category: {category}")
        
        os.makedirs(target_category_path)
        
        copied_count = copy_and_rename_folders(source_category_path, target_category_path)
        
        if copied_count == 0:
            print(f"    No valid folders found for {category}, removing empty directory")
            os.rmdir(target_category_path)
        else:
            print(f"    Successfully processed {category} ({copied_count}/3 folders copied)")
    
    print(f"\nRemoving files containing 'good' from GT folders...")
    remove_good_files_from_gt(target_dir)
    
    parent_target = os.path.join(parent_dir, 'MulSen_AD_processed')
    if os.path.exists(parent_target):
        shutil.rmtree(parent_target)
        print(f"Cleaned existing MulSen_AD_processed in parent directory")
    
    shutil.move(target_dir, parent_target)
    print(f"\nMoved MulSen_AD_processed to parent directory: {parent_target}")
    
    print(f"\nAll processing completed! MulSen_AD_processed folder created with reorganized structure in parent directory.")

def cleanup_intermediate_files():
    current_dir = os.getcwd()
    folders_to_delete = ['gt', 'Test', 'Train']
    
    categories = [d for d in os.listdir(current_dir) 
                 if os.path.isdir(os.path.join(current_dir, d))]
    
    print(f"Found {len(categories)} directories: {categories}")
    
    total_deleted = 0
    
    for category in categories:
        category_path = os.path.join(current_dir, category)
        deleted_in_category = 0
        
        print(f"\nProcessing: {category}")
        
        for folder_name in folders_to_delete:
            folder_path = os.path.join(category_path, folder_name)
            
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"  Deleted: {folder_name}")
                deleted_in_category += 1
                total_deleted += 1
            else:
                print(f"  Not found: {folder_name}")
        
        print(f"  Deleted {deleted_in_category}/3 folders from {category}")
    
    print(f"\nCleanup completed! Total folders deleted: {total_deleted}")

def run_all_steps():
    print("Step 1: Moving pointcloud contents...")
    move_pointcloud_contents()
    time.sleep(1)  
    print("\nStep 2: Reorganizing files...")
    reorganize_files()
    time.sleep(1)  
    print("\nStep 3: Generating GT labels...")
    generate_gt_labels()
    time.sleep(1)  
    print("\nStep 4: Converting to PCD format...")
    convert_to_pcd()
    time.sleep(1)  
    print("\nStep 5: Creating final dataset...")
    create_final_dataset()
    time.sleep(1)  
    print("\nStep 6: Cleaning up intermediate files...")
    cleanup_intermediate_files()
    time.sleep(1)  
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    run_all_steps()