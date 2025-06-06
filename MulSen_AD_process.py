"""
author: Hanzhe Liang
modified: 2025/06/06
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
    categories = list(Path(".").iterdir())
    categories = [d for d in categories if d.is_dir()]
    
    for class_dir in categories:
        test_dir = class_dir / "test"
        gt_dir = class_dir / "GT"
        train_dir = class_dir / "train"
        gt_output_dir = class_dir / "gt1"
        test_output_dir = class_dir / "test1"
        train_output_dir = class_dir / "train1"
        
        if not test_dir.exists():
            continue
        
        if gt_output_dir.exists():
            shutil.rmtree(gt_output_dir)
        gt_output_dir.mkdir()
        
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        test_output_dir.mkdir()
        
        if train_output_dir.exists():
            shutil.rmtree(train_output_dir)
        train_output_dir.mkdir()
        
        for subfolder in test_dir.iterdir():
            if subfolder.is_dir():
                gt_subfolder = gt_dir / subfolder.name if gt_dir.exists() else None
                gt_output_subfolder = gt_output_dir / subfolder.name
                gt_output_subfolder.mkdir()
                
                test_files = list(subfolder.glob("*"))
                for test_file in test_files:
                    if test_file.is_file():
                        test_base_name = test_file.stem
                        
                        vertices = load_stl_vertices(str(test_file))
                        if vertices is None:
                            continue
                        
                        gt_file_path = gt_subfolder / "{}.txt".format(test_base_name) if gt_subfolder and gt_subfolder.exists() else None
                        
                        if gt_file_path and gt_file_path.exists():
                            gt_points = load_gt_points(str(gt_file_path))
                            if len(gt_points) > 0:
                                labels = mark_stl_with_anomalies(vertices, gt_points)
                            else:
                                labels = np.zeros(len(vertices), dtype=int)
                        else:
                            labels = np.zeros(len(vertices), dtype=int)
                        
                        output_file = gt_output_subfolder / "{}.txt".format(test_base_name)
                        save_labels_to_txt(vertices, labels, str(output_file))
        
        counter = 0
        for subfolder in test_dir.iterdir():
            if subfolder.is_dir():
                gt_subfolder = gt_dir / subfolder.name if gt_dir.exists() else None
                gt_output_subfolder = gt_output_dir / subfolder.name
                
                test_files = list(subfolder.glob("*"))
                for test_file in test_files:
                    if test_file.is_file():
                        test_base_name = test_file.stem
                        
                        gt_file_path = gt_subfolder / "{}.txt".format(test_base_name) if gt_subfolder and gt_subfolder.exists() else None
                        
                        if gt_file_path and gt_file_path.exists():
                            new_name = "{}_bad".format(counter)
                        else:
                            new_name = "{}_good".format(counter)
                        
                        save_stl_to_pcd(str(test_file), str(test_output_dir / "{}.pcd".format(new_name)))
                        
                        gt_output_file = gt_output_subfolder / "{}.txt".format(test_base_name)
                        if gt_output_file.exists():
                            shutil.move(str(gt_output_file), str(gt_output_dir / "{}.txt".format(new_name)))
                        
                        counter += 1
                
                if gt_output_subfolder.exists():
                    try:
                        gt_output_subfolder.rmdir()
                    except OSError:
                        pass
        
        if train_dir.exists():
            train_files = list(train_dir.glob("*.stl"))
            for train_file in train_files:
                if train_file.is_file():
                    train_base_name = train_file.stem
                    save_stl_to_pcd(str(train_file), str(train_output_dir / "{}.pcd".format(train_base_name)))
        
        if gt_dir.exists():
            all_gt_files = []
            for gt_subfolder in gt_dir.iterdir():
                if gt_subfolder.is_dir():
                    for gt_file in gt_subfolder.glob("*.txt"):
                        if gt_file.is_file():
                            all_gt_files.append(gt_file)
                    try:
                        gt_subfolder.rmdir()
                    except OSError:
                        pass
            
            all_gt_files.sort(key=lambda x: x.name)
            
            for i, gt_file in enumerate(all_gt_files, 1):
                new_gt_name = "{}.txt".format(i)
                shutil.move(str(gt_file), str(gt_dir / new_gt_name))

def norm_pcd(point_cloud):
    center = np.average(point_cloud, axis=0)
    return point_cloud - np.expand_dims(center, axis=0)

def mark_stl_with_anomalies(stl_vertices, txt_points, tolerance=1000):
    labels = np.zeros(len(stl_vertices), dtype=int)
    # labels = np.ones(len(stl_vertices), dtype=int)

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
    print("Processing category: {}".format(category_name))
    
    test_path = os.path.join(category_path, 'test')
    gt_input_path = os.path.join(category_path, 'GT')
    gt_output_path = os.path.join(category_path, 'gt1')
    
    if not os.path.exists(test_path):
        print("  Warning: test folder not found in {}".format(category_name))
        return
    
    if os.path.exists(gt_output_path):
        shutil.rmtree(gt_output_path)
        print("  Cleaned existing gt folder in {}".format(category_name))
    os.makedirs(gt_output_path, exist_ok=True)
    
    stl_files = glob.glob(os.path.join(test_path, "*.stl"))
    print("  Found {} STL files in {}".format(len(stl_files), category_name))
    
    for stl_file in stl_files:
        base_name = os.path.splitext(os.path.basename(stl_file))[0]
        vertices = load_stl_vertices(stl_file)
        if vertices is None:
            print("    Failed to load: {}".format(base_name))
            continue
        
        if '_good' in base_name:
            labels = np.zeros(len(vertices), dtype=int)
            print("    Processing good sample: {}".format(base_name))
        elif '_bad' in base_name:
            gt_file = os.path.join(gt_input_path, "{}.txt".format(base_name))
            gt_points = load_gt_points(gt_file)
            if len(gt_points) > 0:
                labels = mark_stl_with_anomalies(vertices, gt_points)
                print("    Processing bad sample with GT: {}".format(base_name))
            else:
                labels = np.zeros(len(vertices), dtype=int)
                print("    Warning: bad sample without GT, treating as good: {}".format(base_name))
        else:
            print("    Skipped unknown type: {}".format(base_name))
            continue
        
        output_file = os.path.join(gt_output_path, "{}.txt".format(base_name))
        save_labels_to_txt(vertices, labels, output_file)
        print("    Processed: {} ({} vertices)".format(base_name, len(vertices)))

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
    print("Processing category: {}".format(category_name))
    
    gt_path = os.path.join(category_path, 'gt1')
    train_path = os.path.join(category_path, 'train')
    test_output_path = os.path.join(category_path, 'Test1')
    train_output_path = os.path.join(category_path, 'Train1')
    
    if os.path.exists(gt_path):
        if os.path.exists(test_output_path):
            shutil.rmtree(test_output_path)
            print("  Cleaned existing Test folder in {}".format(category_name))
        os.makedirs(test_output_path, exist_ok=True)
        
        txt_files = glob.glob(os.path.join(gt_path, "*.txt"))
        print("  Found {} TXT files in gt folder".format(len(txt_files)))
        
        for txt_file in txt_files:
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            pcd_file = os.path.join(test_output_path, "{}.pcd".format(base_name))
            
            if save_txt_to_pcd(txt_file, pcd_file):
                print("    Converted: {}.txt -> {}.pcd".format(base_name, base_name))
            else:
                print("    Failed to convert: {}.txt".format(base_name))
    else:
        print("  No gt folder found in {}".format(category_name))
    
    if os.path.exists(train_path):
        if os.path.exists(train_output_path):
            shutil.rmtree(train_output_path)
            print("  Cleaned existing Train folder in {}".format(category_name))
        os.makedirs(train_output_path, exist_ok=True)
        
        stl_files = glob.glob(os.path.join(train_path, "*.stl"))
        print("  Found {} STL files in train folder".format(len(stl_files)))
        
        for stl_file in stl_files:
            base_name = os.path.splitext(os.path.basename(stl_file))[0]
            pcd_file = os.path.join(train_output_path, "{}.pcd".format(base_name))
            
            if save_stl_to_pcd(stl_file, pcd_file):
                print("    Converted: {}.stl -> {}.pcd".format(base_name, base_name))
            else:
                print("    Failed to convert: {}.stl".format(base_name))
    else:
        print("  No train folder found in {}".format(category_name))

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
        'Train1': 'train',
        'Test1': 'test', 
        'gt1': 'GT'
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
    target_dir = os.path.join(current_dir, 'MulSen_AD_process')
    
    categories = [d for d in os.listdir(current_dir) 
                 if os.path.isdir(os.path.join(current_dir, d)) and d != 'MulSen_AD_process']
    
    print(f"Found {len(categories)} categories: {categories}")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"Cleaned existing MulSen_AD_process folder")
    
    os.makedirs(target_dir)
    print(f"Created MulSen_AD_process folder")
    
    for category in categories:
        source_category_path = os.path.join(current_dir, category)
        target_category_path = os.path.join(target_dir, category)
        
        print(f"\nProcessing category: {category}")
        
        os.makedirs(target_category_path)
        
        folders_to_copy = {
            'train1': 'train1',
            'test1': 'test1', 
            'gt1': 'gt1'
        }
        
        copied_count = 0
        for source_folder, target_folder in folders_to_copy.items():
            source_path = os.path.join(source_category_path, source_folder)
            target_path = os.path.join(target_category_path, target_folder)
            
            if os.path.exists(source_path):
                shutil.copytree(source_path, target_path)
                print(f"    Copied: {source_folder} -> {target_folder}")
                copied_count += 1
            else:
                print(f"    Warning: {source_folder} not found")
        
        if copied_count == 0:
            print(f"    No valid folders found for {category}, removing empty directory")
            os.rmdir(target_category_path)
        else:
            print(f"    Successfully processed {category} ({copied_count}/3 folders copied)")
    
    print(f"\nRenaming folders to final structure...")
    for category in os.listdir(target_dir):
        category_path = os.path.join(target_dir, category)
        if os.path.isdir(category_path):
            rename_mappings = {
                'train1': 'train',
                'test1': 'test',
                'gt1': 'GT'
            }
            
            for old_name, new_name in rename_mappings.items():
                old_path = os.path.join(category_path, old_name)
                new_path = os.path.join(category_path, new_name)
                
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f"    Renamed: {category}/{old_name} -> {category}/{new_name}")
    
    print(f"\nRemoving good files from GT folders...")
    for category in os.listdir(target_dir):
        category_path = os.path.join(target_dir, category)
        gt_path = os.path.join(category_path, 'GT')
        if os.path.isdir(gt_path):
            files = os.listdir(gt_path)
            good_files = [f for f in files if 'good' in f]
            
            for good_file in good_files:
                file_path = os.path.join(gt_path, good_file)
                os.remove(file_path)
                print(f"    Removed: {category}/GT/{good_file}")
            
            if good_files:
                print(f"    Removed {len(good_files)} good files from {category}/GT")
    
    print(f"\nAll processing completed! MulSen_AD_process folder created with final structure.")

def cleanup_intermediate_files():
    current_dir = os.getcwd()
    folders_to_delete = ['gt1', 'test1', 'train1']
    
    categories = [d for d in os.listdir(current_dir) 
                 if os.path.isdir(os.path.join(current_dir, d)) and d != 'MulSen_AD_process']
    
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

def cleanup_intermediate_files():
    current_dir = os.getcwd()
    folders_to_delete = ['gt1', 'test1', 'train1']
    
    categories = [d for d in os.listdir(current_dir) 
                 if os.path.isdir(os.path.join(current_dir, d)) and d != 'MulSen_AD_process']
    
    for category in categories:
        category_path = os.path.join(current_dir, category)
        
        for folder_name in folders_to_delete:
            folder_path = os.path.join(category_path, folder_name)
            
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)

def run_all_steps():
    move_pointcloud_contents()
    time.sleep(1)  
    reorganize_files()
    time.sleep(1)  
    create_final_dataset()
    time.sleep(1)  
    cleanup_intermediate_files()
    time.sleep(1)
    restore_original_structure()
    time.sleep(1)

if __name__ == "__main__":
    run_all_steps()
