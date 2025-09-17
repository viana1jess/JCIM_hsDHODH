# -*- coding: utf-8 -*-
"""
This script automates the process of screening a single query molecule against
a large multi-molecule SDF library (like BindingDB or a detailed PubChem library)
using LSalign.

It leverages parallel processing to significantly speed up the screening workflow.
For each molecule in the library, it extracts the structure and relevant metadata,
runs the LSalign comparison, and collates the scores and data into a
single detailed CSV output file.

Prerequisites:
- 'obabel' must be installed and accessible in the system's PATH.
- 'LSalign' must be installed and accessible in the system's PATH.
- It's recommended to install 'tqdm' for a progress bar (`pip install tqdm`).
"""

import os
import csv
import argparse
import subprocess
import tempfile
import shutil
from multiprocessing import Pool
from functools import partial
import re

# It's recommended to install tqdm for a nice progress bar: pip install tqdm
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

def read_sdf_library_generator(sdf_filepath, sample_size=None):
    """
    Reads a large, multi-molecule SDF file and yields individual molecule
    blocks as strings. This is a generator function to save memory.
    """
    if not os.path.exists(sdf_filepath):
        print(f"Error: Library file '{sdf_filepath}' not found.")
        return

    print(f"Reading library file: {sdf_filepath}...")
    mol_buffer = []
    count = 0
    with open(sdf_filepath, 'r', errors='ignore') as f:
        for line in f:
            mol_buffer.append(line)
            if '$$$$' in line:
                yield "".join(mol_buffer)
                mol_buffer = []
                count += 1
                if sample_size and count >= sample_size:
                    print(f"Reached sample limit of {sample_size} molecules.")
                    break

def parse_sdf_metadata(sdf_block):
    """
    Parses a single molecule's SDF block to extract specific metadata fields.
    This version is restored to capture detailed annotations.
    """
    metadata = {
        'Target Name': 'N/A',
        'Article DOI': 'N/A',
        'PubChem CID of Ligand': 'N/A',
        'ZINC ID of Ligand': 'N/A'
    }
    # Get the molecule title from the first line.
    title = sdf_block.split('\n', 1)[0].strip()
    metadata['SDF_Title'] = title if title else 'N/A'

    # Find a field and capture the line that follows.
    for key in metadata:
        search_key = f"> <{key}>"
        try:
            # Find the line with the key, and get the line right after it.
            lines = sdf_block.splitlines()
            for i, line in enumerate(lines):
                if search_key in line:
                    value = lines[i + 1].strip()
                    if value:
                        metadata[key] = value
                    break # Move to the next key
        except (IndexError, ValueError):
            # This handles cases where the key is not found or is at the end of the block
            continue

    return metadata


def run_comparison(task_data, query_mol2_path, query_id, main_temp_dir):
    """
    Worker function to run the full comparison workflow for a single library molecule.
    `task_data` is a tuple: (index, sdf_block)
    """
    index, sdf_block = task_data
    
    # --- Step 1: Parse metadata and determine a unique ID for the target ---
    metadata = parse_sdf_metadata(sdf_block)
    # Use a hierarchy of potential IDs for the target molecule
    target_id = metadata.get('PubChem CID of Ligand') or metadata.get('ZINC ID of Ligand') or metadata.get('SDF_Title') or f"index_{index}"
    safe_target_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(target_id))

    # --- RESTORED: Result dictionary with all metadata fields ---
    result_data = {
        'Query': query_id,
        'Target_ID': target_id,
        'PC-score8Q': 'N/A',
        'PC-score8T': 'N/A',
        'Status': 'Failed',
        'Error_Info': 'N/A'
    }
    # Add all parsed metadata to the result
    result_data.update(metadata)

    # Create a unique subdirectory for this worker to avoid file collisions
    worker_temp_dir = os.path.join(main_temp_dir, f"{query_id}_vs_{safe_target_id}_{index}")
    os.makedirs(worker_temp_dir, exist_ok=True)

    temp_sdf_path = os.path.join(worker_temp_dir, f"{safe_target_id}.sdf")
    temp_mol2_path = os.path.join(worker_temp_dir, f"{safe_target_id}.mol2")

    try:
        # --- Step 2: Write the single molecule SDF block to a temp file ---
        with open(temp_sdf_path, 'w', encoding='utf-8') as f:
            f.write(sdf_block)
            if not sdf_block.endswith('$$$$\n'):
                f.write('\n$$$$\n')

        # --- Step 3: Convert temporary SDF to MOL2 using obabel ---
        obabel_proc = subprocess.run(
            ['obabel', temp_sdf_path, '-O', temp_mol2_path, '--title', safe_target_id],
            capture_output=True, text=True
        )

        if obabel_proc.returncode != 0:
            result_data['Status'] = 'Obabel Error'
            result_data['Error_Info'] = obabel_proc.stderr.strip()
            shutil.rmtree(worker_temp_dir)
            return result_data

        # --- Step 4: Run LSalign using explicit file paths ---
        lsalign_proc = subprocess.run(
            ['LSalign', query_mol2_path, temp_mol2_path],
            capture_output=True, text=True
        )
        
        # --- Step 5: Parse the LSalign output ---
        output_lines = lsalign_proc.stdout.strip().split('\n')
        parsed = False
        for line in output_lines:
            parts = line.split()
            if len(parts) > 3 and parts[1] == safe_target_id:
                result_data['PC-score8Q'] = parts[2]
                result_data['PC-score8T'] = parts[3]
                result_data['Status'] = 'Success'
                result_data['Error_Info'] = 'N/A'
                parsed = True
                break
        
        if not parsed:
            result_data['Status'] = 'Parsing Error'
            result_data['Error_Info'] = lsalign_proc.stdout.strip() or lsalign_proc.stderr.strip()

        shutil.rmtree(worker_temp_dir) # Clean up the temporary directory
        return result_data

    except FileNotFoundError as e:
        print(f"\n--- FATAL ERROR in worker: Command not found: '{e.filename}'. Ensure it's in your PATH. ---")
        raise
    except Exception as e:
        result_data['Status'] = 'Unhandled Exception'
        result_data['Error_Info'] = str(e)
        if os.path.exists(worker_temp_dir):
            shutil.rmtree(worker_temp_dir)
        return result_data

def main():
    """Main function to run the entire parallel screening workflow."""
    parser = argparse.ArgumentParser(
        description="Runs a parallel LSalign screen of a query molecule against a large SDF library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-q", "--query", required=True, type=str, help="Path to the query molecule file (e.g., query.mol2).")
    parser.add_argument("-l", "--library", required=True, type=str, help="Path to the large multi-molecule SDF library file.")
    parser.add_argument("-o", "--output", type=str, default="screening_results.csv", help="Name of the output CSV file.")
    parser.add_argument("-p", "--processes", type=int, default=os.cpu_count(), help="Number of parallel processes to use. Defaults to all available CPU cores.")
    parser.add_argument("-s", "--sample", type=int, default=None, help="Run on a sample of the first N compounds in the library for testing.")
    args = parser.parse_args()

    # --- Pre-flight checks ---
    print("Checking for required command-line tools...")
    if not shutil.which("obabel"):
        print("\nFATAL ERROR: 'obabel' command not found. Please ensure Open Babel is installed and in your system's PATH.")
        return
    if not shutil.which("LSalign"):
        print("\nFATAL ERROR: 'LSalign' command not found. Please ensure LSalign is installed and in your system's PATH.")
        return
    
    query_abs_path = os.path.abspath(args.query)
    if not os.path.exists(query_abs_path):
        print(f"\nFATAL ERROR: Query file not found at '{query_abs_path}'")
        return
    print("--> 'obabel' and 'LSalign' found successfully.\n")

    # --- 1. Prepare tasks ---
    library_molecule_generator = read_sdf_library_generator(args.library, args.sample)
    tasks = enumerate(library_molecule_generator)
    query_id = os.path.splitext(os.path.basename(args.query))[0]
    
    if args.sample:
         print(f"Prepared a sample of {args.sample} comparison tasks for query '{query_id}'.")
    else:
         print(f"Prepared comparison tasks for query '{query_id}'. Starting now...")

    # --- 2. Run the parallel workflow ---
    with tempfile.TemporaryDirectory() as main_temp_dir:
        print(f"Using main temporary directory: {main_temp_dir}")
        worker_func = partial(run_comparison, 
                              query_mol2_path=query_abs_path,
                              query_id=query_id,
                              main_temp_dir=main_temp_dir)
                              
        print(f"Starting parallel screening with {args.processes} workers...")
        results = []
        with Pool(args.processes) as pool:
            if USE_TQDM:
                results = list(tqdm(pool.imap_unordered(worker_func, tasks), desc="Screening Library"))
            else:
                results = pool.map(worker_func, tasks)
                print("Processing complete.")

    # --- 3. Write results to CSV ---
    if not results:
        print("\nWARNING: No results were generated. Please check the library file format and content.")
        return

    print(f"\nWriting {len(results)} total attempted comparisons to '{args.output}'...")
    
    # --- RESTORED: Headers for the detailed output CSV ---
    headers = [
        'Query', 'Target_ID', 'PC-score8Q', 'PC-score8T', 'Status', 'SDF_Title',
        'Target Name', 'Article DOI', 'PubChem CID of Ligand', 'ZINC ID of Ligand',
        'Error_Info'
    ]
    
    final_results = [r for r in results if r is not None]
    
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_results)

    print("\n--- ✅ All Done! ---")
    print(f"Output file '{args.output}' is ready and contains all attempts.")

if __name__ == "__main__":
    main()
