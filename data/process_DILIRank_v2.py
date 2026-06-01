"""
Process DILIRank v2.0 data by fetching SMILES, CIDs, ATC codes, and approval dates
from multiple sources: PubChem, ChEMBL, FDA, and DrugBank.
"""

import argparse
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ============================================================================
# PubChem Functions
# ============================================================================

def get_retry_session():
    """
    Creates a requests Session that automatically retries on 503 errors
    with exponential backoff (wait time increases after each fail).
    """
    session = requests.Session()
    
    # Retry configuration
    retry_strategy = Retry(
        total=5,              # Maximum retry attempts
        backoff_factor=2,     # Wait 2s, then 4s, then 8s...
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these errors
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Thread-local storage for sessions (one session per thread)
_thread_local = threading.local()


def get_retry_session_local():
    """Get or create a thread-local session with retry logic (shared for all APIs)."""
    if not hasattr(_thread_local, 'retry_session'):
        _thread_local.retry_session = get_retry_session()
    return _thread_local.retry_session


def get_pubchem_data_all(compound_name):
    """
    Get all PubChem data in one optimized pass:
    - SMILES, CID, Parent CID, Parent Name, and ATC Level 4 codes
    
    Uses direct REST API calls for better performance and consistency.
    
    Args:
        compound_name: Name of the compound to search
        
    Returns:
        Tuple of (smiles, original_cid, parent_cid, parent_name, atc_level4_list)
        Returns (None, None, None, None, None) if not found or error occurs
    """
    if pd.isna(compound_name):
        return None, None, None, None, None
    
    try:
        # Get thread-local session with retry logic
        session = get_retry_session_local()
        
        # Step 1: Search by name to get CID
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/cids/JSON"
        search_resp = session.get(search_url, timeout=15)
        
        if search_resp.status_code != 200:
            return None, None, None, None, None
        
        search_data = search_resp.json()
        if 'IdentifierList' not in search_data or 'CID' not in search_data['IdentifierList']:
            return None, None, None, None, None
        
        original_cid = search_data['IdentifierList']['CID'][0]
        
        # Step 2: Get SMILES
        props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{original_cid}/property/CanonicalSMILES/JSON"
        props_resp = session.get(props_url, timeout=15)
        smiles = None
        if props_resp.status_code == 200:
            props_data = props_resp.json()
            if 'PropertyTable' in props_data and 'Properties' in props_data['PropertyTable']:
                if props_data['PropertyTable']['Properties']:
                    smiles = props_data['PropertyTable']['Properties'][0].get('ConnectivitySMILES')
        
        # Step 3: Get parent CID
        parent_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{original_cid}/cids/JSON?cids_type=parent"
        parent_resp = session.get(parent_url, timeout=15)
        parent_cid = original_cid  # Default to original if no parent found
        parent_name = None
        
        if parent_resp.status_code == 200:
            parent_data = parent_resp.json()
            if 'IdentifierList' in parent_data and 'CID' in parent_data['IdentifierList']:
                parent_cid = parent_data['IdentifierList']['CID'][0]
                
                # Step 4: Get parent name (only if parent is different)
                if parent_cid != original_cid:
                    parent_name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{parent_cid}/property/Title/JSON"
                    parent_name_resp = session.get(parent_name_url, timeout=15)
                    if parent_name_resp.status_code == 200:
                        parent_name_data = parent_name_resp.json()
                        if 'PropertyTable' in parent_name_data and 'Properties' in parent_name_data['PropertyTable']:
                            if parent_name_data['PropertyTable']['Properties']:
                                parent_name = parent_name_data['PropertyTable']['Properties'][0].get('Title')
        
        # Step 5: Get ATC codes for parent
        atc_codes = None
        atc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{parent_cid}/JSON?heading=ATC+Code"
        atc_resp = session.get(atc_url, timeout=15)
        
        if atc_resp.status_code == 200:
            try:
                atc_data = atc_resp.json()
                raw_list = []
                sections = atc_data["Record"]["Section"][0]["Section"][0]["Information"]
                for info in sections:
                    if "Value" in info and "StringWithMarkup" in info["Value"]:
                        for entry in info["Value"]["StringWithMarkup"]:
                            raw_list.append(entry["String"])
                
                # Clean and truncate to Level 4
                atc_set = set()
                for entry in raw_list:
                    code_part = re.split(r"[\s\-]", entry)[0]
                    if code_part.startswith("Q"):  # Skip veterinary codes
                        continue
                    if len(code_part) >= 5:
                        l4_code = code_part[:5]
                        if re.match(r"^[A-Z]\d{2}[A-Z]{2}$", l4_code):
                            atc_set.add(l4_code)
                
                if atc_set:
                    atc_codes = sorted(list(atc_set))
            except (KeyError, IndexError, TypeError):
                pass
        
        return smiles, original_cid, parent_cid, parent_name, atc_codes
        
    except Exception:
        # Silently fail - errors are expected for some compounds
        return None, None, None, None, None


# ============================================================================
# ChEMBL Functions
# ============================================================================

def get_chembl_data(drug_name):
    """
    Get ChEMBL data: approval year, SMILES, molecule type, and ATC codes.
    
    Args:
        drug_name: Name of the drug to search
        
    Returns:
        Tuple of (first_approval, canonical_smiles, molecule_type, atc_codes)
        Returns (None, None, None, None) if not found or error occurs
    """
    if pd.isna(drug_name):
        return None, None, None, None
    
    try:
        # Get thread-local session with retry logic
        session = get_retry_session_local()
        
        url = f'https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name}&format=json'
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'molecules' in data and len(data['molecules']) > 0:
                molecule = data['molecules'][0]
                
                # Get first approval year
                first_approval = molecule.get('first_approval')
                
                # Get molecule type
                molecule_type = molecule.get('molecule_type')
                
                # Get canonical SMILES
                canonical_smiles = None
                if 'molecule_structures' in molecule and molecule['molecule_structures']:
                    canonical_smiles = molecule['molecule_structures'].get('canonical_smiles')
                
                # Get ATC codes (requires second call using molecule_chembl_id)
                atc_codes = None
                chembl_id = molecule.get('molecule_chembl_id')
                if chembl_id:
                    try:
                        detail_url = f'https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json'
                        detail_resp = session.get(detail_url, timeout=10)
                        if detail_resp.status_code == 200:
                            detail_data = detail_resp.json()
                            atc_list = detail_data.get('atc_classifications')
                            if atc_list:
                                atc_codes = ';'.join(atc_list)
                    except Exception:
                        atc_codes = None
                
                return first_approval, canonical_smiles, molecule_type, atc_codes
        
        return None, None, None, None
    except Exception:
        return None, None, None, None


# ============================================================================
# FDA Functions
# ============================================================================

def get_fda_approval_from_file(drug_name, fda_products_df):
    """
    Get FDA approval date from local products.txt file using case-insensitive matching.
    First tries Trade_Name, then falls back to Ingredient if no match.
    
    Args:
        drug_name: Name of the drug to search
        fda_products_df: DataFrame containing FDA products data
        
    Returns:
        Approval date as string in format 'YYYYMMDD', or None if not found
    """
    if pd.isna(drug_name) or fda_products_df is None or len(fda_products_df) == 0:
        return None
    
    try:
        drug_name_lower = str(drug_name).lower().strip()
        
        # Try Trade_Name first
        if 'Trade_Name' in fda_products_df.columns:
            matches = fda_products_df[
                fda_products_df['Trade_Name'].notna() & 
                (fda_products_df['Trade_Name'].str.lower() == drug_name_lower)
            ]
            
            if len(matches) > 0 and 'Approval_Date' in matches.columns:
                dates = pd.to_datetime(matches['Approval_Date'], format='%b %d, %Y', errors='coerce').dropna()
                if len(dates) > 0:
                    return dates.min().strftime('%Y%m%d')
        
        # If no match in Trade_Name, try Ingredient
        if len(matches) == 0 and 'Ingredient' in fda_products_df.columns:
            matches = fda_products_df[
                fda_products_df['Ingredient'].notna() & 
                (fda_products_df['Ingredient'].str.lower() == drug_name_lower)
            ]
            
            if len(matches) > 0 and 'Approval_Date' in matches.columns:
                dates = pd.to_datetime(matches['Approval_Date'], format='%b %d, %Y', errors='coerce').dropna()
                if len(dates) > 0:
                    return dates.min().strftime('%Y%m%d')
        
        return None
    except Exception:
        return None


def get_fda_approval_date(drug_name):
    """
    Get FDA approval date from FDA API (fallback if local file not available).
    Tries brand name search first, then generic/active ingredient name.
    
    Args:
        drug_name: Name of the drug to search
        
    Returns:
        Approval date as string, or None if not found
    """
    if pd.isna(drug_name):
        return None
    
    try:
        # Try brand name search
        url = f'https://api.fda.gov/drug/drugsfda.json?search=products.brand_name:"{drug_name}"&limit=1'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                submissions = data['results'][0].get('submissions', [])
                for s in submissions:
                    if s.get('submission_type') == 'ORIG' and s.get('submission_status') == 'AP':
                        return s.get('submission_status_date')
        
        # Try generic/active ingredient name
        url = f'https://api.fda.gov/drug/drugsfda.json?search=products.active_ingredients.name:"{drug_name}"&limit=1'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                submissions = data['results'][0].get('submissions', [])
                for s in submissions:
                    if s.get('submission_type') == 'ORIG':
                        return s.get('submission_status_date')
        
        return None
    except Exception:
        return None


# ============================================================================
# DrugBank Functions
# ============================================================================

def match_drugbank_atc(compound_name, drugbank_df):
    """
    Match compound name with DrugBank and return ATC codes.
    Uses case-insensitive matching on the 'name' column.
    
    Args:
        compound_name: Name of the compound to match
        drugbank_df: DataFrame with DrugBank data (must have 'name' and 'atc_codes' columns)
    
    Returns:
        ATC codes as a string (comma-separated if multiple), or None if not found
    """
    if pd.isna(compound_name) or drugbank_df is None or len(drugbank_df) == 0:
        return None
    
    try:
        compound_name_lower = str(compound_name).lower().strip()
        
        # Try exact case-insensitive match
        matches = drugbank_df[
            drugbank_df['name'].notna() & 
            (drugbank_df['name'].str.lower() == compound_name_lower)
        ]
        
        if len(matches) > 0 and 'atc_codes' in matches.columns:
            # Get first match's ATC codes
            atc_value = matches.iloc[0]['atc_codes']
            
            if pd.isna(atc_value):
                return None
            
            # Handle different formats: string, list, or semicolon-separated
            if isinstance(atc_value, str):
                # Clean up the string (remove brackets if present, handle semicolons/commas)
                atc_value = atc_value.strip()
                if atc_value.startswith('[') and atc_value.endswith(']'):
                    atc_value = atc_value[1:-1]
                atc_value = atc_value.replace(';', ',')
                atc_codes = [code.strip() for code in atc_value.split(',') if code.strip()]
                if atc_codes:
                    return ', '.join(atc_codes)
            elif isinstance(atc_value, list):
                atc_codes = [str(code).strip() for code in atc_value if code and str(code).strip()]
                if atc_codes:
                    return ', '.join(atc_codes)
        
        return None
    except Exception:
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def process_parallel(func, items, desc, num_workers):
    """
    Process items in parallel using ThreadPoolExecutor.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        desc: Description for progress bar
        num_workers: Number of parallel workers
        
    Returns:
        List of results in the same order as items
    """
    results = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None
                pbar.update(1)
    
    return results


# ============================================================================
# Main Processing Function
# ============================================================================

def process_dilirank_v2(input_file, output_dir, num_workers, fda_products_file=None, drugbank_file=None):
    """
    Process DILIRank v2.0 data by fetching SMILES, CIDs, ATC codes, and approval dates
    from multiple sources: PubChem, ChEMBL, FDA, and DrugBank.
    
    Args:
        input_file: Path to input Excel file
        output_dir: Directory to save output files
        num_workers: Number of parallel workers (required)
        fda_products_file: Optional path to FDA products.txt file (uses local file instead of API if provided)
        drugbank_file: Optional path to DrugBank TSV file (matches ATC codes from DrugBank)
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    if num_workers is None:
        raise ValueError("num_workers must be specified")
    
    # Load input data
    df = pd.read_excel(input_file, header=1)
    compound_names = df['CompoundName'].tolist()
    
    # ========================================================================
    # Step 1: Fetch PubChem data (SMILES, CIDs, Parent CID, Parent Name, ATC)
    # ========================================================================
    print("\nFetching PubChem data (SMILES, CIDs, Parent CIDs, Parent Names, ATC codes)...")
    pubchem_results = process_parallel(get_pubchem_data_all, compound_names, "PubChem data (all)", num_workers)
    df['PubChem_SMILES'] = [r[0] if r is not None else None for r in pubchem_results]
    df['Orig CID'] = [r[1] if r is not None else None for r in pubchem_results]
    df['Parent CID'] = [r[2] if r is not None else None for r in pubchem_results]
    df['Parent Name'] = [r[3] if r is not None else None for r in pubchem_results]
    df['Parent ATC (Level 4)'] = [
        ", ".join(r[4]) if (r is not None and isinstance(r[4], list)) else None
        for r in pubchem_results
    ]
    
    # ========================================================================
    # Step 2: Fetch ChEMBL data (approval year, SMILES, molecule type, ATC)
    # ========================================================================
    print("\nFetching ChEMBL data...")
    chembl_results = process_parallel(get_chembl_data, compound_names, "ChEMBL data", num_workers)
    df['ChEMBL_First_Approval'] = [r[0] for r in chembl_results]
    df['ChEMBL_SMILES'] = [r[1] for r in chembl_results]
    df['ChEMBL_Molecule_Type'] = [r[2] for r in chembl_results]
    df['ChEMBL_ATC_Codes'] = [r[3] for r in chembl_results]
    
    # ========================================================================
    # Step 3: Fetch FDA approval dates
    # ========================================================================
    fda_products_df = None
    if fda_products_file and os.path.exists(fda_products_file):
        print(f"\nLoading FDA products file: {fda_products_file}")
        try:
            fda_products_df = pd.read_csv(fda_products_file, sep='~', encoding='utf-8', low_memory=False)
            print(f"Loaded {len(fda_products_df)} FDA product records")
        except Exception as e:
            print(f"Warning: Failed to load FDA products file: {e}")
            print("Falling back to API...")
    
    if fda_products_df is not None:
        def get_fda_date(name):
            return get_fda_approval_from_file(name, fda_products_df)
        df['FDA_Approval_Date'] = process_parallel(get_fda_date, compound_names, "FDA approval dates (from file)", num_workers)
    else:
        df['FDA_Approval_Date'] = process_parallel(get_fda_approval_date, compound_names, "FDA approval dates (API)", num_workers)
    
    df['FDA_Approval_Year'] = pd.to_datetime(df['FDA_Approval_Date'], format='%Y%m%d', errors='coerce').dt.year
    
    # ========================================================================
    # Step 4: Match DrugBank ATC codes (use Parent Name if available)
    # ========================================================================
    if drugbank_file and os.path.exists(drugbank_file):
        print(f"\nLoading DrugBank file: {drugbank_file}")
        try:
            drugbank_df = pd.read_csv(drugbank_file, sep='\t', encoding='utf-8', low_memory=False)
            print(f"Loaded {len(drugbank_df)} DrugBank records")
            
            print("Matching DrugBank ATC codes (using Parent Name when available)...")
            
            def match_with_parent_name(row):
                """Try parent name first, then fall back to original compound name."""
                parent_name = row.get('Parent Name') if 'Parent Name' in row else None
                compound_name = row.get('CompoundName')
                
                if parent_name and pd.notna(parent_name):
                    result = match_drugbank_atc(parent_name, drugbank_df)
                    if result is not None:
                        return result
                
                return match_drugbank_atc(compound_name, drugbank_df)
            
            df['DrugBank_ATC_Codes'] = df.apply(match_with_parent_name, axis=1)
            
            matched_count = df['DrugBank_ATC_Codes'].notna().sum()
            parent_matched = df[df['Parent Name'].notna() & df['DrugBank_ATC_Codes'].notna()].shape[0]
            print(f"Matched DrugBank ATC codes for {matched_count}/{len(df)} compounds ({matched_count/len(df)*100:.1f}%)")
            print(f"  - {parent_matched} matches using Parent Name")
            print(f"  - {matched_count - parent_matched} matches using original Compound Name")
        except Exception as e:
            print(f"Warning: Failed to load DrugBank file: {e}")
            df['DrugBank_ATC_Codes'] = None
    else:
        df['DrugBank_ATC_Codes'] = None
    
    # ========================================================================
    # Save results and print summary
    # ========================================================================
    output_path = os.path.join(output_dir, 'DILIrank_2.0_complete.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(df)} compounds processed")
    print(f"{'='*60}")
    print(f"  PubChem SMILES: {df['PubChem_SMILES'].notna().sum()}/{len(df)} ({df['PubChem_SMILES'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  ChEMBL SMILES: {df['ChEMBL_SMILES'].notna().sum()}/{len(df)} ({df['ChEMBL_SMILES'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  ChEMBL Approval: {df['ChEMBL_First_Approval'].notna().sum()}/{len(df)} ({df['ChEMBL_First_Approval'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  FDA Approval: {df['FDA_Approval_Date'].notna().sum()}/{len(df)} ({df['FDA_Approval_Date'].notna().sum()/len(df)*100:.1f}%)")
    if 'DrugBank_ATC_Codes' in df.columns:
        print(f"  DrugBank ATC: {df['DrugBank_ATC_Codes'].notna().sum()}/{len(df)} ({df['DrugBank_ATC_Codes'].notna().sum()/len(df)*100:.1f}%)")
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DILIRank v2.0 data - fetch SMILES from PubChem/ChEMBL and approval dates from ChEMBL/FDA"
    )
    parser.add_argument("--input_file", "-i", type=str, required=True,
                       help="Path to input Excel file (DILIRank_raw.xlsx)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Output directory for processed CSV files")
    parser.add_argument("--num_workers", "-n", type=int, required=True,
                       help="Number of parallel workers (required)")
    parser.add_argument("--fda_products_file", "-f", type=str, default=None,
                       help="Optional path to FDA products.txt file (uses local file instead of API if provided)")
    parser.add_argument("--drugbank_file", "-d", type=str, default=None,
                       help="Optional path to DrugBank TSV file (matches ATC codes from DrugBank)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    try:
        process_dilirank_v2(args.input_file, args.output_dir, args.num_workers, args.fda_products_file, args.drugbank_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
