"""
trial_analysis_helper.py

Clinical Trial Analysis Utility Functions

This module provides utilities for analyzing clinical trial data to identify
molecular targets and classify trials by mechanism of action (MOA) and 
innovation status.
"""

import pandas as pd
import re
from typing import Tuple, Dict
from collections import Counter


def load_trial_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if verbose:
        print(f"Loaded {len(df)} clinical trials from {filepath}")
    return df


def is_valid_target(target: str) -> bool:
    """
    Validate if a discovered target is likely a real molecular target. Filters out common noise patterns like generic pharmaceutical terms.
    
    Args:
        target: Candidate target name to validate
        
    Returns:
        True if target passes validation checks
    """
    target = target.upper().strip()
    
    # Common non-target words
    noise_words = {
        'BODY', 'INJECTION', 'MONOCLONAL', 'HUMANIZED', 'RECOMBINANT',
        'ANTIBODY', 'PROTEIN', 'THERAPY', 'TREATMENT', 'DRUG', 'AGENT',
        'ACTIVITY', 'TARGETING', 'AGAINST', 'FUSION', 'RECEPTOR ALPHA',
        'CHARACTERISTICS', 'AND', 'THE', 'OF', 'IN', 'ON', 'AT'
    }
    
    if target in noise_words:
        return False
    
    # Filter if mostly noise
    words = target.split()
    if len(words) > 1:
        noise_count = sum(1 for word in words if word in noise_words)
        if noise_count >= len(words) - 1:
            return False
    
    # Validate length
    if len(target) < 2 or len(target) > 30:
        return False
    
    # Common biological target patterns
    valid_patterns = [
        r'^IL[-\s]?\d+',       # Interleukins: IL-1, IL-4, IL-17
        r'^INTERLEUKIN[-\s]?\d+', # INTERLEUKIN-4 RECEPTOR
        r'^TNF',               # Tumor Necrosis Factor
        r'^TUMOR NECROSIS FACTOR',
        r'^PD[-\s]?[L1]?\d*',  # PD-1, PD-L1
        r'^PROGRAMMED DEATH',
        r'^CD\d+',             # CD markers: CD20, CD38
        r'^HER[-\s]?\d',       # HER2
        r'^VEGF',              # VEGF
        r'^VASCULAR ENDOTHELIAL'
        r'^EGFR',              # EGFR
        r'^EPIDERMAL GROWTH FACTOR',
        r'^HUMAN EPIDERMAL',
        r'^BCMA', r'^CTLA', r'^JAK', r'^BTK',
        r'^PCSK\d', r'^CGRP', r'^GLP', r'^RANKL',
        r'ERYTHROPOIETIN',
        r'^BDCA\d+',   # BDCA2, BDCA3, BDCA4
        r'^TACI\b',    # TACI
        r'^BAFF',      # BAFF, BAFF-R
        r'^APRIL\b',   # APRIL
        r'^TIGIT\b',   # TIGIT
        r'^LAG[-\s]?\d+', # LAG-3
    ]
    
    return any(re.match(pattern, target) for pattern in valid_patterns)


def discover_molecular_targets(df: pd.DataFrame, 
                               min_frequency: int = 2,
                               verbose: bool = True) -> Dict[str, str]:
    """
    Discover molecular targets from trial descriptions.
    
    Args:
        df: DataFrame containing trial data with 'title', 'objective', 
            'treatment_plan' columns
        min_frequency: Minimum number of mentions to consider target valid
        verbose: Whether to print discovery progress
        
    Returns:
        Dictionary mapping target names to their regex patterns
    """
    
    potential_targets = []
    
    for idx, row in df.iterrows():
        title = str(row.get('title', ''))
        objective = str(row.get('objective', ''))
        treatment_plan = str(row.get('treatment_plan', ''))
        combined_text = f"{title} {objective} {treatment_plan}"
        
        # Pattern 1: "anti-[TARGET]"
        for match in re.findall(
            r'anti[-\s]?([A-Za-z]{2,}[-\s]?\d*[A-Za-z]*(?:\s+(?:alpha|beta|receptor|α|β))?)',
            combined_text, re.IGNORECASE
        ):
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if is_valid_target(cleaned):
                potential_targets.append(cleaned.upper())
        
        # Pattern 2: "[TARGET] inhibitor/antagonist/blocker"
        for match in re.findall(
            r'([A-Za-z]{2,}[-\s]?\d*[A-Za-z]*(?:\s+(?:alpha|beta|α|β))?)\s+(?:inhibit(?:or|ion)|antagonist|blocker)',
            combined_text, re.IGNORECASE
        ):
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if is_valid_target(cleaned):
                potential_targets.append(cleaned.upper())
        
        # Pattern 3: "targeting [TARGET]"
        for match in re.findall(
            r'targeting\s+([A-Za-z]{2,}[-\s]?\d*[A-Za-z]*)',
            combined_text, re.IGNORECASE
        ):
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if is_valid_target(cleaned):
                potential_targets.append(cleaned.upper())
        
        # Pattern 4: Standalone biological codes
        context_pattern = r'\b((?:rh)?IL[-\s]?\d+[A-Za-z]*|TNF[-\s]?(?:alpha|α)?|PD[-\s]?[L]?\d|CD\d+|HER[-\s]?\d|VEGF[R]?|EGFR|BCMA|CTLA[-\s]?\d|JAK[-\s]?\d*|BTK|PCSK\d|CGRP|GLP[-\s]?\d|RANKL)\b'
        for match in re.findall(context_pattern, combined_text, re.IGNORECASE):
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if is_valid_target(cleaned):
                potential_targets.append(cleaned.upper())

        # Pattern 5: Full protein names
        protein_pattern = r'\b(erythropoietin|interferon|insulin|tumor necrosis factor[-\s]?(?:alpha|α)?|epidermal growth factor receptor|interleukin[-\s]?\d+)\b'
        for match in re.findall(protein_pattern, combined_text, re.IGNORECASE):
            cleaned = re.sub(r'\s+', ' ', match.strip())
            if is_valid_target(cleaned):
                potential_targets.append(cleaned.upper())
    
    # Count and filter by frequency
    target_counter = Counter()
    for target in potential_targets:
        normalized = re.sub(r'\s+', ' ', target.strip())
        normalized = re.sub(r'\s*-\s*', '-', normalized)
        target_counter[normalized] += 1
    
    # Build pattern dictionary
    discovered_patterns = {}
    
    if verbose:
        print(f"Discovered {len(target_counter)} unique candidates")
        print(f"Filtering for frequency >= {min_frequency}...\n")
        print("-" * 80)
    
    valid_targets = [(t, c) for t, c in target_counter.items() if c >= min_frequency]
    valid_targets.sort(key=lambda x: x[1], reverse=True)
    
    for target, count in valid_targets:
        if verbose:
            print(f"  {target:30s} mentioned {count:3d} times")
        pattern = re.escape(target).replace(r'\ ', r'[-\s]?').replace(r'\-', r'[-\s]?')
        discovered_patterns[target] = pattern
    
    if verbose:
        print(f"\nTotal targets discovered: {len(discovered_patterns)}")
    
    return discovered_patterns


def extract_drug_name(title: str, treatment_plan: str) -> str:
    """
    Extract drug name from trial title or treatment plan.
    
    Args:
        title: Trial title
        treatment_plan: Treatment plan description
        
    Returns:
        Extracted drug name or "Not specified"
    """
    # Drug codes (e.g., ABC-123)
    code_match = re.search(r'\b([A-Z]{2,}[-\s]?\d{2,})\b', title)
    
    # Monoclonal antibodies (ending in -mab)
    mab_match = re.search(r'\b([A-Z][a-z]+mab)\b', title, re.IGNORECASE)
    
    # From treatment plan
    treatment_match = None
    if 'drug:' in treatment_plan.lower():
        treatment_match = re.search(r'Drug:\s*([^\n,;]+)', treatment_plan, re.IGNORECASE)
    
    if mab_match:
        return mab_match.group(1)
    elif code_match:
        return code_match.group(1)
    elif treatment_match:
        name = treatment_match.group(1).strip()
        name = re.sub(r'^(Placebo|Experimental:|Active Comparator:)\s*', '', 
                     name, flags=re.IGNORECASE)
        return name.strip()
    
    return "Not specified"


def extract_moa(title: str, objective: str, treatment_plan: str, 
                target_patterns: Dict[str, str]) -> Tuple[str, str]:
    """
    Extract mechanism of action from trial text.
    
    Args:
        title: Trial title
        objective: Trial objective
        treatment_plan: Treatment plan description
        target_patterns: Dictionary of discovered target patterns
        
    Returns:
        Tuple of (molecular_target, mechanism_description)
    """
    combined_text = f"{title} {objective} {treatment_plan}".lower()
    
    molecular_target = "Unknown"
    mechanism = "Unknown"
    
    # Find discovered target
    for target, pattern in target_patterns.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            molecular_target = target
            break
    
    # Determine mechanism type from keywords
    if re.search(r'\b(?:monoclonal\s+)?antibod(?:y|ies)\b|\bmab\b', combined_text):
        mechanism = "Bispecific monoclonal antibody" if re.search(r'\bbispecific\b', combined_text) else "Monoclonal antibody"
        
        if molecular_target != "Unknown":
            if re.search(r'\binhibit', combined_text):
                mechanism = f"{molecular_target} inhibitor - {mechanism}"
            elif re.search(r'\bantagonist\b', combined_text):
                mechanism = f"{molecular_target} antagonist - {mechanism}"
            elif re.search(r'\bblock', combined_text):
                mechanism = f"{molecular_target} blocker - {mechanism}"
            else:
                mechanism = f"{molecular_target} targeting - {mechanism}"
                
    elif re.search(r'\bcar[-\s]?t\b|chimeric antigen receptor', combined_text):
        mechanism = f"{molecular_target} CAR-T cell therapy" if molecular_target != "Unknown" else "CAR-T cell therapy"
    
    elif re.search(r'\bkinase inhibitor\b', combined_text):
        mechanism = f"{molecular_target} kinase inhibitor" if molecular_target != "Unknown" else "Kinase inhibitor"
    
    elif re.search(r'\btyrosine kinase inhibitor\b|\btki\b', combined_text):
        mechanism = f"{molecular_target} TKI" if molecular_target != "Unknown" else "Tyrosine kinase inhibitor"
    
    elif re.search(r'\bfusion protein\b', combined_text):
        mechanism = f"{molecular_target} fusion protein" if molecular_target != "Unknown" else "Fusion protein"
    
    elif re.search(r'\bagonist\b', combined_text):
        mechanism = f"{molecular_target} receptor agonist" if molecular_target != "Unknown" else "Receptor agonist"
    
    elif re.search(r'\bsmall molecule\b', combined_text):
        mechanism = f"{molecular_target} small molecule" if molecular_target != "Unknown" else "Small molecule inhibitor"
    
    elif re.search(r'\bvaccine\b', combined_text):
        mechanism = "Vaccine"

    elif re.search(r'\breceptor\s+antagonist\b', combined_text):
        mechanism = f"{molecular_target} receptor antagonist"
    
    elif re.search(r'\brecombinant\b.*\bprotein\b', combined_text):
        mechanism = f"Recombinant {molecular_target}" if molecular_target != "Unknown" else "Recombinant protein"
    
    return molecular_target, mechanism


def classify_innovation_status(title: str, objective: str, treatment_plan: str) -> str:
    """
    Determine if trial is for an innovative drug or biosimilar.
    
    Args:
        title: Trial title
        objective: Trial objective  
        treatment_plan: Treatment plan description
        
    Returns:
        "Innovative" or "Biosimilar"
    """
    combined_text = f"{title} {objective} {treatment_plan}".lower()
    
    # Biosimilar indicators
    biosimilar_keywords = [
        r'\bbiosimilar\b', r'\bnon[-\s]?inferiority\b', r'\bequivalence\b',
        r'\btherapeutic equivalence\b', r'\bbioequivalence\b', r'\breference product\b',
    ]
    
    for keyword in biosimilar_keywords:
        if re.search(keyword, combined_text):
            return "Biosimilar"
    
    # Innovative indicators
    innovative_keywords = [
        r'\bversus placebo\b', r'\bplacebo[-\s]controlled\b',
        r'\bevaluate (?:the )?(?:safety|efficacy)\b',
        r'\bnovel\b', r'\bfirst[-\s]in[-\s]human\b',
        r'\bdose[-\s]escalation\b', r'\bsingle.*ascending.*dose\b',
    ]
    
    for keyword in innovative_keywords:
        if re.search(keyword, combined_text):
            return "Innovative"
    
    return "Innovative"


def categorize_trial(molecular_target: str, mechanism: str, innovation_status: str) -> str:
    """
    Classify category based on MOA and innovation status.
    
    Args:
        molecular_target: Identified molecular target
        mechanism: Mechanism description
        innovation_status: "Innovative" or "Biosimilar"
        
    Returns:
        Category string combining mechanism type and innovation status
    """
    if molecular_target == "Unknown" or mechanism == "Unknown":
        base_category = "Other/Undefined"
    else:
        mechanism_lower = mechanism.lower()
        
        if "antibody" in mechanism_lower:
            base_category = "Antibody-based therapy"
        elif "kinase inhibitor" in mechanism_lower or "tki" in mechanism_lower:
            base_category = "Kinase inhibitor"
        elif "car-t" in mechanism_lower:
            base_category = "Cell therapy"
        elif "fusion protein" in mechanism_lower:
            base_category = "Fusion protein"
        elif "vaccine" in mechanism_lower:
            base_category = "Vaccine"
        elif "agonist" in mechanism_lower:
            base_category = "Receptor agonist"
        elif "recombinant" in mechanism_lower:
            base_category = "Recombinant protein"
        else:
            base_category = "Other targeted therapy"
    
    return f"{base_category} - {innovation_status}"


def analyze_single_trial(row: pd.Series, target_patterns: Dict[str, str]) -> Dict:
    """
    Analyze a single trial and return classification results.
    
    Args:
        row: DataFrame row containing trial data
        target_patterns: Dictionary of discovered target patterns
        
    Returns:
        Dictionary with trial classification results
    """
    title = str(row.get('title', ''))
    objective = str(row.get('objective', ''))
    treatment_plan = str(row.get('treatment_plan', ''))
    
    drug_name = extract_drug_name(title, treatment_plan)
    molecular_target, mechanism = extract_moa(title, objective, treatment_plan, target_patterns)
    innovation_status = classify_innovation_status(title, objective, treatment_plan)
    category = categorize_trial(molecular_target, mechanism, innovation_status)
    
    return {
        'Trial Title': title,
        'Drug Name': drug_name,
        'MOA': f"{molecular_target} - {mechanism}",
        'Innovation': innovation_status,
        'Category': category
    }


def analyze_all_trials(df: pd.DataFrame, 
                       target_patterns: Dict[str, str],
                       verbose: bool = True) -> pd.DataFrame:
    """
    Analyze all trials using discovered target patterns.
    
    Args:
        df: DataFrame containing trial data
        target_patterns: Dictionary of discovered target patterns
        verbose: Whether to print progress
        
    Returns:
        DataFrame with analysis results for all trials
    """
    
    results = []
    
    for idx, row in df.iterrows():
        result = analyze_single_trial(row, target_patterns)
        results.append(result)
        
        if verbose and (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} trials...")
    
    return pd.DataFrame(results)


def get_summary_statistics(results_df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics from analysis results.
    
    Args:
        results_df: DataFrame with analysis results
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_trials': len(results_df),
        'innovative_count': (results_df['Innovation'] == 'Innovative').sum(),
        'biosimilar_count': (results_df['Innovation'] == 'Biosimilar').sum(),
        'category_distribution': results_df['Category'].value_counts().to_dict(),
        'innovation_distribution': results_df['Innovation'].value_counts().to_dict()
    }


def save_results(results_df: pd.DataFrame, 
                 discovered_patterns: Dict[str, str],
                 output_excel: str):
    """
    Save analysis results to files.
    """
    # Save Excel
    results_df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✓ Results saved to: {output_excel}")
    