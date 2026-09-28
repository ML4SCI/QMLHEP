# processor_top.py
import os
import h5py
import hdf5plugin
import pickle
import numpy as np
import torch
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from src.utils.workspace import get_config, set_seed
from src.preprocessing.balance import balance_classes, split_into_subsets, resolve_sample_size

# ============================================================================
# ============================================================================
# Vectorized Jet Kinematics Engine (Calculations directly on raw NumPy arrays)
# ============================================================================
# ============================================================================

def _compute_physics_features(raw_matrix, config, scaler=None):
    """
    Core math transformation. Combines macro-physics and Pareto sub-structure 
    into a structured [N, m] matrix ready for KAN / Random Forest baselines. 
    N is the number of jets and m is the number of features (2 + 2 * n_particles).
    The last 4 features are the global jet four-momentum components (E, px, py, pz) and the 
    rest are the four-momentum particle-level features.
    The first two columns are global jet properties (scaled invariant mass and multiplicity),
    followed by pairs of columns for each particle: [dR_i, z_i]. The function optionally applies
    a cut in the invariant mass between config['mass_cut_lo'] and config['mass_cut_hi'] GeV,
    controlled by config['apply_mass_cut'] (default True); when False, all jets are kept.

    - dr: Geometric distance of the i-th particle from the jet axis in the (eta, phi) plane.
    - z: Fraction of the jet's transverse momentum carried by the i-th particle.

    Parameters:
    - raw_matrix: NumPy array of shape [N, 800] containing raw jet data.
    - config: Configuration dictionary containing parameters for processing.
    - scaler: Optional StandardScaler object for normalizing multiplicity.

    return
    - processed_matrix: NumPy array of shape [N, 2 + 2 * n_particles] containing processed features.
    - scaler: StandardScaler object used for multiplicity normalization.
    - mass_mask: Boolean array indicating which jets passed the invariant mass cut.
    """
    # -------------------------------------------------------------------------
    # 1. Global Masking and Initial Geometric Reduction
    # -------------------------------------------------------------------------
    E_jet   = np.sum(raw_matrix[:, 0:800:4], axis=1)
    px_jet  = np.sum(raw_matrix[:, 1:800:4], axis=1)
    py_jet  = np.sum(raw_matrix[:, 2:800:4], axis=1)
    pz_jet  = np.sum(raw_matrix[:, 3:800:4], axis=1)

    pt_jet  = np.sqrt(px_jet**2 + py_jet**2)
    eta_jet = -np.log(np.tan(np.arctan2(pt_jet, pz_jet + 1e-12) / 2.0) + 1e-12)
    phi_jet = np.arctan2(py_jet, px_jet + 1e-12)

    mass_sq = np.clip(E_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2), 0.0, None)
    invariant_mass = np.sqrt(mass_sq)

    del px_jet, py_jet, pz_jet, E_jet, pt_jet, mass_sq
    gc.collect()

    # Global kinematic filter for the Jet. Controlled by config["apply_mass_cut"];
    # when False, all jets pass (no invariant-mass cut is applied).
    apply_mass_cut = config.get("apply_mass_cut", True)
    mass_cut_lo = config.get("mass_cut_lo", 145.0)
    mass_cut_hi = config.get("mass_cut_hi", 205.0)
    if apply_mass_cut:
        mass_mask = (invariant_mass > mass_cut_lo) & (invariant_mass < mass_cut_hi)
    else:
        mass_mask = np.ones_like(invariant_mass, dtype=bool)
    invariant_mass = invariant_mass[mass_mask]
    raw_matrix = raw_matrix[mass_mask]
    eta_jet = eta_jet[mass_mask]
    phi_jet = phi_jet[mass_mask]
    
    # Extract the raw global multiplicity before truncating the matrix
    multiplicity = np.sum(raw_matrix[:, 0:800:4] > 1e-3, axis=1)
    n_events = raw_matrix.shape[0]

    # Controlled reduction to the optimal window of 100 particles to save RAM
    n_initial_particles = 100
    
    # Extract optimized local blocks in memory
    px_block = raw_matrix[:, 1:n_initial_particles*4:4].copy()
    py_block = raw_matrix[:, 2:n_initial_particles*4:4].copy()
    pz_block = raw_matrix[:, 3:n_initial_particles*4:4].copy()

    # -------------------------------------------------------------------------
    # STEP 1 AND 2: GHOST IDENTIFICATION AND GEOMETRIC CLONING
    # -------------------------------------------------------------------------
    pt_block = np.sqrt(px_block**2 + py_block**2)
    is_real = pt_block >= 1e-3

    # Controlled initialization along the jet axis to mitigate artificial teleportation
    eta_i = np.repeat(eta_jet[:, None], n_initial_particles, axis=1)
    phi_i = np.repeat(phi_jet[:, None], n_initial_particles, axis=1)

    theta_real = np.arctan2(pt_block[is_real], pz_block[is_real] + 1e-12)
    eta_i[is_real] = -np.log(np.tan(theta_real / 2.0) + 1e-12)
    phi_i[is_real] = np.arctan2(py_block[is_real], px_block[is_real] + 1e-12)
    
    del pz_block, px_block, py_block, theta_real
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 3: RELATIVE MINKOWSKI DISTANCE WITHOUT ASYMMETRIC ARTIFACTS
    # -------------------------------------------------------------------------
    d_eta_i = eta_i - eta_jet[:, None]
    d_phi_i = np.arctan2(np.sin(phi_i - phi_jet[:, None]), np.cos(phi_i - phi_jet[:, None]))
    d_R = np.sqrt(d_eta_i**2 + d_phi_i**2)

    del phi_i, d_eta_i, d_phi_i
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 4: IN-PLACE PHYSICAL VIABILITY FILTER (EXCLUSIVE p_T SPRAY)
    # -------------------------------------------------------------------------
    # Filter: Outside the cone (> 0.80) OR outside the calorimeter acceptance (|eta| >= 3.0)
    unphysical_mask = (~is_real) | (d_R > 0.80) | (np.abs(eta_i) >= 3.0)
    pt_block[unphysical_mask] = 0.0

    del eta_i, is_real, unphysical_mask
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 5: HIGH-SPEED SYNCHRONIZED SORTING
    # -------------------------------------------------------------------------
    sorted_indices = np.argsort(-pt_block, axis=1)
    row_indices = np.arange(n_events)[:, None]
    
    pt_block = pt_block[row_indices, sorted_indices]
    d_R = d_R[row_indices, sorted_indices]

    del sorted_indices, row_indices
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 6: DYNAMIC AMPUTATION BY PARETO (80% CONE ENERGY THRESHOLD)
    # -------------------------------------------------------------------------
    pt_cumsum = np.cumsum(pt_block, axis=1)
    pt_total_jet = pt_cumsum[:, -1]
    
    pt_total_cone_safe = np.where(pt_total_jet <= 0, 1.0, pt_total_jet)
    pt_frac_cumsum = pt_cumsum / pt_total_cone_safe[:, None]
    
    idx_80 = np.argmax(pt_frac_cumsum >= 0.80, axis=1) + 1
    n_particles = int(np.percentile(idx_80, 90))
    n_particles = max(n_particles, 5) 

    pt_block = pt_block[:, :n_particles]
    d_R = d_R[:, :n_particles]

    # CCompute the effective energy fraction z_effective relative to the purified cone
    sum_pt_final = np.sum(pt_block, axis=1)
    sum_pt_final_safe = np.where(sum_pt_final <= 0, 1.0, sum_pt_final)
    z_effective = pt_block / sum_pt_final_safe[:, None]

    # Force d_R to 0.0 in the remaining empty channels from legitimate padding
    d_R[pt_block <= 0.0] = 0.0

    del pt_block, pt_cumsum, pt_frac_cumsum, sum_pt_final, sum_pt_final_safe, idx_80
    gc.collect()

    # -------------------------------------------------------------------------
    # PHYSICAL VALIDATION LOGGING
    # -------------------------------------------------------------------------
    '''
    if (~is_ghost).any():
        print(f"eta_i range (processed): {eta_i[:, :n_particles][~is_ghost].min():.2f} \
        to {eta_i[:, :n_particles][~is_ghost].max():.2f}")
    
    valid_dR = d_R[d_R > 0]
    if valid_dR.size > 0:
        print(f"dR range (real particles): {valid_dR.min():.2f} to {valid_dR.max():.2f}")

    print("\n*--- DYNAMIC CONSTITUENT AUDIT (PARETO 80%) ---*")
    print(f"--> NUMBER OF CONSTITUENTS FIXED FOR THIS BATCH: {n_particles}")
    print(f"Average individual index to capture 80% of p_T: {idx_80.mean():.2f}")
    print(f"Minimum number of real particles in final window: {real_particles_per_jet.min()}")
    print(f"Maximum number of real particles in final window: {real_particles_per_jet.max()}")
    print(f"Average number of real particles in final window: {real_particles_per_jet.mean():.2f}")
    print("*-----------------------------------------------------------*\n")
    
    sum_z_test = np.sum(z_effective, axis=1)
    active_jets = sum_z_test > 0
    if active_jets.any():
        is_normalized = np.allclose(sum_z_test[active_jets], 1.0, atol=1e-3)
        print(f"Relative Normalization Test (sum(z_i) == 1.0): {is_normalized}")
    '''

    # -------------------------------------------------------------------------
    # ASYMPTOTIC COMPRESSION AND GLOBAL VARIABLE NORMALIZATION
    # -------------------------------------------------------------------------
    # Invariant Mass: Log + RobustScaler + Asymptotic Tanh (Protects Outliers)
    m_log = np.log(invariant_mass + 1.0)
    del invariant_mass
    
    robust_scaler = RobustScaler()
    m_robust = robust_scaler.fit_transform(m_log.reshape(-1, 1)).flatten()
    m_scaled = np.tanh(m_robust)
    
    del m_log, m_robust
    gc.collect()

    # Multiplicity: StandardScaler + Strict 3 Sigma Clip
    # Preserves the real morphology of physical peaks without deforming tails
    if scaler is None:
        scaler = StandardScaler()
        M_robust = scaler.fit_transform(multiplicity.reshape(-1, 1)).flatten()
    else:
        M_robust = scaler.transform(multiplicity.reshape(-1, 1)).flatten()
        
    del multiplicity
    
    sigma_max = 3.0
    M_clipped = np.clip(M_robust, -sigma_max, sigma_max)
    M_scaled = M_clipped / sigma_max

    del M_robust, M_clipped
    gc.collect()

    # -------------------------------------------------------------------------
    # INTERLEAVED PACKING COLUMNS [N, 2 + 2 * n_particles]
    # -------------------------------------------------------------------------
    processed_matrix = np.zeros((n_events, 2 + 2 * n_particles), dtype=np.float32)
    processed_matrix[:, 0] = m_scaled
    processed_matrix[:, 1] = M_scaled
    
    processed_matrix[:, 2::2] = d_R
    processed_matrix[:, 3::2] = z_effective
    
    del d_R, z_effective, m_scaled, M_scaled
    gc.collect()
        
    return processed_matrix, scaler, mass_mask


# ============================================================================
# ============================================================================
# Load and preprocess data
# ============================================================================
# ============================================================================

def load_and_preprocess_data(data_dir, task, seed=42, force_process=False,
    balance=False, apply_mass_cut=None, full_dataset=False):
    """
    Processes separate train.h5, val.h5, and test.h5 files sequentially, then
    partitions each balanced split into n_subsets mutually disjoint, class-balanced
    chunks (the canonical partition -- built once, seed-independent, cached under
    config["canonical_cache_file"]). Returns only the seed % n_subsets-th subset's
    8-tuple: (X_train, y_train, X_val, y_val, X_test, y_test, X_train_sample, scaler).

    force_process=True is the ONLY path that (re)builds the canonical partition --
    reserved for scripts/run_preprocessing.py. Every other caller (the training
    scripts) must find an existing cache; a cache miss with force_process=False
    raises RuntimeError instead of silently building it, so a --seed run can only
    ever select a subset, never construct one.

    full_dataset=True forces apply_mass_cut=False and n_subsets=1: the entire dataset,
    unpartitioned (train/val/test stay separate splits). The regime is resolved by
    workspace.get_config, which also picks the canonical cache directory
    (data/processed/<task>/<cut>/<full|n{N}>/), so different regimes never share a cache.
    """
    config = get_config(task, seed, full_dataset=full_dataset, apply_mass_cut=apply_mass_cut)
    resolved_mass_cut = config["apply_mass_cut"]

    n_subsets = config["n_subsets"]
    subset_split_seed = config.get("subset_split_seed", 37)
    subset_id = config["subset_id"]

    DATA_DIR = Path(data_dir)
    canonical_dir = Path(config["canonical_data_dir"])
    canonical_dir.mkdir(parents=True, exist_ok=True)

    cache_file = Path(config["canonical_cache_file"])
    scaler_file = Path(config["canonical_scaler_path"])

    # --- STEP 1: CACHE SYSTEM DETECTOR ---
    if cache_file.exists() and not force_process:
        print(f"\n[CACHE DETECTED] Loading canonical {n_subsets}-way partition from: '{cache_file}'")
        try:
            cached_data = torch.load(cache_file)
            if cached_data.get("n_subsets") != n_subsets:
                raise RuntimeError(
                    f"Cached canonical partition at '{cache_file}' has "
                    f"n_subsets={cached_data.get('n_subsets')}, but hyperparams.py "
                    f"currently requests n_subsets={n_subsets}. Re-run "
                    f"scripts/run_preprocessing.py with force_process=True to "
                    f"intentionally rebuild the canonical partition."
                )
            if "apply_mass_cut" not in cached_data:
                # Caches built before this metadata existed: the regime is encoded in the
                # directory name (workspace.get_config), so trust it rather than rebuild.
                print(f"[CACHE] '{cache_file}' has no apply_mass_cut metadata (legacy cache); "
                      f"trusting its directory ({config['variant']}).")
            elif cached_data["apply_mass_cut"] != resolved_mass_cut:
                raise RuntimeError(
                    f"Cached canonical partition at '{cache_file}' was built with "
                    f"apply_mass_cut={cached_data['apply_mass_cut']}, but this run "
                    f"requests apply_mass_cut={resolved_mass_cut}. Rebuild it with "
                    f"scripts/run_preprocessing.py --force."
                )
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
            print(f">> Selecting subset {subset_id} (seed={seed} % n_subsets={n_subsets}).")
            set_seed(seed, purpose="run seed: subset selection")
            return (
                cached_data['X_train_subsets'][subset_id], cached_data['y_train_subsets'][subset_id],
                cached_data['X_val_subsets'][subset_id], cached_data['y_val_subsets'][subset_id],
                cached_data['X_test_subsets'][subset_id], cached_data['y_test_subsets'][subset_id],
                cached_data['X_train_sample_subsets'][subset_id], scaler
            )
        except RuntimeError:
            raise
        except Exception as e:
            print(f"CRITICAL cache error: {e}. Falling back to execution loops.")

    if not force_process:
        raise RuntimeError(
            f"[Preprocessing] Canonical {n_subsets}-way partition not found at "
            f"'{cache_file}'. Training scripts only ever SELECT an existing subset, "
            f"they never build it. Run scripts/run_preprocessing.py once first to "
            f"build the canonical partition."
        )

    # --- STEP 2: SEQUENTIAL PROCESS (Train, Val, Test) -- builds the canonical
    # partition. Uses subset_split_seed (NOT seed) so the partition is stable
    # regardless of which seed later selects a subset from it. ---
    set_seed(subset_split_seed, purpose="subset_split_seed: canonical partition build (hyperparams.py)")

    if task in ["top"]:
        raw_files = {
            "train": DATA_DIR / "train.h5",
            "val": DATA_DIR / "val.h5",
            "test": DATA_DIR / "test.h5"
            }
        print(f"Raw files for task '{task}':")
        for split, path in raw_files.items():
            print(f"  {split}: {path}")
    
    processed_tensors = {}
    scaler = None

    for split, file_path in raw_files.items():
        print("\n*---------------------------------------------------*")
        print(f"Loading and processing raw split: [{split.upper()}] from HDF5")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Missing mandatory TopTagging partition file: {file_path.name}")
            
        with h5py.File(file_path, "r") as f:
            # Reconstruct structured event tables
            f = f["table"]["table"] # Navigate to the nested group containing the data
            raw_matrix = f["values_block_0"][:]
            # Index 1 holds the categorical value (1: Top Signal, 0: QCD Background)
            raw_labels = f["values_block_1"][:, 1]

        print(f"Data chunk successfully mounted in RAM. Extracted shape: {raw_matrix.shape}")

        # DIAGNOSTIC BEFORE PROCESSING
        print(f"--> RAW on-disk label distribution for [{split.upper()}]:")
        classes, class_counts = np.unique(raw_labels, return_counts=True)
        for c, n in zip(classes, class_counts):
            print(f"    Class {c}: {n} events")

        print(f"DEBUG BEFORE: raw_matrix shape = {raw_matrix.shape}, raw_labels shape = {raw_labels.shape}")

        X_norm, split_scaler, mask = _compute_physics_features(raw_matrix, config, scaler=scaler)

        print(f"DEBUG AFTER: X_norm shape = {X_norm.shape}, mask True count = {np.sum(mask)}")
        print(f"DEBUG FILTERED LABELS: Zeros: {np.sum(raw_labels[mask] == 0)}, Ones: {np.sum(raw_labels[mask] == 1)}")

        if split == "train":
            scaler = split_scaler
            with open(scaler_file, "wb") as f:
                pickle.dump(split_scaler, f)
            print(f"Global scaler object saved to: '{scaler_file}'")

        if balance:
            # Balance classes to ~50/50 AFTER feature engineering (the scaler above
            # is fit on the full, unbalanced masked data; balancing only undersamples
            # rows of the already-engineered feature matrix and labels).
            X_masked = X_norm
            y_masked = raw_labels[mask]
            X_balanced, y_balanced = balance_classes(X_masked, y_masked)
            print(f"--> Balanced label distribution for [{split.upper()}]:")
            balanced_classes, balanced_counts = np.unique(y_balanced, return_counts=True)
            for c, n in zip(balanced_classes, balanced_counts):
                print(f"    Class {c}: {n} events")

        else:
            X_balanced = X_norm
            y_balanced = raw_labels[mask]

        # Partition the balanced pool into n_subsets mutually disjoint,
        # class-balanced chunks -- the canonical statistical-replicate partition.
        X_sub_list, y_sub_list = split_into_subsets(X_balanced, y_balanced, n_subsets)
        print(f"--> Partitioned [{split.upper()}] into {n_subsets} disjoint subsets "
              f"(sizes: {[len(x) for x in X_sub_list]})")

        processed_tensors[f"X_{split}_subsets"] = [torch.from_numpy(x).float() for x in X_sub_list]
        y_tensors = []
        for y_arr in y_sub_list:
            t = torch.from_numpy(y_arr).float()
            if t.ndim == 1:
                t = t.unsqueeze(1)  # Match required [N, 1] output dimension
            y_tensors.append(t)
        processed_tensors[f"y_{split}_subsets"] = y_tensors

        del raw_matrix, raw_labels, X_norm, X_balanced, y_balanced
        gc.collect()

    print("\n--- Final canonical disjoint partition built (Vectorized Slices Framework) ---")
    for split in ("train", "val", "test"):
        shapes = [tuple(x.shape) for x in processed_tensors[f"X_{split}_subsets"]]
        print(f"X_{split} subset shapes: {shapes}")

    # --- STEP 3: PER-SUBSET SYMBOLIC INTERPOLATION SAMPLE ---
    print(f"\nIsolating per-subset warm-up samples for high-speed symbolic KAN regressions...")
    sample_fraction = config.get("symbolic_sample_fraction", 0.05)
    sample_min_floor = config.get("symbolic_sample_min_floor", 500)
    sample_fallback_size = config.get("symbolic_sample_fallback_size", 5000)

    X_train_sample_subsets = []
    for k in range(n_subsets):
        Xk = processed_tensors["X_train_subsets"][k]
        sample_size = resolve_sample_size(
            len(Xk), fraction=sample_fraction,
            min_floor=sample_min_floor, fallback_size=sample_fallback_size
        )
        if len(Xk) > sample_size:
            random_indices = torch.randperm(len(Xk))[:sample_size]
            X_train_sample_subsets.append(Xk[random_indices])
        else:
            X_train_sample_subsets.append(Xk)
    print(f"Warm-up tensors isolated. Sizes: {[len(x) for x in X_train_sample_subsets]}")

    # --- STEP 4: CANONICAL PARTITION SERIALIZATION ---
    canonical_data = {
        'X_train_subsets': processed_tensors["X_train_subsets"],
        'y_train_subsets': processed_tensors["y_train_subsets"],
        'X_val_subsets':   processed_tensors["X_val_subsets"],
        'y_val_subsets':   processed_tensors["y_val_subsets"],
        'X_test_subsets':  processed_tensors["X_test_subsets"],
        'y_test_subsets':  processed_tensors["y_test_subsets"],
        'X_train_sample_subsets': X_train_sample_subsets,
        'n_subsets': n_subsets,
        'subset_split_seed': subset_split_seed,
        'apply_mass_cut': resolved_mass_cut,
        'balance': balance,
        'mass_cut_lo': config.get("mass_cut_lo", 145.0),
        'mass_cut_hi': config.get("mass_cut_hi", 205.0),
    }

    torch.save(canonical_data, cache_file)
    print(f"\n[CACHE WRITTEN] Canonical {n_subsets}-way partition saved to: '{cache_file}'")
    print(f"Inference metrics scaler object written into workspace folder structures.")

    # force_process=True is build-only (reserved for run_preprocessing*.py), whose
    # callers discard the returned subset -- selecting one via the run seed here
    # would have no effect, so it is not seeded/printed in this branch.
    return (
        canonical_data['X_train_subsets'][subset_id], canonical_data['y_train_subsets'][subset_id],
        canonical_data['X_val_subsets'][subset_id], canonical_data['y_val_subsets'][subset_id],
        canonical_data['X_test_subsets'][subset_id], canonical_data['y_test_subsets'][subset_id],
        canonical_data['X_train_sample_subsets'][subset_id], scaler
    )

def load_quantum_inputs(metadata_path, X_tensor):
    """
    Reads the surviving feature indices from the pruned classical model
    and filters the data tensor for the quantum architecture.
    """
    import json
    import torch

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    active_indices = metadata.get('active_input_indices', [])

    if not active_indices:
        raise ValueError("No active indices found in the metadata.")

    print(f"Filtering quantum dataset. Original dimensions: {X_tensor.shape[1]}")

    # Slice the tensor keeping only the columns for the active indices
    X_quantum = X_tensor[:, active_indices]

    print(f"New dimensions for QKAN: {X_quantum.shape[1]}")
    return X_quantum
