# src/architectures/extractor.py
import os
import torch
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.hep_kan import HEPKAN


class SymbolicWarmStartExtractor:
    """
    Warm-start extractor that preserves the STRUCTURE of the classical graph
    (sum nodes vs. multiplication nodes, and which edges feed into each one),
    instead of flattening the network into a flat list of active inputs.

    Index convention:
        - act_fun[l].mask[i][j]        -> [previous_node=i][raw_neuron=j]
        - symbolic_fun[l].mask[j][i]   -> [raw_neuron=j][previous_node=i]
    An edge (i -> j) in layer l is active if act_fun[l].mask[i][j] != 0.
    """

    basis = "chebyshev"

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_degree = self.config.get("chebyshev_max_degree", 4)
        self.dynamic_range_threshold = self.config.get("qkan_dynamic_range_threshold", 1e-3)

    # ------------------------------------------------------------------
    # Isolated evaluation of one edge
    # ------------------------------------------------------------------
    def _evaluate_isolated_edges(self, model, layer_index, input_index, output_index, x_vals):
        layer_width = model.width[layer_index]
        in_dim = layer_width[0] if isinstance(layer_width, list) else layer_width
        # Actual in_dim of the layer: use width_in to avoid assuming 'width''s format
        in_dim = int(model.width_in[layer_index])
        n = len(x_vals)

        x_zero = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var[:, input_index] = torch.tensor(x_vals, dtype=torch.float32).to(self.device)

        def layer_forward(x_in):
            try:
                # Reads the NUMERIC branch instead of the symbolic one
                numeric = model.act_fun[layer_index](x_in)
                x_out = numeric[0] if isinstance(numeric, tuple) else numeric
            except Exception:
                out_dim = int(model.width_out[layer_index + 1])
                x_out = torch.zeros((n, out_dim), dtype=torch.float32).to(self.device)

            # x_out is in the RAW dimension pre-collapse by multiplication
            # (width_out[l+1]), same as output_index (see _build_node_groups /
            # raw_indices). The affine that corresponds to THAT point of
            # pykan's real forward pass is subnode_bias/subnode_scale (sized
            # width_out[l+1]), NOT node_bias/node_scale (sized width_in[l+1],
            # applied AFTER the multiplication collapse). Using
            # node_bias/node_scale here is a shape bug on any layer with
            # surviving multiplication nodes (width_out != width_in), such as
            # the hidden layer in the production config (9 sum + 9 mult).
            if hasattr(model, "subnode_bias") and model.subnode_bias is not None and len(model.subnode_bias) > layer_index:
                x_out = x_out + model.subnode_bias[layer_index]
            if hasattr(model, "subnode_scale") and model.subnode_scale is not None and len(model.subnode_scale) > layer_index:
                x_out = x_out * model.subnode_scale[layer_index]
            return x_out

        with torch.no_grad():
            y_var = layer_forward(x_var)[:, output_index].cpu().numpy()
            y_zero = layer_forward(x_zero)[:, output_index].cpu().numpy()

        y_var = np.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0)
        y_zero = np.nan_to_num(y_zero, nan=0.0, posinf=0.0, neginf=0.0)
        return y_var - y_zero + (y_zero / in_dim)

    def _fit_edge(self, model, layer_index, input_index, output_index, x_vals):
        """
        Fits a Chebyshev polynomial to this edge's isolated response at a
        fixed degree (self.max_degree). No R2-gated search, no early
        acceptance at a lower degree..

        Args:
            model: The neural network model.
            layer_index: Index of the source layer.
            input_index: Index of the input neuron in the source layer.
            output_index: Index of the output neuron in the destination layer.
            x_vals: Input values to evaluate the edge response.

        Returns:
            A tuple (coefs, dynamic_range, degree, r2) where:
                coefs: List of Chebyshev polynomial coefficients.
                dynamic_range: Dynamic range of the edge response.
                degree: Degree of the fitted Chebyshev polynomial (always self.max_degree).
                r2: R-squared value of the fit (reported, not used to gate).
        """
        y_vals = self._evaluate_isolated_edges(model, layer_index, input_index, output_index, x_vals)
        dynamic_range = float(np.max(y_vals) - np.min(y_vals))

        ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
        ss_tot_safe = ss_tot if ss_tot > 1e-12 else 1e-12

        coefs = chebfit(x_vals, y_vals, deg=self.max_degree)
        y_pred = chebval(x_vals, coefs)
        r2 = 1.0 - float(np.sum((y_vals - y_pred) ** 2)) / ss_tot_safe

        return coefs.tolist(), dynamic_range, self.max_degree, r2

    # ------------------------------------------------------------------
    # Grouping of raw neurons into collapsed nodes (sum / mult)
    # ------------------------------------------------------------------
    def _build_node_groups(self, model, layer_plus_1_idx):
        """
        Returns a list of collapsed nodes for layer `layer_plus_1_idx`,
        i.e. the destination layer of the edges from layer layer_plus_1_idx-1.
        Each node is a dict: {'type': 'sum'|'mult', 'raw_indices': [...]}
        """
        width = model.width
        width_out = np.array(model.width_out)
        n_sum = width[layer_plus_1_idx][0]
        n_raw_total = int(width_out[layer_plus_1_idx])

        groups = []
        for j in range(n_sum):
            groups.append({"type": "sum", "raw_indices": [j]})

        mult_id = 0
        i = n_sum
        while i < n_raw_total:
            ma = model.mult_arity if isinstance(model.mult_arity, int) else model.mult_arity[layer_plus_1_idx][mult_id]
            raw_idx = list(range(i, i + ma))
            groups.append({"type": "mult", "raw_indices": raw_idx})
            i += ma
            mult_id += 1

        return groups

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------
    def extract_and_save(self, classic_model_path, output_graph_path, report_path):
        print("\n" + "=" * 40)
        print("[Extractor] Extracting structured graph (sum/mult) for QKAN")
        print("=" * 40)

        base_kan = HEPKAN.loadckpt(classic_model_path)
        model = HEPKAN.__new__(HEPKAN)
        model.__dict__.update(base_kan.__dict__)
        model.to(self.device)
        model.eval()

        x_vals = np.linspace(-1, 1, 500)
        depth = len(model.width) - 1
        if depth != 2:
            print(f"[Extractor] WARNING: expected depth=2 (input->hidden->output), "
                  f"found depth={depth}. This version of the extractor does NOT "
                  f"generalize to more depth without mid-circuit measurement.")

        # ---- Layer 0: inputs -> raw hidden neurons -----------------------
        n_inputs = int(model.width_in[0])
        n_raw_hidden = int(model.width_out[1])

        raw_edges_layer0 = {j: [] for j in range(n_raw_hidden)}  # j -> list of {input_idx, coefs}
        active_inputs_set = set()

        print("[Extractor] Evaluating active edges: inputs -> hidden layer...")
        for j in range(n_raw_hidden):
            for i in range(n_inputs):
                mask_act = model.act_fun[0].mask[i, j].item()
                if mask_act == 0.0:
                    continue

                coefs, dyn_range, degree, r2 = self._fit_edge(model, 0, i, j, x_vals)
                if dyn_range <= self.dynamic_range_threshold:
                    continue

                raw_edges_layer0[j].append({
                    "input_idx": i, "coefs": coefs, "dynamic_range": dyn_range,
                    "degree": degree, "r2": r2,
                })
                active_inputs_set.add(i)

        active_inputs = sorted(active_inputs_set)
        if not active_inputs:
            raise RuntimeError("[Extractor] No input edge cleared the dynamic-range threshold. "
                                "Check 'qkan_dynamic_range_threshold' or the classical pruning.")

        input_pos = {raw_idx: pos for pos, raw_idx in enumerate(active_inputs)}
        print(f"[Extractor] {len(active_inputs)} active classical variables (raw inputs: {active_inputs})")

        for j in raw_edges_layer0:
            for e in raw_edges_layer0[j]:
                e["col"] = input_pos[e["input_idx"]]

        # ---- Group raw neurons into collapsed hidden-layer nodes ----------
        # This is where the real quantum WIRES are assigned: one per surviving
        # raw neuron/group (an accumulator), never one per input. The same input
        # can write (via _qkan_edge) to several of these wires if it feeds
        # several branches.
        hidden_groups = self._build_node_groups(model, 1)  # layer 1 (hidden) nodes
        hidden_nodes = []
        wire_counter = 0
        for node in hidden_groups:
            edge_groups = []
            has_any_edge = False
            for raw_j in node["raw_indices"]:
                edges = raw_edges_layer0.get(raw_j, [])
                if edges:
                    has_any_edge = True
                    group_wire = wire_counter
                    wire_counter += 1
                    for e in edges:
                        # wire dedicated to this accumulator (not to the input)
                        e["wire"] = group_wire
                edge_groups.append(edges)
            if not has_any_edge:
                # Fully pruned hidden node (none of its raw neurons survived)
                continue
            hidden_nodes.append({"type": node["type"], "edge_groups": edge_groups})

        n_qubits = wire_counter
        n_sum_survivors = sum(1 for h in hidden_nodes if h["type"] == "sum")
        n_mult_survivors = sum(1 for h in hidden_nodes if h["type"] == "mult")
        print(f"[Extractor] Surviving hidden nodes: {n_sum_survivors} sum, {n_mult_survivors} mult")
        print(f"[Extractor] {n_qubits} qubits required (one per surviving accumulator/raw neuron, "
              f"NOT one per input — may differ from the number of active classical variables)")

        # ---- Layer 1: collapsed hidden nodes -> output ---------------------
        n_hidden_collapsed = int(model.width_in[1])
        n_output_raw = int(model.width_out[2])  # normally 1

        # Map "collapsed" hidden node index (0..n_hidden_collapsed-1) -> position in hidden_nodes
        # (the order from _build_node_groups already matches pykan's real collapsed index)
        collapsed_to_hidden = {}
        collapsed_idx = 0
        kept_idx = 0
        for node in hidden_groups:
            # need to know whether this node survived (is in hidden_nodes) to map it
            raw_j0 = node["raw_indices"][0]
            survived = any(raw_edges_layer0.get(rj, []) for rj in node["raw_indices"])
            if survived:
                collapsed_to_hidden[collapsed_idx] = kept_idx
                kept_idx += 1
            collapsed_idx += 1

        output_edges = []
        print("[Extractor] Evaluating active edges: hidden layer -> output...")
        for out_j in range(n_output_raw):
            for h_collapsed in range(n_hidden_collapsed):
                mask_act = model.act_fun[1].mask[h_collapsed, out_j].item()
                if mask_act == 0.0:
                    continue
                if h_collapsed not in collapsed_to_hidden:
                    continue  # the hidden node feeding this edge was fully pruned

                coefs, dyn_range, degree, r2 = self._fit_edge(model, 1, h_collapsed, out_j, x_vals)
                if dyn_range <= self.dynamic_range_threshold:
                    continue

                output_edges.append({
                    "hidden_idx": collapsed_to_hidden[h_collapsed],
                    "coefs": coefs,
                    "dynamic_range": dyn_range,
                    "degree": degree,
                    "r2": r2,
                })

        if not output_edges:
            raise RuntimeError("[Extractor] No hidden->output edge survived the threshold. "
                                "Check layer-1 node/edge pruning.")

        print(f"[Extractor] Active hidden->output edges: {len(output_edges)}")

        # ---- Pad every edge's coefs to a single uniform degree -------------
        all_edges = [e for edges in raw_edges_layer0.values() for e in edges] + output_edges
        final_degree = max(e["degree"] for e in all_edges)
        for e in all_edges:
            pad_width = final_degree + 1 - len(e["coefs"])
            if pad_width > 0:
                e["coefs"] = e["coefs"] + [0.0] * pad_width

        graph = {
            "n_qubits": n_qubits,
            "active_inputs": active_inputs,   # classical RAW indices (to filter X in forward())
            "degree": final_degree,
            "basis": self.basis,               # "chebyshev" | "sine" (selects the circuit edge in QKANModel)
            "hidden_nodes": hidden_nodes,      # list of {type, edge_groups: [[{wire,coefs}, ...], ...]}
            "output_edges": output_edges,      # list of {hidden_idx, coefs}
        }

        os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
        torch.save(graph, output_graph_path)
        print(f"[Extractor] Quantum graph exported to: {output_graph_path}")

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("=== STRUCTURED EXTRACTION REPORT (GRAPH) ===\n")
            f.write(f"n_qubits: {n_qubits}\n")
            f.write(f"active_inputs (raw): {active_inputs}\n")
            f.write(f"Chebyshev degree (padded, uniform across graph): {final_degree}\n")
            f.write(f"Surviving hidden nodes: {len(hidden_nodes)} "
                    f"({n_sum_survivors} sum, {n_mult_survivors} mult)\n")
            f.write(f"Hidden->output edges: {len(output_edges)}\n\n")
            for idx, node in enumerate(hidden_nodes):
                n_edges = sum(len(g) for g in node["edge_groups"])
                f.write(f"  Hidden node {idx} [{node['type']}]: {len(node['edge_groups'])} group(s), "
                        f"{n_edges} total edge(s)\n")
                for group in node["edge_groups"]:
                    for e in group:
                        f.write(f"    edge input_idx={e['input_idx']} col={e['col']}: "
                                f"degree={e['degree']}, R2={e['r2']:.5f}\n")
            f.write("\nOutput edges:\n")
            for oe in output_edges:
                f.write(f"  hidden_idx={oe['hidden_idx']}: degree={oe['degree']}, R2={oe['r2']:.5f}\n")

        return graph