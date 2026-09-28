# src/architectures/qkan_model.py
import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from src.architectures.sine_basis import build_sine_grid


class QKANModel(nn.Module):
    """
    QKAN with:
      - n_qubits = number of ACCUMULATORS (raw neurons/groups of the hidden
        layer that survive pruning), NOT the number of active inputs. An
        input can be re-uploaded onto several different wires if it feeds
        several branches; a wire is never identified with "the input".
      - Fan-out solved by re-uploading: the same classical variable is
        re-uploaded (RY/RZ) on every accumulator-wire where it participates,
        always read from its own data column (in_col), never from an
        "owned" wire that doesn't exist.
      - Summation within a node solved FOR FREE: applying RZ(theta) repeatedly
        on the SAME wire accumulates the angles (rotations on the same axis
        compose additively) -> no 2-qubit gates needed to sum.
      - Multiplication solved with IsingZZ, and ONLY between the wires that
        the pruned classical graph marks as feeding into a real mult node
        (the depth/count of IsingZZ gates is now dynamic, not fixed).

    EXPLICIT LIMITATION:
    The hidden->output stage ("stage 2") cannot re-upload a hidden node's
    value with a new exact DRU encoding, because that value lives in a
    qubit's accumulated phase/rotation, not as a classical number readable
    without measuring. What this version DOES achieve is that the
    connectivity of that second stage (which wires combine and with what
    weight) is derived from the real pruned graph, instead of being fixed
    as before. It's still a variational readout layer, not a literal
    functional composition of two KAN layers. Preserving that exactly would
    require mid-circuit measurement + re-encoding (see earlier discussion on
    the QKAN paper).
    """

    def __init__(self, graph_path, backend_mode="ideal", random_init=False):
        super().__init__()

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Quantum graph not found at {graph_path}. Run the extractor first.")

        graph = torch.load(graph_path, weights_only=False)
        self.n_qubits = graph["n_qubits"]
        self.active_inputs = graph["active_inputs"]
        self.degree = graph["degree"]
        self.basis = graph.get("basis", "chebyshev")
        if self.basis == "sine":
            # Fixed (non-trainable) SineKAN grid: theta_k(x) = freq_k * x + phase_k
            self._sine_freq, self._sine_phase = build_sine_grid(self.degree, is_first=True)

        # ------------------------------------------------------------
        # Building the STATIC PLAN (once, not inside the circuit)
        # ------------------------------------------------------------
        # edge_table: flat list of all layer-0 edges (input->hidden), each with
        # (data column to read, accumulator wire to write to).
        # zz_table: list of multiplication transfers (wire_a, wire_b).
        # output_table: list of hidden->output edges already resolved to wires.
        edge_table = []       # [{'in_col':.., 'acc_wire':.., 'coefs':[...]}]
        zz_table = []         # [{'wire_a':.., 'wire_b':..}]
        hidden_final_wire = []  # per surviving hidden node: wire representing its collapsed value
        output_table = []     # [{'src_wire':.., 'coefs':[...]}]  (or marker for "same output wire")

        for node in graph["hidden_nodes"]:
            raw_carrier_wires = []
            for group in node["edge_groups"]:
                if not group:
                    continue
                # The extractor already assigned the SAME dedicated wire to every
                # edge in this group (one per accumulator/raw neuron, not per input).
                acc_wire = group[0]["wire"]
                for edge in group:
                    assert edge["wire"] == acc_wire, (
                        "All edges in the same group must share a dedicated "
                        "wire; check the assignment in the extractor."
                    )
                    edge_table.append({
                        "in_col": edge["col"],   # where the classical data is READ from (filtered column)
                        "acc_wire": acc_wire,    # which wire the rotation is WRITTEN to (accumulator)
                        "coefs": edge["coefs"],
                    })
                raw_carrier_wires.append(acc_wire)

            if not raw_carrier_wires:
                continue

            if node["type"] == "mult" and len(raw_carrier_wires) > 1:
                # IsingZZ chain: standard approximation for arity > 2
                # (IsingZZ is a 2-body interaction; for arity>2 this is a
                # chain-composition approximation, not an exact N-ary
                # product).
                base = raw_carrier_wires[0]
                for other in raw_carrier_wires[1:]:
                    zz_table.append({"wire_a": other, "wire_b": base})
                hidden_final_wire.append(base)
            else:
                hidden_final_wire.append(raw_carrier_wires[0])

        # Output wire: the one from the hidden node with the most edges (simple,
        # deterministic heuristic; any active wire would work as a final accumulator).
        if hidden_final_wire:
            output_wire = hidden_final_wire[0]
        else:
            output_wire = 0

        for oe in graph["output_edges"]:
            src_wire = hidden_final_wire[oe["hidden_idx"]]
            output_table.append({"src_wire": src_wire, "coefs": oe["coefs"]})

        self._edge_table = edge_table
        self._zz_table = zz_table
        self._output_table = output_table
        self._output_wire = output_wire

        print(f"[QKAN] Plan built: {len(edge_table)} input edges, "
              f"{len(zz_table)} IsingZZ transfers (multiplication), "
              f"{len(output_table)} output edges. Output wire: {output_wire}.")

        # ------------------------------------------------------------
        # Trainable parameters (one per edge/transfer, shape (degree+1,)
        # for re-uploading edges, scalar for IsingZZ/output)
        # ------------------------------------------------------------
        if random_init:
            print("[QKAN] Random init: ignoring KAN-extracted coefficients, "
                  "drawing edge/output weights from a standard normal instead.")
            self.edge_weights = nn.Parameter(
                torch.randn((len(edge_table), self.degree + 1), dtype=torch.float32)
                if edge_table else torch.zeros((0, self.degree + 1))
            )
            self.output_weights = nn.Parameter(
                torch.randn((len(output_table), self.degree + 1), dtype=torch.float32)
            )
        else:
            self.edge_weights = nn.Parameter(
                torch.stack([torch.tensor(e["coefs"], dtype=torch.float32) for e in edge_table])
                if edge_table else torch.zeros((0, self.degree + 1))
            )
            self.output_weights = nn.Parameter(
                torch.stack([torch.tensor(o["coefs"], dtype=torch.float32) for o in output_table])
            )
        self.zz_weights = nn.Parameter(torch.zeros(len(zz_table)))

        self.backend_mode = backend_mode
        self.dev = self._initialize_device()
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _initialize_device(self):
        if self.backend_mode == "noisy":
            print("[QKAN] Configuring noisy simulator (FakeManilaV2 + NoiseModel)...")
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            from qiskit_aer.noise import NoiseModel

            fake_backend = FakeManilaV2()
            noise_model = NoiseModel.from_backend(fake_backend)

            return qml.device(
                "qiskit.aer",
                wires=self.n_qubits,
                backend="aer_simulator_density_matrix",
                noise_model=noise_model,
                shots=1024,
            )
        elif self.backend_mode == "shots":
            print("[QKAN] Configuring shot-based simulator...")
            return qml.device("default.qubit", wires=self.n_qubits, shots=1024)
        else:
            print("[QKAN] Configuring ideal simulator (lightning.qubit)...")
            return qml.device("lightning.qubit", wires=self.n_qubits)

    def _qkan_edge(self, x_val, weights, wire):
        """Chebyshev-basis re-uploading on `wire`. Calling this several times
        on the SAME wire for different inputs ACCUMULATES (sums) their
        contributions, because RZ rotations on the same axis compose their
        angles additively."""
        theta = torch.acos(torch.clamp(x_val, -0.9999, 0.9999))
        for i in range(self.degree):
            qml.RY(weights[i], wires=wire)
            qml.RZ(theta, wires=wire)
        qml.RY(weights[self.degree], wires=wire)

    def _qkan_edge_sine(self, x_val, weights, wire):
        """SineKAN-basis re-uploading on `wire`: per harmonic k, RY(A_k) then
        RZ(freq_k * x + phase_k); the last weight (index `degree`) is a final RY."""
        for k in range(self.degree):
            qml.RY(weights[k], wires=wire)
            qml.RZ(float(self._sine_freq[k]) * x_val + float(self._sine_phase[k]), wires=wire)
        qml.RY(weights[self.degree], wires=wire)

    def _circuit(self, inputs):
        # --- Stage 1: inputs -> hidden nodes (free sum + mult via ZZ) ---
        for idx, edge in enumerate(self._edge_table):
            # Reads the classical data from its column (in_col); writes the
            # rotation to its dedicated wire (acc_wire). If the same input
            # feeds several branches, it appears here multiple times with the
            # same in_col but a different acc_wire. `inputs` carries a batch
            # dimension (shape (batch, n_features)), so this reads the whole
            # column at once -- PennyLane's parameter broadcasting then
            # executes the full batch as one tape instead of one sample at a
            # time (see forward()).
            edge_fn = self._qkan_edge_sine if self.basis == "sine" else self._qkan_edge
            edge_fn(inputs[:, edge["in_col"]], self.edge_weights[idx], wire=edge["acc_wire"])

        for idx, zz in enumerate(self._zz_table):
            qml.IsingZZ(self.zz_weights[idx], wires=[zz["wire_a"], zz["wire_b"]])
            qml.CNOT(wires=[zz["wire_a"], zz["wire_b"]])

        # --- Stage 2: hidden nodes -> output (variational readout, see docstring) ---
        for idx, oe in enumerate(self._output_table):
            if oe["src_wire"] == self._output_wire:
                qml.RY(self.output_weights[idx][0], wires=self._output_wire)
            else:
                qml.IsingZZ(self.output_weights[idx][0], wires=[oe["src_wire"], self._output_wire])
                qml.CNOT(wires=[oe["src_wire"], self._output_wire])

        return qml.expval(qml.PauliZ(self._output_wire))

    def forward(self, x):
        # Single batched QNode call (PennyLane parameter broadcasting) instead
        # of one Python-level call per sample: same gates, same wires, same
        # weights per sample -- only the execution is batched so the device
        # (lightning.qubit/default.qubit/qiskit.aer) can run the whole batch
        # as one dispatch instead of `batch_size` separate ones.
        x_filtered = x[:, self.active_inputs]
        outputs = self.qnode(x_filtered)
        return outputs.to(dtype=torch.float32, device=x.device)

    def plot_circuit(self, save_path):
        print(f"[QKAN] Generating circuit diagram at {save_path}...")
        # NOTE: the qnode's argument is the vector of filtered classical DATA
        # (one per column in active_inputs), not one per wire — these are
        # different quantities now that an input can write to several wires.
        # _circuit now always expects a batch dimension, so draw a batch of 1.
        dummy_inputs = torch.rand(1, len(self.active_inputs))
        fig, ax = qml.draw_mpl(self.qnode, decimals=2, style="pennylane")(dummy_inputs)
        plt.title(f"Structured QKAN ({self.n_qubits} qubits, "
                  f"{len(self._zz_table)} live mult-nodes)", fontsize=20)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()