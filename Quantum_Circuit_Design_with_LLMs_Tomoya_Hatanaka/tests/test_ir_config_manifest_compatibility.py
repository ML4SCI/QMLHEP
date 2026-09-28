"""Compatibility between the circuit IR and the Phase 0 config/manifest
infrastructure (llm_vqc.config, llm_vqc.manifest).

A future search-arm runner needs to: (a) embed a CircuitIR (or a sampler
seed that produces one) inside a RunConfig's free-form `params`, and
(b) have a proposed circuit's canonical form/hash show up in a run's
manifest for provenance. Both must survive a YAML/JSON round trip
losslessly.
"""

from __future__ import annotations

import json

from llm_vqc.config import RunConfig, load_config
from llm_vqc.ir.canonicalize import canonical_json, structural_hash
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.manifest import create_run_dir, write_manifest


def _sample_ir() -> CircuitIR:
    return CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires="all"),
        layers=[
            RotationLayer(gates=["RY", "RZ"], wires="all"),
            EntangleLayer(pattern="ring", gate="CNOT"),
        ],
        measurements=MeasurementSpec(observable="Z", wires="all"),
    )


def test_circuit_ir_round_trips_through_run_config_params(tmp_path):
    ir = _sample_ir()
    config = RunConfig(name="ir_smoke", seed=0, params={"circuit_ir": ir.model_dump(mode="json")})

    config_path = tmp_path / "config.yaml"
    import yaml

    with config_path.open("w") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f)

    loaded = load_config(config_path)
    restored_ir = CircuitIR.model_validate(loaded.params["circuit_ir"])
    assert restored_ir == ir
    assert structural_hash(restored_ir) == structural_hash(ir)


def test_circuit_ir_canonical_form_survives_manifest_write(tmp_path):
    ir = _sample_ir()
    config = RunConfig(
        name="ir_manifest_smoke",
        seed=0,
        params={
            "circuit_canonical_json": canonical_json(ir),
            "circuit_structural_hash": structural_hash(ir),
        },
    )
    run_dir = create_run_dir(tmp_path, config.name)
    manifest_path = write_manifest(run_dir, config)

    manifest = json.loads(manifest_path.read_text())
    recorded_hash = manifest["config"]["params"]["circuit_structural_hash"]
    recorded_canonical = manifest["config"]["params"]["circuit_canonical_json"]

    assert recorded_hash == structural_hash(ir)
    restored_ir = CircuitIR.model_validate(json.loads(recorded_canonical))
    assert restored_ir == ir
