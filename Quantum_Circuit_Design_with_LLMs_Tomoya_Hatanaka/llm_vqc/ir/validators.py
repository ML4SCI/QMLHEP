"""Semantic validation for circuit IR proposals.

`llm_vqc.ir.schema` enforces structural typing (field names, literal enums,
per-field numeric bounds) via pydantic. This module enforces everything
pydantic cannot check context-free: qubit index bounds, duplicate operands,
pattern-specific field requirements, and circuit size limits.

The key design choice: `validate_proposal` collects *every* violation in
one pass rather than raising on the first one. A search algorithm or LLM
agent proposing a malformed circuit needs to know all the reasons it was
rejected, not just the first — fail-fast feedback makes iterative repair
much slower (Knipfer et al. report circuits needing 1-3 correction rounds
partly because errors were discovered one at a time).

Invalid input is never silently repaired or reinterpreted. If a proposal
is malformed, `validate_proposal` reports why and returns `valid=False`;
nothing here changes the meaning of a proposal to make it pass.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from llm_vqc.ir.expand import count_gate_applications
from llm_vqc.ir.schema import CircuitIR, EntangleLayer, Layer, RotationLayer
from llm_vqc.ir.wires import resolve_wires

# Circuit size / operation-count constraints (Phase 1 engineering defaults;
# no specific numbers are mandated by the master plan grammar — see
# llm_vqc/ir/README.md "Design decisions").
MAX_TOP_LEVEL_LAYERS = 20
MAX_EXPANDED_GATE_APPLICATIONS = 500


class ValidationIssue(BaseModel):
    """One structured, machine-actionable validation failure."""

    code: str
    message: str
    path: str = ""


class ValidationResult(BaseModel):
    """Result of validating a raw proposal dict against the IR schema."""

    model_config = {"arbitrary_types_allowed": True}

    valid: bool
    issues: list[ValidationIssue] = []
    ir: CircuitIR | None = None


def _pydantic_errors_to_issues(exc: ValidationError) -> list[ValidationIssue]:
    issues = []
    for error in exc.errors():
        path = ".".join(str(p) for p in error["loc"])
        issues.append(
            ValidationIssue(code=f"schema.{error['type']}", message=error["msg"], path=path)
        )
    return issues


def _check_duplicate_wires(wires: list[int], path: str) -> list[ValidationIssue]:
    seen: set[int] = set()
    issues = []
    for w in wires:
        if w in seen:
            issues.append(
                ValidationIssue(
                    code="wires.duplicate",
                    message=f"duplicate qubit index {w} in wire list",
                    path=path,
                )
            )
        seen.add(w)
    return issues


def _check_wire_bounds(wires: list[int], n_qubits: int, path: str) -> list[ValidationIssue]:
    issues = []
    for w in wires:
        if not (0 <= w < n_qubits):
            issues.append(
                ValidationIssue(
                    code="wires.out_of_bounds",
                    message=f"qubit index {w} out of bounds for n_qubits={n_qubits}",
                    path=path,
                )
            )
    return issues


def _validate_rotation_layer(
    layer: RotationLayer, n_qubits: int, path: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if layer.wires == "all":
        wires = list(range(n_qubits))
    else:
        wires = layer.wires
        issues += _check_wire_bounds(wires, n_qubits, f"{path}.wires")
        issues += _check_duplicate_wires(wires, f"{path}.wires")
        if not wires:
            issues.append(
                ValidationIssue(
                    code="rot.empty_wires", message="rotation layer has no wires", path=path
                )
            )
    return issues


def _validate_entangle_layer(
    layer: EntangleLayer, n_qubits: int, path: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if layer.pattern == "star":
        if layer.center is None:
            issues.append(
                ValidationIssue(
                    code="entangle.missing_center",
                    message="pattern 'star' requires 'center'",
                    path=path,
                )
            )
        elif not (0 <= layer.center < n_qubits):
            issues.append(
                ValidationIssue(
                    code="entangle.center_out_of_bounds",
                    message=f"center {layer.center} out of bounds for n_qubits={n_qubits}",
                    path=f"{path}.center",
                )
            )
    elif layer.center is not None:
        issues.append(
            ValidationIssue(
                code="entangle.unexpected_center",
                message=f"'center' is only used by pattern 'star', not {layer.pattern!r}",
                path=f"{path}.center",
            )
        )

    if layer.pattern == "pairs":
        pairs = layer.pairs or []
        if not pairs:
            issues.append(
                ValidationIssue(
                    code="entangle.missing_pairs",
                    message="pattern 'pairs' requires a non-empty 'pairs' list",
                    path=path,
                )
            )
        for control, target in pairs:
            if control == target:
                issues.append(
                    ValidationIssue(
                        code="entangle.self_loop",
                        message=f"pair ({control}, {target}) has control == target",
                        path=f"{path}.pairs",
                    )
                )
            for idx in (control, target):
                if not (0 <= idx < n_qubits):
                    issues.append(
                        ValidationIssue(
                            code="entangle.pair_out_of_bounds",
                            message=f"qubit index {idx} out of bounds for n_qubits={n_qubits}",
                            path=f"{path}.pairs",
                        )
                    )
    elif layer.pairs is not None:
        issues.append(
            ValidationIssue(
                code="entangle.unexpected_pairs",
                message=f"'pairs' is only used by pattern 'pairs', not {layer.pattern!r}",
                path=f"{path}.pairs",
            )
        )

    if layer.pattern in ("pairs", "none") and layer.wires != "all":
        issues.append(
            ValidationIssue(
                code="entangle.unexpected_wires",
                message=f"'wires' is not used by pattern {layer.pattern!r}; leave as default 'all'",
                path=f"{path}.wires",
            )
        )
    elif layer.pattern in ("ring", "line", "star", "all_to_all"):
        wires = resolve_wires(layer.wires, n_qubits)
        if layer.wires != "all":
            issues += _check_wire_bounds(wires, n_qubits, f"{path}.wires")
            issues += _check_duplicate_wires(wires, f"{path}.wires")
        if layer.pattern == "star" and layer.center is not None and layer.center not in wires:
            issues.append(
                ValidationIssue(
                    code="entangle.center_not_in_wires",
                    message=f"center {layer.center} must be included in the resolved wire scope",
                    path=path,
                )
            )
        min_wires = 2
        if len(wires) < min_wires and not issues:
            issues.append(
                ValidationIssue(
                    code="entangle.too_few_wires",
                    message=(
                        f"pattern {layer.pattern!r} needs at least {min_wires} wires, "
                        f"got {len(wires)}"
                    ),
                    path=path,
                )
            )

    return issues


def _validate_layer(layer: Layer, n_qubits: int, path: str) -> list[ValidationIssue]:
    if layer.type == "rot":
        return _validate_rotation_layer(layer, n_qubits, path)
    if layer.type == "entangle":
        return _validate_entangle_layer(layer, n_qubits, path)
    if layer.type == "repeat":
        issues: list[ValidationIssue] = []
        for i, sub in enumerate(layer.body):
            issues += _validate_layer(sub, n_qubits, f"{path}.body[{i}]")
        return issues
    return []  # pragma: no cover - schema already restricts `type`


def _validate_semantic(ir: CircuitIR) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if ir.encoding.type == "angle" and ir.encoding.gate is None:
        issues.append(
            ValidationIssue(
                code="encoding.missing_gate",
                message="encoding type 'angle' requires 'gate' (RX/RY/RZ)",
                path="encoding.gate",
            )
        )
    if ir.encoding.type == "amplitude":
        if ir.encoding.gate is not None:
            issues.append(
                ValidationIssue(
                    code="encoding.unexpected_gate",
                    message="encoding type 'amplitude' must not set 'gate'",
                    path="encoding.gate",
                )
            )
        if ir.encoding.reupload != 0:
            issues.append(
                ValidationIssue(
                    code="encoding.amplitude_reupload_unsupported",
                    message="'reupload' must be 0 for amplitude encoding (Phase 1 limitation)",
                    path="encoding.reupload",
                )
            )
    if ir.encoding.wires != "all":
        wires = ir.encoding.wires
        issues += _check_wire_bounds(wires, ir.n_qubits, "encoding.wires")
        issues += _check_duplicate_wires(wires, "encoding.wires")
        if not wires:
            issues.append(
                ValidationIssue(
                    code="encoding.empty_wires", message="encoding has no wires", path="encoding"
                )
            )

    if len(ir.layers) > MAX_TOP_LEVEL_LAYERS:
        issues.append(
            ValidationIssue(
                code="circuit.too_many_layers",
                message=f"{len(ir.layers)} top-level layers exceeds limit {MAX_TOP_LEVEL_LAYERS}",
                path="layers",
            )
        )

    for i, layer in enumerate(ir.layers):
        issues += _validate_layer(layer, ir.n_qubits, f"layers[{i}]")

    if ir.measurements.wires != "all":
        wires = ir.measurements.wires
        issues += _check_wire_bounds(wires, ir.n_qubits, "measurements.wires")
        issues += _check_duplicate_wires(wires, "measurements.wires")
        if not wires:
            issues.append(
                ValidationIssue(
                    code="measurements.empty_wires",
                    message="measurement has no wires",
                    path="measurements",
                )
            )

    # Size constraint requires expansion, which requires the circuit to
    # already be structurally sound; skip if earlier checks already failed
    # to avoid a confusing cascade (e.g. expanding an out-of-bounds star).
    if not issues:
        total_gates = count_gate_applications(ir)
        if total_gates > MAX_EXPANDED_GATE_APPLICATIONS:
            issues.append(
                ValidationIssue(
                    code="circuit.too_many_gate_applications",
                    message=(
                        f"{total_gates} expanded gate applications exceeds limit "
                        f"{MAX_EXPANDED_GATE_APPLICATIONS}"
                    ),
                    path="layers",
                )
            )

    return issues


def validate_proposal(raw: dict | CircuitIR) -> ValidationResult:
    """Validate a raw proposal (dict, as an LLM or search arm would emit).

    Runs pydantic schema validation first; on failure, converts every
    pydantic error into a ValidationIssue and returns immediately (semantic
    checks assume a structurally well-typed model and cannot run safely on
    a raw dict that failed even basic typing). On schema success, runs all
    semantic checks and collects every issue found, without stopping at
    the first one.
    """
    if isinstance(raw, CircuitIR):
        ir = raw
    else:
        try:
            ir = CircuitIR.model_validate(raw)
        except ValidationError as exc:
            return ValidationResult(valid=False, issues=_pydantic_errors_to_issues(exc))

    issues = _validate_semantic(ir)
    if issues:
        return ValidationResult(valid=False, issues=issues)
    return ValidationResult(valid=True, issues=[], ir=ir)
