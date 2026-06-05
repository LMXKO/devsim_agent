from __future__ import annotations

import math
from typing import Any


EPS0_F_PER_CM = 8.8541878128e-14
SIO2_RELATIVE_PERMITTIVITY = 3.9


def issue(
    code: str,
    severity: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence or {},
    }


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def oxide_capacitance_f_per_cm2(oxide_thickness_nm: float) -> float:
    oxide_thickness_cm = oxide_thickness_nm * 1e-7
    return EPS0_F_PER_CM * SIO2_RELATIVE_PERMITTIVITY / oxide_thickness_cm


def check_doping_ranges(
    parameters: dict[str, Any],
    *,
    min_cm3: float = 1e10,
    max_cm3: float = 5e21,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for key in ["p_doping_cm3", "n_doping_cm3", "substrate_doping_cm3", "source_drain_doping_cm3"]:
        if key not in parameters:
            continue
        value = float_or_none(parameters.get(key))
        if value is None or value <= 0:
            issues.append(
                issue(
                    "invalid_doping_value",
                    "error",
                    "Doping concentration must be a positive finite value.",
                    {"parameter": key, "value": parameters.get(key)},
                )
            )
        elif not min_cm3 <= value <= max_cm3:
            issues.append(
                issue(
                    "doping_out_of_expected_range",
                    "warning",
                    "Doping concentration is outside the configured semiconductor sanity range.",
                    {"parameter": key, "value_cm3": value, "expected_range_cm3": [min_cm3, max_cm3]},
                )
            )
    return issues


def check_temperature_range(
    parameters: dict[str, Any],
    *,
    min_k: float = 150.0,
    max_k: float = 500.0,
) -> list[dict[str, Any]]:
    if "temperature_k" not in parameters:
        return []
    value = float_or_none(parameters.get("temperature_k"))
    if value is None or value <= 0:
        return [
            issue(
                "invalid_temperature_value",
                "error",
                "Temperature must be a positive finite value in kelvin.",
                {"temperature_k": parameters.get("temperature_k")},
            )
        ]
    if not min_k <= value <= max_k:
        return [
            issue(
                "temperature_out_of_expected_range",
                "warning",
                "Temperature is outside the expected TCAD sanity range.",
                {"temperature_k": value, "expected_range_k": [min_k, max_k]},
            )
        ]
    return []


def check_transport_model_ranges(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for key in ["electron_lifetime_s", "hole_lifetime_s"]:
        if key not in parameters:
            continue
        value = float_or_none(parameters.get(key))
        if value is None or value <= 0:
            issues.append(
                issue(
                    "invalid_lifetime_value",
                    "error",
                    "Carrier lifetime must be a positive finite value in seconds.",
                    {"parameter": key, "value": parameters.get(key)},
                )
            )
        elif not 1.0e-12 <= value <= 1.0e-3:
            issues.append(
                issue(
                    "lifetime_out_of_expected_range",
                    "warning",
                    "Carrier lifetime is outside a broad semiconductor sanity range.",
                    {"parameter": key, "value_s": value, "expected_range_s": [1.0e-12, 1.0e-3]},
                )
            )
    for key in ["electron_mobility_cm2_v_s", "hole_mobility_cm2_v_s"]:
        if key not in parameters or parameters.get(key) is None:
            continue
        value = float_or_none(parameters.get(key))
        if value is None or value <= 0:
            issues.append(
                issue(
                    "invalid_mobility_value",
                    "error",
                    "Carrier mobility must be a positive finite value.",
                    {"parameter": key, "value": parameters.get(key)},
                )
            )
        elif not 1.0 <= value <= 3000.0:
            issues.append(
                issue(
                    "mobility_out_of_expected_range",
                    "warning",
                    "Carrier mobility is outside a broad silicon sanity range.",
                    {"parameter": key, "value_cm2_v_s": value, "expected_range_cm2_v_s": [1.0, 3000.0]},
                )
            )
    interface_trap = float_or_none(parameters.get("interface_trap_density_cm2"))
    if interface_trap is not None and interface_trap > 1.0e13:
        issues.append(
            issue(
                "interface_trap_density_extreme",
                "warning",
                "Interface trap density is extremely high; verify units and whether the model is equation-coupled.",
                {"interface_trap_density_cm2": interface_trap},
            )
        )
    fixed_charge = float_or_none(parameters.get("fixed_oxide_charge_cm2"))
    if fixed_charge is not None and abs(fixed_charge) > 1.0e13:
        issues.append(
            issue(
                "fixed_oxide_charge_extreme",
                "warning",
                "Fixed oxide charge is extremely high; verify cm^-2 units and expected flat-band shift.",
                {"fixed_oxide_charge_cm2": fixed_charge},
            )
        )
    return issues


def check_geometry_ranges(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for key, value in parameters.items():
        if not (key.endswith("_um") or key.endswith("_nm")):
            continue
        numeric = float_or_none(value)
        if numeric is None or numeric <= 0:
            issues.append(
                issue(
                    "invalid_geometry_value",
                    "error",
                    "Geometry and mesh dimensions must be positive finite values.",
                    {"parameter": key, "value": value},
                )
            )

    length_um = float_or_none(parameters.get("length_um"))
    junction_um = float_or_none(parameters.get("junction_um"))
    if length_um is not None and junction_um is not None and junction_um >= length_um:
        issues.append(
            issue(
                "junction_not_inside_device",
                "error",
                "PN junction position must be inside the simulated device length.",
                {"length_um": length_um, "junction_um": junction_um},
            )
        )

    for spacing_key in ["contact_spacing_um", "junction_spacing_um", "silicon_spacing_um"]:
        spacing = float_or_none(parameters.get(spacing_key))
        if spacing is None or length_um is None:
            continue
        if spacing > length_um / 5.0:
            issues.append(
                issue(
                    "mesh_spacing_too_coarse_for_device",
                    "warning",
                    "Mesh spacing is coarse compared with the simulated device length.",
                    {"parameter": spacing_key, "spacing_um": spacing, "length_um": length_um},
                )
            )

    oxide_thickness_nm = float_or_none(parameters.get("oxide_thickness_nm"))
    if oxide_thickness_nm is not None and not 0.5 <= oxide_thickness_nm <= 200.0:
        issues.append(
            issue(
                "oxide_thickness_out_of_expected_range",
                "warning",
                "Oxide thickness is outside the expected sanity range for compact TCAD examples.",
                {"oxide_thickness_nm": oxide_thickness_nm, "expected_range_nm": [0.5, 200.0]},
            )
        )

    silicon_thickness_um = float_or_none(parameters.get("silicon_thickness_um"))
    source_drain_depth_um = float_or_none(parameters.get("source_drain_depth_um"))
    if (
        silicon_thickness_um is not None
        and source_drain_depth_um is not None
        and source_drain_depth_um >= silicon_thickness_um
    ):
        issues.append(
            issue(
                "source_drain_depth_exceeds_silicon",
                "error",
                "Source/drain junction depth must be smaller than silicon thickness.",
                {"source_drain_depth_um": source_drain_depth_um, "silicon_thickness_um": silicon_thickness_um},
            )
        )

    source_drain_length_um = float_or_none(parameters.get("source_drain_length_um"))
    if length_um is not None and source_drain_length_um is not None and source_drain_length_um * 2.0 >= length_um:
        issues.append(
            issue(
                "source_drain_regions_leave_no_channel",
                "error",
                "Source/drain regions must leave a channel region between them.",
                {"source_drain_length_um": source_drain_length_um, "length_um": length_um},
            )
        )
    return issues


def check_parameter_sanity(
    parameters: dict[str, Any] | None,
    *,
    check_temperature: bool = True,
) -> list[dict[str, Any]]:
    if not parameters:
        return []
    issues = []
    issues.extend(check_doping_ranges(parameters))
    issues.extend(check_geometry_ranges(parameters))
    issues.extend(check_transport_model_ranges(parameters))
    if check_temperature:
        issues.extend(check_temperature_range(parameters))
    return issues


def check_voltage_span(
    voltage_range: list[Any] | tuple[Any, Any] | None,
    *,
    max_span_v: float = 200.0,
    code: str = "voltage_span_unusually_large",
) -> list[dict[str, Any]]:
    if not voltage_range or len(voltage_range) != 2:
        return []
    left = float_or_none(voltage_range[0])
    right = float_or_none(voltage_range[1])
    if left is None or right is None:
        return [
            issue(
                "invalid_voltage_range",
                "error",
                "Voltage range must contain finite numeric endpoints in volts.",
                {"voltage_range_v": list(voltage_range)},
            )
        ]
    span = abs(right - left)
    if span > max_span_v:
        return [
            issue(
                code,
                "warning",
                "Voltage span is unusually large; check whether voltage units were entered correctly.",
                {"voltage_range_v": [left, right], "span_v": span, "max_span_v": max_span_v},
            )
        ]
    return []


def check_mos_capacitor_physics(
    metrics: dict[str, Any],
    parameters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(check_parameter_sanity(parameters))
    issues.extend(check_voltage_span(metrics.get("voltage_range_v"), max_span_v=50.0, code="mos_cv_voltage_span_unusually_large"))

    for key in ["min_capacitance_f_per_cm2", "max_capacitance_f_per_cm2", "final_capacitance_f_per_cm2"]:
        value = float_or_none(metrics.get(key))
        if value is None:
            continue
        if value <= 0:
            issues.append(
                issue(
                    "capacitance_nonpositive",
                    "error",
                    "MOS capacitance must be positive after absolute-value extraction.",
                    {"metric": key, "value": metrics.get(key)},
                )
            )
        elif value < 1e-12 or value > 1e-3:
            issues.append(
                issue(
                    "capacitance_out_of_expected_range",
                    "warning",
                    "MOS capacitance per area is outside a broad sanity range.",
                    {"metric": key, "value_f_per_cm2": value, "expected_range_f_per_cm2": [1e-12, 1e-3]},
                )
            )

    oxide_thickness_nm = float_or_none((parameters or {}).get("oxide_thickness_nm"))
    max_cap = float_or_none(metrics.get("max_capacitance_f_per_cm2"))
    min_cap = float_or_none(metrics.get("min_capacitance_f_per_cm2"))
    if oxide_thickness_nm is not None and max_cap is not None and oxide_thickness_nm > 0:
        cox = oxide_capacitance_f_per_cm2(oxide_thickness_nm)
        metrics["oxide_capacitance_estimate_f_per_cm2"] = cox
        if max_cap > 1.5 * cox:
            issues.append(
                issue(
                    "capacitance_exceeds_oxide_capacitance",
                    "warning",
                    "Extracted MOS capacitance exceeds the oxide capacitance estimate.",
                    {"max_capacitance_f_per_cm2": max_cap, "oxide_capacitance_estimate_f_per_cm2": cox},
                )
            )
        elif max_cap < 0.01 * cox:
            issues.append(
                issue(
                    "capacitance_far_below_oxide_capacitance",
                    "warning",
                    "Maximum MOS capacitance is far below the oxide capacitance estimate; check bias window, area normalization, or depletion-only sweep.",
                    {"max_capacitance_f_per_cm2": max_cap, "oxide_capacitance_estimate_f_per_cm2": cox},
                )
            )

    if min_cap is not None and max_cap is not None and min_cap > 0 and max_cap > 0:
        ratio = max_cap / min_cap
        metrics["capacitance_dynamic_range"] = ratio
        span = None
        voltage_range = metrics.get("voltage_range_v")
        if isinstance(voltage_range, (list, tuple)) and len(voltage_range) == 2:
            left = float_or_none(voltage_range[0])
            right = float_or_none(voltage_range[1])
            if left is not None and right is not None:
                span = abs(right - left)
        if span is not None and span >= 1.0 and ratio < 1.02:
            issues.append(
                issue(
                    "moscap_cv_dynamic_range_too_low",
                    "warning",
                    "MOS C-V curve is nearly flat across a meaningful voltage span; check whether the sweep window, oxide charge, or area normalization is wrong.",
                    {"capacitance_dynamic_range": ratio, "voltage_span_v": span},
                )
            )
        elif ratio > 1.0e4:
            issues.append(
                issue(
                    "moscap_cv_dynamic_range_extreme",
                    "warning",
                    "MOS C-V capacitance dynamic range is extremely large; inspect derivative noise or convergence around the transition.",
                    {"capacitance_dynamic_range": ratio},
                )
            )

    fixed_charge = float_or_none((parameters or {}).get("fixed_oxide_charge_cm2"))
    fixed_shift = float_or_none(metrics.get("fixed_charge_voltage_shift_v"))
    if fixed_charge is not None and fixed_charge > 0 and fixed_shift is None:
        issues.append(
            issue(
                "fixed_oxide_charge_not_accounted_in_metrics",
                "warning",
                "Fixed oxide charge was requested but no equivalent voltage-shift metric was recorded.",
                {"fixed_oxide_charge_cm2": fixed_charge},
            )
        )
    if fixed_shift is not None:
        voltage_range = metrics.get("voltage_range_v")
        if isinstance(voltage_range, (list, tuple)) and len(voltage_range) == 2:
            left = float_or_none(voltage_range[0])
            right = float_or_none(voltage_range[1])
            if left is not None and right is not None and abs(fixed_shift) > abs(right - left):
                issues.append(
                    issue(
                        "fixed_charge_shift_exceeds_sweep_window",
                        "warning",
                        "Fixed-charge equivalent voltage shift is larger than the requested C-V sweep window.",
                        {"fixed_charge_voltage_shift_v": fixed_shift, "voltage_range_v": [left, right]},
                    )
                )
    return issues


def check_mosfet_physics(
    metrics: dict[str, Any],
    parameters: dict[str, Any] | None,
    *,
    gate_start_v: float | None = None,
    gate_stop_v: float | None = None,
    drain_start_v: float | None = None,
    drain_stop_v: float | None = None,
    physics_models: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(check_parameter_sanity(parameters))

    ss = float_or_none(metrics.get("subthreshold_swing_mv_dec"))
    if ss is not None:
        if ss < 55.0:
            issues.append(
                issue(
                    "subthreshold_swing_below_thermal_limit",
                    "warning",
                    "Extracted subthreshold swing is below the room-temperature thermal limit; check curve extraction or units.",
                    {"subthreshold_swing_mv_dec": ss, "thermal_limit_mv_dec": 60.0},
                )
            )
        elif ss > 5000.0:
            issues.append(
                issue(
                    "subthreshold_swing_unusually_large",
                    "warning",
                    "Extracted subthreshold swing is unusually large for a usable MOSFET transfer curve.",
                    {"subthreshold_swing_mv_dec": ss},
                )
            )

    vth = float_or_none(metrics.get("vth_at_threshold_current_v"))
    if vth is None and int(metrics.get("idvg_points") or 0) >= 2:
        issues.append(
            issue(
                "threshold_not_crossed",
                "warning",
                "Id-Vg did not cross the configured threshold current.",
                {"threshold_current_a": metrics.get("threshold_current_a")},
            )
        )
    elif vth is not None and gate_start_v is not None and gate_stop_v is not None:
        low = min(gate_start_v, gate_stop_v)
        high = max(gate_start_v, gate_stop_v)
        if not low <= vth <= high:
            issues.append(
                issue(
                    "threshold_outside_gate_sweep",
                    "warning",
                    "Extracted threshold voltage lies outside the requested gate sweep.",
                    {"vth_at_threshold_current_v": vth, "gate_range_v": [low, high]},
                )
            )

    gm = float_or_none(metrics.get("max_transconductance_s"))
    if gm is not None and gm == 0 and int(metrics.get("idvg_points") or 0) >= 2:
        issues.append(
            issue(
                "zero_transconductance",
                "warning",
                "Id-Vg has zero extracted transconductance.",
                {"max_transconductance_s": gm},
            )
        )

    idvd_negative = int(float_or_none(metrics.get("idvd_negative_differential_segments")) or 0)
    if idvd_negative > 0:
        issues.append(
            issue(
                "idvd_negative_differential_conductance",
                "warning",
                "Id-Vd output current decreases with increasing drain voltage in one or more segments.",
                {"idvd_negative_differential_segments": idvd_negative},
            )
        )
    kink_jumps = int(float_or_none(metrics.get("idvd_kink_slope_jumps")) or 0)
    if kink_jumps > 0:
        issues.append(
            issue(
                "idvd_kink_suspected",
                "warning",
                "Id-Vd output curve has abrupt slope increases that may indicate kink behavior or numerical artifacts.",
                {"idvd_kink_slope_jumps": kink_jumps},
            )
        )
    saturation_ratio = float_or_none(metrics.get("idvd_saturation_ratio"))
    idvd_points = int(metrics.get("idvd_points") or 0)
    drain_span = float_or_none(metrics.get("idvd_max_drain_span_v"))
    if drain_span is None and drain_start_v is not None and drain_stop_v is not None:
        drain_span = abs(drain_stop_v - drain_start_v)
    if (
        saturation_ratio is not None
        and idvd_points >= 5
        and drain_span is not None
        and drain_span >= 0.8
        and saturation_ratio > 0.75
    ):
        issues.append(
            issue(
                "idvd_saturation_not_observed",
                "warning",
                "High-drain Id-Vd slope remains close to the maximum slope; saturation may not be captured in the requested bias range.",
                {"idvd_saturation_ratio": saturation_ratio, "drain_span_v": drain_span},
            )
        )

    models = physics_models or {}
    interface_trap_density = float_or_none(models.get("interface_trap_density_cm2"))
    fixed_oxide_charge = float_or_none(models.get("fixed_oxide_charge_cm2"))
    advanced_coupling = models.get("advanced_model_coupling")
    coupled = advanced_coupling in {"compact_equivalent_bias_and_avalanche", "equation_coupled"}
    if interface_trap_density is not None and interface_trap_density > 0 and not coupled:
        issues.append(
            issue(
                "interface_trap_model_metadata_only",
                "warning",
                "Interface trap density is recorded, but this runner has not yet coupled trap charge into the interface equations.",
                {"interface_trap_density_cm2": interface_trap_density},
            )
        )
    if fixed_oxide_charge is not None and fixed_oxide_charge > 0 and not coupled:
        issues.append(
            issue(
                "fixed_oxide_charge_metadata_only",
                "warning",
                "Fixed oxide charge is recorded, but this runner has not yet coupled it into Poisson charge.",
                {"fixed_oxide_charge_cm2": fixed_oxide_charge},
            )
        )
    if models.get("impact_ionization_model") not in {None, "none"} and not coupled:
        issues.append(
            issue(
                "impact_ionization_model_metadata_only",
                "warning",
                "Impact ionization model is recorded, but avalanche generation is not yet coupled into the continuity equations.",
                {"impact_ionization_model": models.get("impact_ionization_model")},
            )
        )
    return issues


def check_diode_breakdown_physics(
    metrics: dict[str, Any],
    parameters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(check_parameter_sanity(parameters))
    issues.extend(check_voltage_span(metrics.get("voltage_range_v"), max_span_v=500.0, code="diode_reverse_voltage_span_unusually_large"))

    min_reverse = float_or_none(metrics.get("min_reverse_voltage_v"))
    if min_reverse is not None and min_reverse >= 0:
        issues.append(
            issue(
                "diode_reverse_sweep_missing_negative_bias",
                "error",
                "Breakdown/leakage analysis requires negative reverse-bias points.",
                {"min_reverse_voltage_v": min_reverse},
            )
        )

    breakdown_voltage = float_or_none(metrics.get("breakdown_voltage_at_threshold_v"))
    if breakdown_voltage is not None and breakdown_voltage > 0:
        issues.append(
            issue(
                "breakdown_voltage_wrong_polarity",
                "error",
                "Breakdown voltage should be negative for the reverse-bias convention used by this tool.",
                {"breakdown_voltage_at_threshold_v": breakdown_voltage},
            )
        )

    leakage = float_or_none(metrics.get("leakage_abs_current_at_target_a"))
    max_reverse = float_or_none(metrics.get("max_reverse_abs_current_a"))
    if leakage is not None and max_reverse is not None and leakage > max_reverse * 1.01:
        issues.append(
            issue(
                "leakage_exceeds_max_reverse_current",
                "warning",
                "Leakage current at the target bias exceeds the maximum reverse current metric; inspect extraction voltage and sign convention.",
                {"leakage_abs_current_at_target_a": leakage, "max_reverse_abs_current_a": max_reverse},
            )
        )
    reverse_gain = float_or_none(metrics.get("reverse_abs_current_gain"))
    if reverse_gain is not None and reverse_gain < 1.0:
        issues.append(
            issue(
                "reverse_current_gain_below_one",
                "warning",
                "Reverse current magnitude does not increase across the reverse sweep.",
                {"reverse_abs_current_gain": reverse_gain},
            )
        )
    return issues
