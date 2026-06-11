from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunnerMaturity(str, Enum):
    REAL_DEVSIM = "real_devsim"
    REAL_EXTERNAL = "real_external"
    PHYSICS_SURROGATE = "physics_surrogate"
    CONTRACT_ONLY = "contract_only"


class IndustrialRunnerDescriptor(BaseModel):
    runner_id: str
    template_id: str
    display_name: str
    tool_name: str
    default_request: dict[str, Any] = Field(default_factory=dict)
    command: str
    maturity: RunnerMaturity
    dimensionality: str
    solver_backend: str
    solver_invoked: bool
    signoff_level: str
    metrics: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    capability_boundary: str
    signoff_gaps: list[str] = Field(default_factory=list)
    public_source_ids: list[str] = Field(default_factory=list)

    @property
    def agent_callable(self) -> bool:
        return self.maturity != RunnerMaturity.CONTRACT_ONLY


def industrial_runner_descriptors() -> list[IndustrialRunnerDescriptor]:
    return [
        IndustrialRunnerDescriptor(
            runner_id="pn_junction_iv_devsim_1d",
            template_id="pn_junction_iv",
            display_name="PN Junction IV DEVSIM 1D",
            tool_name="pn_junction_iv_sweep",
            default_request={"start": 0.0, "stop": 0.5, "step": 0.1},
            command="python3.11 -m tcad_agent.tools.pn_junction_iv",
            maturity=RunnerMaturity.REAL_DEVSIM,
            dimensionality="1d",
            solver_backend="devsim_1d_drift_diffusion",
            solver_invoked=True,
            signoff_level="iteration_evidence",
            metrics=["ideality_factor_estimate", "leakage_current_a", "rectification_ratio_final_to_leakage"],
            artifacts=["state.json", "sweep.csv", "plot", "devsim.log"],
            capability_boundary="Open-source DEVSIM 1D diode runner; final signoff still needs convergence and golden/measured correlation.",
            signoff_gaps=["mesh_or_bias_convergence", "golden_or_measured_correlation"],
            public_source_ids=["devsim_core_examples", "devsim_github"],
        ),
        IndustrialRunnerDescriptor(
            runner_id="mosfet_2d_id_devsim",
            template_id="mosfet_2d_id",
            display_name="2D MOSFET Id DEVSIM",
            tool_name="mosfet_2d_id_sweep",
            default_request={"sweep_type": "both", "gate_start": 0.0, "gate_stop": 1.0, "gate_step": 0.25},
            command="python3.11 -m tcad_agent.tools.mosfet_2d_id --sweep-type both",
            maturity=RunnerMaturity.REAL_DEVSIM,
            dimensionality="2d",
            solver_backend="devsim_2d_drift_diffusion",
            solver_invoked=True,
            signoff_level="layout_sensitive_iteration_evidence",
            metrics=["vth_at_threshold_current_v", "subthreshold_swing_mv_dec", "ion_ioff_ratio", "dibl_mv_per_v"],
            artifacts=["state.json", "mosfet_id_sweep.csv", "device_tecplot.dat", "devsim.log"],
            capability_boundary="Open-source 2D DEVSIM MOSFET runner for Id-Vg/Id-Vd and model-coupling evidence.",
            signoff_gaps=["mesh_model_convergence", "golden_or_measured_correlation"],
            public_source_ids=["devsim_github", "devsim_3dmos"],
        ),
        IndustrialRunnerDescriptor(
            runner_id="power_mosfet_bv_ron_devsim_1d",
            template_id="power_mosfet_bv_ron",
            display_name="Power MOSFET BV/Ron DEVSIM 1D",
            tool_name="power_mosfet_bv_ron_1d_runner",
            default_request={"device_type": "power_mosfet_bv_ron", "fidelity": "physics_1d"},
            command="python3.11 -m tcad_agent.tools.extended_device_sweep --device-type power_mosfet_bv_ron --fidelity physics_1d",
            maturity=RunnerMaturity.REAL_DEVSIM,
            dimensionality="1d",
            solver_backend="devsim_1d_power_mos_drift_poisson_avalanche",
            solver_invoked=True,
            signoff_level="iteration_evidence",
            metrics=["breakdown_voltage_v", "specific_on_resistance_ohm_cm2", "max_electric_field_v_per_cm"],
            artifacts=["runner_contract.json", "sweep.csv", "curve.svg", "device_tecplot.dat", "devsim.log", "summary.json"],
            capability_boundary="DEVSIM solves a 1D source/body/drift/drain electrostatic stack; termination terms are auditable extraction parameters.",
            signoff_gaps=["2d_or_3d_field_plate_geometry", "mesh_convergence", "golden_or_measured_correlation"],
            public_source_ids=["devsim_github", "gts_power_ldmos_si_2d"],
        ),
        IndustrialRunnerDescriptor(
            runner_id="power_mosfet_bv_ron_devsim_2d_field_plate",
            template_id="power_mosfet_bv_ron",
            display_name="Power MOSFET/LDMOS 2D Field-Plate DEVSIM",
            tool_name="power_mosfet_bv_ron_2d_runner",
            default_request={"device_type": "power_mosfet_bv_ron", "fidelity": "devsim_2d_field_plate"},
            command="python3.11 -m tcad_agent.tools.extended_device_sweep --device-type power_mosfet_bv_ron --fidelity devsim_2d_field_plate",
            maturity=RunnerMaturity.REAL_DEVSIM,
            dimensionality="2d",
            solver_backend="devsim_2d_power_mos_field_plate_layout_extraction",
            solver_invoked=True,
            signoff_level="layout_sensitive_iteration_evidence",
            metrics=[
                "breakdown_voltage_v",
                "specific_on_resistance_ohm_cm2",
                "max_electric_field_v_per_cm",
                "field_peak_x_um",
                "field_peak_y_um",
            ],
            artifacts=[
                "runner_contract.json",
                "sweep.csv",
                "curve.svg",
                "inner_devsim_csv",
                "inner_tecplot",
                "inner_devsim_log",
                "summary.json",
            ],
            capability_boundary=(
                "DEVSIM invokes a 2D MOS layout seed with field-plate/drift geometry mapped into a Power MOS extraction layer. "
                "This closes the 1D-only gap for autonomous iteration, but calibrated 2D/3D signoff still needs golden correlation."
            ),
            signoff_gaps=["calibrated_impact_ionization", "mesh_convergence", "golden_or_measured_correlation", "full_process_cross_section"],
            public_source_ids=["devsim_github", "devsim_3dmos", "gts_power_ldmos_si_2d"],
        ),
        IndustrialRunnerDescriptor(
            runner_id="gan_hemt_id_bv_physics_1d",
            template_id="gan_hemt_id_bv",
            display_name="GaN HEMT Id/BV Physics Runner",
            tool_name="gan_hemt_id_bv_runner",
            default_request={"device_type": "gan_hemt_id_bv", "fidelity": "physics_1d"},
            command="python3.11 -m tcad_agent.tools.extended_device_sweep --device-type gan_hemt_id_bv --fidelity physics_1d",
            maturity=RunnerMaturity.PHYSICS_SURROGATE,
            dimensionality="1d_surrogate",
            solver_backend="physics_1d_algan_gan_polarization_trap",
            solver_invoked=False,
            signoff_level="planning_evidence",
            metrics=["two_deg_density_cm2", "threshold_voltage_v", "on_current_a", "breakdown_voltage_v"],
            artifacts=["state.json", "sweep.csv", "curve.svg", "tcad_deck_spec.json"],
            capability_boundary="Agent-callable heterojunction/trap surrogate; not a DEVSIM/Sentaurus solved GaN stack.",
            signoff_gaps=["real_heterojunction_solver", "trap_calibration", "self_heating", "golden_or_measured_correlation"],
            public_source_ids=["genius_tcad_open", "gts_tutorial_catalog"],
        ),
        IndustrialRunnerDescriptor(
            runner_id="sic_power_diode_bv_leakage_physics_1d",
            template_id="sic_power_diode_bv_leakage",
            display_name="SiC Power Diode BV/Leakage Physics Runner",
            tool_name="sic_power_diode_bv_leakage_runner",
            default_request={"device_type": "sic_power_diode_bv_leakage", "fidelity": "physics_1d"},
            command="python3.11 -m tcad_agent.tools.extended_device_sweep --device-type sic_power_diode_bv_leakage --fidelity physics_1d",
            maturity=RunnerMaturity.PHYSICS_SURROGATE,
            dimensionality="1d_surrogate",
            solver_backend="physics_1d_sic_wide_bandgap_avalanche",
            solver_invoked=False,
            signoff_level="planning_evidence",
            metrics=["breakdown_voltage_v", "leakage_abs_current_at_target_a", "max_electric_field_v_per_cm"],
            artifacts=["state.json", "sweep.csv", "curve.svg", "tcad_deck_spec.json"],
            capability_boundary="Agent-callable SiC high-voltage surrogate; not a calibrated wide-bandgap TCAD solve.",
            signoff_gaps=["real_wide_bandgap_solver", "temperature_calibration", "mesh_convergence", "golden_or_measured_correlation"],
            public_source_ids=["genius_tcad_open", "devsim_core_examples"],
        ),
        IndustrialRunnerDescriptor(
            runner_id="igbt_output_turnoff_physics_1d",
            template_id="igbt_output_turnoff",
            display_name="IGBT Output/Turn-off Physics Runner",
            tool_name="igbt_output_turnoff_runner",
            default_request={"device_type": "igbt_output_turnoff", "fidelity": "physics_1d"},
            command="python3.11 -m tcad_agent.tools.extended_device_sweep --device-type igbt_output_turnoff --fidelity physics_1d",
            maturity=RunnerMaturity.PHYSICS_SURROGATE,
            dimensionality="1d_surrogate",
            solver_backend="physics_1d_bipolar_transient_tail",
            solver_invoked=False,
            signoff_level="planning_evidence",
            metrics=["on_state_voltage_v", "blocking_voltage_v", "tail_current_a"],
            artifacts=["state.json", "sweep.csv", "curve.svg", "tcad_deck_spec.json"],
            capability_boundary="Agent-callable IGBT output/turn-off surrogate; not a solved layered bipolar transient deck.",
            signoff_gaps=["real_bipolar_transient_solver", "lifetime_calibration", "thermal_coupling", "golden_or_measured_correlation"],
            public_source_ids=["genius_tcad_open", "gts_tutorial_catalog"],
        ),
    ]


def runner_descriptors_for_template(template_id: str) -> list[IndustrialRunnerDescriptor]:
    return [runner for runner in industrial_runner_descriptors() if runner.template_id == template_id]


def runner_descriptor_by_id(runner_id: str) -> IndustrialRunnerDescriptor | None:
    for runner in industrial_runner_descriptors():
        if runner.runner_id == runner_id:
            return runner
    return None


def preferred_runner_for_template(template_id: str) -> IndustrialRunnerDescriptor | None:
    candidates = runner_descriptors_for_template(template_id)
    if not candidates:
        return None
    rank = {
        RunnerMaturity.REAL_DEVSIM: 0,
        RunnerMaturity.REAL_EXTERNAL: 1,
        RunnerMaturity.PHYSICS_SURROGATE: 2,
        RunnerMaturity.CONTRACT_ONLY: 3,
    }
    return sorted(candidates, key=lambda item: (rank[item.maturity], 0 if item.dimensionality.startswith("2d") else 1))[0]


def industrial_runner_coverage_matrix() -> dict[str, Any]:
    by_template: dict[str, list[dict[str, Any]]] = {}
    for runner in industrial_runner_descriptors():
        by_template.setdefault(runner.template_id, []).append(runner.model_dump(mode="json"))
    return {
        "schema_version": "actsoft.tcad.industrial_runner_registry.v1",
        "runner_count": len(industrial_runner_descriptors()),
        "template_count": len(by_template),
        "by_template": by_template,
    }


def agent_callable_runner_tool_names() -> list[str]:
    return [runner.tool_name for runner in industrial_runner_descriptors() if runner.agent_callable]
