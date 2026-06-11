from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PublicTCADSource(BaseModel):
    source_id: str
    name: str
    url: str
    source_type: str
    access: str
    license_note: str
    useful_for: list[str] = Field(default_factory=list)
    runnable_seed: str | None = None
    notes: list[str] = Field(default_factory=list)


class PublicTCADCategory(BaseModel):
    category_id: str
    display_name: str
    device_template_ids: list[str]
    source_ids: list[str]
    tasks: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    required_models: list[str] = Field(default_factory=list)
    convergence_strategy: list[str] = Field(default_factory=list)
    promotion_steps: list[str] = Field(default_factory=list)
    signoff_boundary: str


class PublicEvidenceSourceCard(BaseModel):
    source_id: str
    name: str
    url: str
    source_type: str
    access: str
    useful_for: list[str] = Field(default_factory=list)
    evidence_status: str = "registry_reference"
    usage_notes: list[str] = Field(default_factory=list)


class PublicEvidenceDossier(BaseModel):
    schema_version: str = "actsoft.tcad.public_evidence_dossier.v1"
    status: str
    goal_text: str
    simulator: str | None = None
    selected_category_ids: list[str] = Field(default_factory=list)
    source_cards: list[PublicEvidenceSourceCard] = Field(default_factory=list)
    convergence_strategy: list[str] = Field(default_factory=list)
    required_models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    signoff_boundaries: list[str] = Field(default_factory=list)
    live_lookup_queries: list[str] = Field(default_factory=list)
    live_lookup_status: str | None = None
    live_evidence_findings: list[dict[str, Any]] = Field(default_factory=list)
    verified_source_ids: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    evidence_gate: dict[str, Any] = Field(default_factory=dict)


def public_tcad_sources() -> list[PublicTCADSource]:
    return [
        PublicTCADSource(
            source_id="devsim_core_examples",
            name="DEVSIM core examples and manual",
            url="https://devsim.net/examples.html",
            source_type="open_source_examples",
            access="public",
            license_note="DEVSIM repository is Apache-2.0; verify each copied example before vendoring.",
            useful_for=["mos_capacitor_cv", "diode_breakdown_leakage", "mosfet_2d_id"],
            runnable_seed="examples/capacitance, examples/diode, examples/mobility",
            notes=[
                "Includes 1D/2D capacitor examples, 1D/2D/3D diode examples, mobility examples, small-signal diode scripts.",
                "Best first source for runners that must stay open-source and reproducible.",
            ],
        ),
        PublicTCADSource(
            source_id="devsim_github",
            name="DEVSIM GitHub repository",
            url="https://github.com/devsim/devsim",
            source_type="open_source_repository",
            access="public",
            license_note="Apache-2.0 repository; preserve notices if importing code.",
            useful_for=["mos_capacitor_cv", "diode_breakdown_leakage", "mosfet_2d_id"],
            runnable_seed="examples/, testing/, python_packages/",
            notes=[
                "Documents Python scripting, DC, AC, transient, user PDEs, and 1D/2D/3D simulation support.",
                "Use as the baseline source for local smoke tests and solver command patterns.",
            ],
        ),
        PublicTCADSource(
            source_id="devsim_3dmos",
            name="DEVSIM 3D MOSFET example",
            url="https://github.com/devsim/devsim_3dmos",
            source_type="open_source_example_repository",
            access="public",
            license_note="Check repository license and citation notice before importing simulation scripts.",
            useful_for=["mosfet_2d_id", "finfet_id_cv"],
            runnable_seed="ieee/ 3D MOSFET scripts and mesh-processing flow",
            notes=[
                "Useful as a public reference for 3D MOSFET geometry, mesh import, and publication-grade evidence packaging.",
                "Not a foundry deck; treat as template material only.",
            ],
        ),
        PublicTCADSource(
            source_id="devsim_bjt_example",
            name="DEVSIM BJT publication examples",
            url="https://github.com/devsim/devsim_bjt_example",
            source_type="open_source_example_repository",
            access="public",
            license_note="Apache-2.0 repository.",
            useful_for=["bjt_gummel_output"],
            runnable_seed="simdir/ BJT simulation files",
            notes=[
                "Public BJT example set tied to the DEVSIM publication.",
                "Best candidate for promoting the current compact BJT baseline to a real TCAD runner.",
            ],
        ),
        PublicTCADSource(
            source_id="devsim_density_gradient",
            name="DEVSIM density-gradient quantum correction example",
            url="https://github.com/devsim/devsim_density_gradient",
            source_type="open_source_example_repository",
            access="public",
            license_note="Apache-2.0 repository.",
            useful_for=["mos_capacitor_cv", "finfet_id_cv"],
            runnable_seed="moscap.py, moscap2d.py, ramp.py",
            notes=[
                "Useful public seed for density-gradient / quantum-correction workflow.",
                "Repository notes say the drift-diffusion coupling is mostly tested for MOSCAP, so FinFET use needs validation.",
            ],
        ),
        PublicTCADSource(
            source_id="opensource_tcad_bundle",
            name="OpenSourceTCAD Docker bundle",
            url="https://github.com/thesourcerer8/OpenSourceTCAD",
            source_type="open_source_bundle",
            access="public",
            license_note="Bundle references multiple tools; check each tool and example license before importing.",
            useful_for=["mosfet_2d_id", "bjt_gummel_output", "diode_breakdown_leakage"],
            runnable_seed="Genius examples/MOS/2D/nmos1_quad.inp; DevSim testing/mos_2d.py; Charon bjt1dbasebc",
            notes=[
                "Provides Docker recipes for Charon, DEVSIM, and Genius.",
                "Good source for external smoke-test runners, but the full build is heavy.",
            ],
        ),
        PublicTCADSource(
            source_id="genius_tcad_open",
            name="Genius-TCAD-Open",
            url="https://github.com/cogenda/Genius-TCAD-Open",
            source_type="open_source_repository",
            access="public",
            license_note="GPL-3.0; do not copy code into Apache/MIT-compatible paths without a license decision.",
            useful_for=[
                "diode_breakdown_leakage",
                "power_mosfet_bv_ron",
                "gan_hemt_id_bv",
                "igbt_output_turnoff",
            ],
            runnable_seed="examples/ plus 2D drift-diffusion, lattice heating, energy-balance, impact-ionization, BTBT, traps",
            notes=[
                "Useful as a model-coverage reference for high-field and wide-bandgap templates.",
                "GPL boundary means prefer adapter/runner documentation over code import.",
            ],
        ),
        PublicTCADSource(
            source_id="charon_user_manual",
            name="Charon user manual",
            url="https://www.sandia.gov/research/publications/details/charon-user-manual-v-2-2-revision1-2022-06-01/",
            source_type="public_manual",
            access="public",
            license_note="Sandia report/manual; use for reference, not vendored code.",
            useful_for=["bjt_gummel_output", "gan_hemt_id_bv", "diode_breakdown_leakage"],
            runnable_seed="OpenSourceTCAD Charon nightly-test entry points",
            notes=[
                "Reference for large parallel semiconductor simulation and radiation/displacement-damage workflows.",
                "Treat as long-term advanced-physics context rather than immediate local runner source.",
            ],
        ),
        PublicTCADSource(
            source_id="sentaurus_quasistationary_training",
            name="Sentaurus quasistationary sweep training",
            url="https://ghzphy.github.io/Sentaurus_Training/sd/sd_8.html",
            source_type="public_training",
            access="public_web",
            license_note="Public training mirror; use as methodology reference only.",
            useful_for=[
                "mosfet_2d_id",
                "diode_breakdown_leakage",
                "power_mosfet_bv_ron",
                "igbt_output_turnoff",
            ],
            runnable_seed="Quasistationary InitialStep/MaxStep/MinStep/Increment/Decrement pattern",
            notes=[
                "Use as the canonical public pattern for continuation, step growth, and shrink-on-failure.",
                "Do not copy proprietary deck syntax into generated open-source decks.",
            ],
        ),
        PublicTCADSource(
            source_id="sentaurus_inspect_extraction",
            name="Sentaurus Inspect standard extraction training",
            url="https://kolegite.com/EE_library/books_and_lectures/%D0%90%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F%20%D0%BD%D0%B0%20%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B8%D1%80%D0%B0%D0%BD%D0%B5%D1%82%D0%BE%20%D0%B2%20%D0%95%D0%BB%D0%B5%D0%BA%D1%82%D1%80%D0%BE%D0%BD%D0%B8%D0%BA%D0%B0%D1%82%D0%B0/Sentaurus_Training/insp/insp_4.html",
            source_type="public_training",
            access="public_web",
            license_note="Public training mirror; use as methodology reference only.",
            useful_for=["mosfet_2d_id", "bjt_gummel_output", "diode_breakdown_leakage"],
            runnable_seed="Vti, gm, SS, Ion/Ioff, DIBL, BJT gain, Early voltage, BV extraction patterns",
            notes=[
                "Useful for implementing metric extraction and failure labels.",
                "DIBL reference uses two Id-Vg curves at different drain voltages and constant-current Vth extraction.",
            ],
        ),
        PublicTCADSource(
            source_id="silvaco_guide_tcad_examples",
            name="Silvaco Guide to Using TCAD with Examples",
            url="https://silvaco.com/wp-content/uploads/product/pdf/GuideTCAD.pdf",
            source_type="vendor_public_guide",
            access="public_pdf",
            license_note="Vendor documentation; use taxonomy and methodology only, not as vendored deck content.",
            useful_for=["mosfet_2d_id", "diode_breakdown_leakage", "mos_capacitor_cv"],
            runnable_seed="MOS1 NMOS Id/Vgs and threshold extraction taxonomy",
            notes=[
                "Good source for industrial example taxonomy and expected extraction names.",
                "Use to improve natural-language routing and benchmark naming.",
            ],
        ),
        PublicTCADSource(
            source_id="gts_tutorial_catalog",
            name="Global TCAD Solutions tutorials and application examples",
            url="https://www.globaltcad.com/resources/tutorials-application-examples/",
            source_type="vendor_public_catalog",
            access="public_catalog_with_some_downloads",
            license_note="Vendor examples; keep as external reference unless license allows project import.",
            useful_for=["power_mosfet_bv_ron", "gan_hemt_id_bv", "finfet_id_cv", "igbt_output_turnoff"],
            runnable_seed="GaN HEMT, SOI FinFET, IGBT, LDMOS, variability application-example taxonomy",
            notes=[
                "Useful for planned-template coverage, metrics, and signoff workflow names.",
                "Some project downloads may require GTS tooling or account access.",
            ],
        ),
        PublicTCADSource(
            source_id="gts_power_ldmos_si_2d",
            name="GTS Power LDMOS Si 2D application example",
            url="https://www.globaltcad.com/download/power-ldmos-si-2d-application-example/",
            source_type="vendor_public_application_example",
            access="public_page_download_metadata",
            license_note="Vendor project files are external reference unless terms allow import.",
            useful_for=["power_mosfet_bv_ron"],
            runnable_seed="IdVg, IdVd, breakdown-voltage extraction flow",
            notes=[
                "Directly maps to the current power MOSFET BV/Ron compact template.",
                "Use for promotion checklist and expected artifact names.",
            ],
        ),
    ]


def public_tcad_categories() -> list[PublicTCADCategory]:
    return [
        PublicTCADCategory(
            category_id="mosfet_id_dibl",
            display_name="MOSFET Id-Vg / Id-Vd / DIBL",
            device_template_ids=["mosfet_2d_id"],
            source_ids=[
                "devsim_core_examples",
                "devsim_github",
                "devsim_3dmos",
                "opensource_tcad_bundle",
                "sentaurus_quasistationary_training",
                "sentaurus_inspect_extraction",
                "silvaco_guide_tcad_examples",
            ],
            tasks=["Id-Vg", "Id-Vd", "Vth", "SS", "Ion/Ioff", "DIBL", "gm"],
            metrics=["vth_at_threshold_current_v", "subthreshold_swing_mv_dec", "ion_ioff_ratio", "dibl_mv_per_v"],
            required_models=["drift_diffusion", "mobility_model", "oxide_interface", "contact_current"],
            convergence_strategy=[
                "solve_equilibrium_before_bias",
                "ramp_drain_before_gate_sweep",
                "split_low_high_drain_idvg_for_dibl",
                "shrink_gate_step_when_threshold_not_crossed",
                "save_load_intermediate_solution_between_bias_phases",
            ],
            promotion_steps=[
                "Add low/high drain continuation smoke test with both Vth values extracted.",
                "Persist per-bias solver traces so repair can restart from the last converged drain point.",
                "Add golden or measured Id-Vg/Id-Vd curve comparison before signoff.",
            ],
            signoff_boundary="Current local 2D DEVSIM runner is executable, but high-drain convergence and golden correlation remain required for industrial signoff.",
        ),
        PublicTCADCategory(
            category_id="diode_sbd_breakdown",
            display_name="Diode / SBD Breakdown",
            device_template_ids=["diode_breakdown_leakage", "schottky_diode", "sic_power_diode_bv_leakage"],
            source_ids=[
                "devsim_core_examples",
                "devsim_github",
                "genius_tcad_open",
                "sentaurus_quasistationary_training",
                "sentaurus_inspect_extraction",
                "silvaco_guide_tcad_examples",
            ],
            tasks=["reverse leakage", "breakdown voltage", "forward IV", "barrier extraction", "temperature corner"],
            metrics=["leakage_abs_current_at_target_a", "breakdown_voltage_at_threshold_v", "barrier_height_ev"],
            required_models=["srh_recombination", "impact_ionization", "thermionic_contact_for_sbd", "field_peak_extraction"],
            convergence_strategy=[
                "start_from_small_reverse_bias",
                "use_local_refinement_near_current_threshold",
                "switch_to_current_or_resistor_control_after_breakdown_onset",
                "cap_max_field_and_power_density_for_safety",
                "sweep_temperature_after_room_temperature_converges",
            ],
            promotion_steps=[
                "Add current-control or series-resistor continuation to reverse breakdown tools.",
                "Add field-peak mesh refinement evidence.",
                "Add Schottky C-V and image-force lowering calibration path.",
            ],
            signoff_boundary="PN and Schottky paths are executable for open-source evidence; high-voltage avalanche signoff needs coupled impact-ionization and local mesh evidence.",
        ),
        PublicTCADCategory(
            category_id="ldmos_igbt_power",
            display_name="LDMOS / IGBT Power Devices",
            device_template_ids=["power_mosfet_bv_ron", "igbt_output_turnoff"],
            source_ids=[
                "gts_tutorial_catalog",
                "gts_power_ldmos_si_2d",
                "genius_tcad_open",
                "sentaurus_quasistationary_training",
            ],
            tasks=["transfer curve", "output curve", "breakdown", "Ron/BV tradeoff", "turn-off transient"],
            metrics=["breakdown_voltage_v", "specific_on_resistance_ohm_cm2", "on_state_voltage_v", "tail_current_a"],
            required_models=["impact_ionization", "self_heating", "lifetime_model", "high_voltage_mesh", "transient_transport"],
            convergence_strategy=[
                "separate_off_state_blocking_from_on_state_output",
                "ramp_high_voltage_with_small_initial_step",
                "use_current_or_resistor_control_for_snapback_or_breakdown",
                "reuse_dc_solution_as_transient_initial_state",
                "tighten_mesh_at_drift_junction_and_field_plate_edges",
            ],
            promotion_steps=[
                "Promote power MOSFET compact baseline to a high-voltage geometry runner.",
                "Add IGBT layered geometry with lifetime and tail-current extraction.",
                "Add BV/Ron Pareto benchmark and thermal warning checks.",
            ],
            signoff_boundary="Current project has compact planning baselines only; public sources define the template and convergence playbook for real runners.",
        ),
        PublicTCADCategory(
            category_id="gan_algan_hemt",
            display_name="GaN / AlGaN HEMT",
            device_template_ids=["gan_hemt_id_bv"],
            source_ids=["gts_tutorial_catalog", "genius_tcad_open", "charon_user_manual"],
            tasks=["Id-Vg", "Id-Vd", "2DEG density", "breakdown", "current-collapse proxy"],
            metrics=["threshold_voltage_v", "on_current_a", "breakdown_voltage_v", "dynamic_ron_ratio"],
            required_models=["heterojunction", "polarization_charge", "trap_model", "field_plate", "self_heating"],
            convergence_strategy=[
                "solve_heterojunction_equilibrium_with_fixed_polarization_first",
                "ramp_trap_occupancy_or_enable_traps_after_dc_converges",
                "split_output_sweeps_by_gate_bias",
                "run_off_state_stress_before_dynamic_ron_probe",
                "limit_high_field_steps_near_gate_edge_and_field_plate",
            ],
            promotion_steps=[
                "Add AlGaN/GaN layer stack and polarization charge deck spec.",
                "Add trap/current-collapse proxy benchmark.",
                "Add high-field breakdown continuation with model-coupling evidence.",
            ],
            signoff_boundary="Planned only; public examples provide taxonomy and model requirements, not an open local signoff runner yet.",
        ),
        PublicTCADCategory(
            category_id="bjt_gummel_output",
            display_name="BJT Gummel / Output",
            device_template_ids=["bjt_gummel_output"],
            source_ids=["devsim_bjt_example", "opensource_tcad_bundle", "sentaurus_inspect_extraction", "charon_user_manual"],
            tasks=["Gummel plot", "Ic-Vce", "beta", "Early voltage", "collector leakage"],
            metrics=["current_gain_beta", "early_voltage_v", "collector_leakage_current_a"],
            required_models=["three_terminal_bipolar_geometry", "srh_recombination", "mobility_model", "contact_current"],
            convergence_strategy=[
                "solve_equilibrium_then_base_emitter_ramp",
                "hold_vce_for_gummel_before_output_family",
                "sweep_collector_voltage_from_saved_base_bias_states",
                "extract_gain_only_above_noise_floor",
                "separate_leakage_and_gain_bias_windows",
            ],
            promotion_steps=[
                "Wrap devsim_bjt_example into an agent-callable BJT runner.",
                "Add beta/Early-voltage physical benchmark rules.",
                "Add BJT-specific repair actions for noisy low-current gain extraction.",
            ],
            signoff_boundary="Current BJT route is compact baseline; DEVSIM BJT examples are the clearest public path to a real runner.",
        ),
        PublicTCADCategory(
            category_id="finfet_soi_variability",
            display_name="FinFET / SOI Variability",
            device_template_ids=["finfet_id_cv"],
            source_ids=["devsim_3dmos", "devsim_density_gradient", "gts_tutorial_catalog"],
            tasks=["3D Id-Vg", "C-V", "DIBL", "quantum correction", "trap/dopant variability"],
            metrics=["vth_at_threshold_current_v", "subthreshold_swing_mv_dec", "dibl_mv_per_v", "capacitance_f_per_cm2"],
            required_models=["3d_geometry", "density_gradient_or_quantum_correction", "variability_sampler", "gate_capacitance"],
            convergence_strategy=[
                "validate_planar_or_2d_surrogate_before_3d",
                "enable_density_gradient_after_drift_diffusion_converges",
                "run_nominal_geometry_before_random_trap_or_dopant_splits",
                "cache_mesh_and_reuse_for_variability_samples",
                "aggregate distributions_instead_of_single_point_signoff",
            ],
            promotion_steps=[
                "Add parameterized fin/nanosheet geometry or open 3D MOS import path.",
                "Add density-gradient smoke test on MOSCAP before FinFET use.",
                "Add variability campaign runner with distribution metrics.",
            ],
            signoff_boundary="Planned only; public materials support roadmap and benchmark design, not current local signoff execution.",
        ),
        PublicTCADCategory(
            category_id="moscap_capacitance",
            display_name="MOS Capacitor / Capacitance",
            device_template_ids=["mos_capacitor_cv"],
            source_ids=["devsim_core_examples", "devsim_github", "devsim_density_gradient", "silvaco_guide_tcad_examples"],
            tasks=["C-V", "Cox benchmark", "flat-band shift", "oxide/interface charge", "small-signal AC"],
            metrics=["max_capacitance_f_per_cm2", "oxide_capacitance_estimate_f_per_cm2", "flatband_shift_v"],
            required_models=["oxide_region", "silicon_poisson", "interface_charge", "small_signal_or_quasi_static_capacitance"],
            convergence_strategy=[
                "solve_equilibrium_before_voltage_sweep",
                "sweep_accumulation_to_inversion_with_step_shrink",
                "compare_max_capacitance_to_analytic_cox",
                "run_oxide_thickness_and_doping_splits",
                "enable_density_gradient_only_after_classical_cv_baseline",
            ],
            promotion_steps=[
                "Add optional AC small-signal path in addition to quasi-static C-V.",
                "Add interface-trap/fixed-charge calibration presets.",
                "Use density-gradient MOSCAP as a quantum-correction qualification test.",
            ],
            signoff_boundary="Current local MOS C-V path is executable; model calibration and measured C-V correlation are still needed for industrial signoff.",
        ),
    ]


def _sources_by_id() -> dict[str, PublicTCADSource]:
    return {source.source_id: source for source in public_tcad_sources()}


def _categories_by_id() -> dict[str, PublicTCADCategory]:
    return {category.category_id: category for category in public_tcad_categories()}


def get_public_tcad_source(source_id: str) -> PublicTCADSource | None:
    return _sources_by_id().get(source_id)


def get_public_tcad_category(category_id: str) -> PublicTCADCategory | None:
    return _categories_by_id().get(category_id)


def list_public_tcad_sources() -> list[dict[str, Any]]:
    return [source.model_dump(mode="json") for source in public_tcad_sources()]


def list_public_tcad_categories() -> list[dict[str, Any]]:
    return [category.model_dump(mode="json") for category in public_tcad_categories()]


def public_sources_for_template(template_id: str) -> list[dict[str, Any]]:
    source_ids: list[str] = []
    for category in public_tcad_categories():
        if template_id in category.device_template_ids:
            source_ids.extend(category.source_ids)
    sources = _sources_by_id()
    unique_ids = list(dict.fromkeys(source_ids))
    return [sources[source_id].model_dump(mode="json") for source_id in unique_ids if source_id in sources]


def public_categories_for_template(template_id: str) -> list[dict[str, Any]]:
    return [
        category.model_dump(mode="json")
        for category in public_tcad_categories()
        if template_id in category.device_template_ids
    ]


def validate_public_tcad_registry() -> list[str]:
    errors: list[str] = []
    sources = _sources_by_id()
    if len(sources) != len(public_tcad_sources()):
        errors.append("duplicate_public_source_id")
    categories = _categories_by_id()
    if len(categories) != len(public_tcad_categories()):
        errors.append("duplicate_public_category_id")
    for category in public_tcad_categories():
        for source_id in category.source_ids:
            if source_id not in sources:
                errors.append(f"missing_source:{category.category_id}:{source_id}")
        if not category.device_template_ids:
            errors.append(f"category_without_template:{category.category_id}")
        if not category.convergence_strategy:
            errors.append(f"category_without_convergence_strategy:{category.category_id}")
        if not category.promotion_steps:
            errors.append(f"category_without_promotion_steps:{category.category_id}")
    return errors


def normalized_goal_text(goal_text: str) -> str:
    return goal_text.lower().replace("_", " ").replace("-", " ")


CATEGORY_GOAL_KEYWORDS: dict[str, list[str]] = {
    "mosfet_id_dibl": ["mosfet", "id-vg", "idvg", "id-vd", "idvd", "dibl", "vth", "threshold", "subthreshold", "gm"],
    "diode_sbd_breakdown": ["diode", "sbd", "schottky", "reverse leakage", "breakdown", "bv", "耐压", "击穿", "漏电"],
    "ldmos_igbt_power": ["ldmos", "igbt", "power mos", "power device", "ron", "field plate", "guard ring", "trench", "终端"],
    "gan_algan_hemt": ["gan", "algan", "hemt", "2deg", "polarization", "current collapse", "dynamic ron"],
    "bjt_gummel_output": ["bjt", "gummel", "beta", "early voltage", "collector"],
    "finfet_soi_variability": ["finfet", "soi", "nanosheet", "variability", "density gradient", "quantum"],
    "moscap_capacitance": ["moscap", "mos capacitor", "c-v", "cv", "capacitance", "oxide", "cox", "flatband"],
}


def score_public_category(goal_text: str, category: PublicTCADCategory, template_ids: list[str] | None = None) -> int:
    text = normalized_goal_text(goal_text)
    score = 0
    if template_ids and any(template_id in category.device_template_ids for template_id in template_ids):
        score += 8
    for token in CATEGORY_GOAL_KEYWORDS.get(category.category_id, []):
        if token in text:
            score += 3
    for task in category.tasks:
        if normalized_goal_text(task) in text:
            score += 2
    for metric in category.metrics:
        if normalized_goal_text(metric) in text:
            score += 1
    return score


def select_public_evidence_categories(
    goal_text: str,
    *,
    template_ids: list[str] | None = None,
    max_categories: int = 3,
) -> list[PublicTCADCategory]:
    scored = [
        (score_public_category(goal_text, category, template_ids), category)
        for category in public_tcad_categories()
    ]
    selected = [category for score, category in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    if selected:
        return selected[:max_categories]
    if template_ids:
        matched = [
            category
            for category in public_tcad_categories()
            if any(template_id in category.device_template_ids for template_id in template_ids)
        ]
        if matched:
            return matched[:max_categories]
    return []


def source_card(source: PublicTCADSource, category_ids: list[str], simulator: str | None) -> PublicEvidenceSourceCard:
    notes = [
        "Use as public methodology or runnable open-source seed only; do not vendor private decks or commercial model files.",
        "Verify any simulator-specific operation against local project evidence before applying semantic patches.",
    ]
    if simulator and simulator.lower() == "sentaurus":
        notes.append("Sentaurus entries are adapter evidence only; real execution requires a user-owned licensed installation/profile.")
    return PublicEvidenceSourceCard(
        source_id=source.source_id,
        name=source.name,
        url=source.url,
        source_type=source.source_type,
        access=source.access,
        useful_for=source.useful_for,
        usage_notes=[*notes, *source.notes[:2], f"Selected by categories: {', '.join(category_ids)}."],
    )


def build_public_evidence_dossier(
    goal_text: str,
    *,
    simulator: str | None = None,
    template_ids: list[str] | None = None,
    max_categories: int = 3,
    live_lookup_result: dict[str, Any] | None = None,
) -> PublicEvidenceDossier:
    categories = select_public_evidence_categories(
        goal_text,
        template_ids=template_ids,
        max_categories=max_categories,
    )
    category_ids = [category.category_id for category in categories]
    source_ids = list(dict.fromkeys(source_id for category in categories for source_id in category.source_ids))
    sources = _sources_by_id()
    cards = [source_card(sources[source_id], category_ids, simulator) for source_id in source_ids if source_id in sources]
    convergence = list(dict.fromkeys(item for category in categories for item in category.convergence_strategy))
    required_models = list(dict.fromkeys(item for category in categories for item in category.required_models))
    metrics = list(dict.fromkeys(item for category in categories for item in category.metrics))
    signoff = [category.signoff_boundary for category in categories]
    queries = []
    if categories:
        queries.extend(f"{category.display_name} TCAD public example extraction convergence" for category in categories)
    if simulator:
        queries.append(f"{simulator} public training command extraction curve CSV")
    live_findings = []
    verified_source_ids: list[str] = []
    live_status = None
    if isinstance(live_lookup_result, dict):
        live_status = str(live_lookup_result.get("status") or "")
        raw_findings = live_lookup_result.get("findings")
        live_findings = [item for item in raw_findings if isinstance(item, dict)] if isinstance(raw_findings, list) else []
        raw_verified = live_lookup_result.get("verified_source_ids")
        verified_source_ids = [str(item) for item in raw_verified if item] if isinstance(raw_verified, list) else []
    return PublicEvidenceDossier(
        status="completed" if cards else "no_public_category_match",
        goal_text=goal_text,
        simulator=simulator,
        selected_category_ids=category_ids,
        source_cards=cards,
        convergence_strategy=convergence,
        required_models=required_models,
        metrics=metrics,
        signoff_boundaries=signoff,
        live_lookup_queries=queries,
        live_lookup_status=live_status,
        live_evidence_findings=live_findings,
        verified_source_ids=verified_source_ids,
        guardrails=[
            "Do not infer proprietary deck syntax from public summaries.",
            "Do not copy Sentaurus software, license strings, PDKs, calibrated model files, or private decks into the repository.",
            "For any operation not covered by local deck IR plus public evidence, pause or require live lookup before patch execution.",
            "Treat fake/contract backends as interface validation only, not physics evidence.",
        ],
        evidence_gate={
            "gate": "public_evidence_before_patch_or_signoff",
            "mode": "registry_seeded_offline_with_live_lookup_plan",
            "passed": bool(cards),
            "requires_live_lookup_for_new_operations": True,
            "source_count": len(cards),
            "category_count": len(categories),
            "live_lookup_status": live_status,
            "live_verified_source_count": len(verified_source_ids),
        },
    )
