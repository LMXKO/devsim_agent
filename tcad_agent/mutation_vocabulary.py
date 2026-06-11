from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MutationVocabularyEntry(BaseModel):
    class_id: str
    display_name: str
    target_kind: str
    default_risk_level: str
    requires_user_confirmation: bool = False
    variable_name_tokens: list[list[str]] = Field(default_factory=list)
    goal_tags: list[str] = Field(default_factory=list)
    primary_metrics: list[str] = Field(default_factory=list)
    tradeoff_metrics: list[str] = Field(default_factory=list)
    semantic_patch_operations: list[str] = Field(default_factory=list)
    expected_curve_evidence: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    public_source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


MUTATION_VOCABULARY: tuple[MutationVocabularyEntry, ...] = (
    MutationVocabularyEntry(
        class_id="lifetime",
        display_name="Carrier lifetime",
        target_kind="model_parameter",
        default_risk_level="low",
        variable_name_tokens=[["LIFETIME"], ["TAU"], ["_TAU"]],
        goal_tags=["leakage"],
        primary_metrics=["leakage_abs_current_at_target_a", "leakage_current_a", "reverse_leakage_current_a"],
        tradeoff_metrics=["breakdown_voltage_v", "specific_on_resistance_ohm_cm2", "tail_current_a"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["reverse-bias leakage interval moves down", "BV/Ron stay within constraints"],
        stop_conditions=["leakage slope does not move", "stored-charge or transient metrics become active constraints"],
        public_source_ids=["sentaurus_inspect_extraction", "gts_tutorial_catalog"],
    ),
    MutationVocabularyEntry(
        class_id="region_specific_lifetime",
        display_name="Region-specific carrier lifetime",
        target_kind="region_model_parameter",
        default_risk_level="medium",
        variable_name_tokens=[
            ["LIFETIME", "REGION"],
            ["TAU", "REGION"],
            ["TAU", "N_DRIFT"],
            ["TAU", "P_BODY"],
            ["TAU", "ANODE"],
            ["TAU", "CATHODE"],
            ["TAU", "BASE"],
            ["TAU", "EMITTER"],
            ["TAU", "COLLECTOR"],
        ],
        goal_tags=["leakage"],
        primary_metrics=["leakage_abs_current_at_target_a", "reverse_leakage_current_a"],
        tradeoff_metrics=["breakdown_voltage_v", "tail_current_a", "specific_on_resistance_ohm_cm2"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["localized leakage improvement with smaller global side effect"],
        stop_conditions=["curve change is not localized", "stored charge or recombination tradeoff dominates"],
        public_source_ids=["sentaurus_inspect_extraction", "gts_tutorial_catalog"],
    ),
    MutationVocabularyEntry(
        class_id="trap_density",
        display_name="Trap density",
        target_kind="defect_model_parameter",
        default_risk_level="medium",
        variable_name_tokens=[["TRAP", "DENS"], ["TRAP", "CONC"], ["TRAP", "_N"], ["TRAPDENS"]],
        goal_tags=["leakage", "gan_algan_hemt"],
        primary_metrics=["leakage_abs_current_at_target_a", "ioff_current_a", "dynamic_ron_ratio"],
        tradeoff_metrics=["subthreshold_swing_mv_dec", "breakdown_voltage_v", "specific_on_resistance_ohm_cm2"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment", "sentaurus_add_model"],
        expected_curve_evidence=["off-state leakage or current-collapse proxy changes", "subthreshold shape remains physical"],
        stop_conditions=["leakage slope is unchanged", "trap-driven hysteresis/current-collapse evidence worsens"],
        public_source_ids=["gts_tutorial_catalog", "genius_tcad_open"],
    ),
    MutationVocabularyEntry(
        class_id="drift_doping",
        display_name="Drift doping",
        target_kind="process_parameter",
        default_risk_level="medium",
        variable_name_tokens=[["DRIFT", "DOP"], ["NDRIFT"], ["DRIFT", "CONC"]],
        goal_tags=["bv", "field", "ron", "leakage"],
        primary_metrics=["breakdown_voltage_v", "max_electric_field_v_per_cm", "specific_on_resistance_ohm_cm2"],
        tradeoff_metrics=["leakage_abs_current_at_target_a", "breakdown_voltage_v", "specific_on_resistance_ohm_cm2"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["BV/Ron/field Pareto movement is visible", "field peak moves without unbounded leakage"],
        stop_conditions=["Ron penalty violates constraints", "BV drops or field peak rises beyond tolerance"],
        public_source_ids=["sentaurus_quasistationary_training", "gts_power_ldmos_si_2d", "genius_tcad_open"],
    ),
    MutationVocabularyEntry(
        class_id="field_plate",
        display_name="Field plate",
        target_kind="geometry_parameter",
        default_risk_level="high",
        requires_user_confirmation=True,
        variable_name_tokens=[["FIELD", "PLATE"], ["FP", "LENGTH"]],
        goal_tags=["field", "bv"],
        primary_metrics=["max_electric_field_v_per_cm", "breakdown_voltage_v"],
        tradeoff_metrics=["specific_on_resistance_ohm_cm2", "capacitance_f_per_cm2", "leakage_abs_current_at_target_a"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["peak field reduces or moves away from the critical junction edge", "BV bracket moves upward"],
        stop_conditions=["field peak moves into oxide/corner", "capacitance/Ron tradeoff exceeds constraints"],
        public_source_ids=["gts_tutorial_catalog", "gts_power_ldmos_si_2d"],
    ),
    MutationVocabularyEntry(
        class_id="guard_ring",
        display_name="Guard ring",
        target_kind="termination_geometry",
        default_risk_level="high",
        requires_user_confirmation=True,
        variable_name_tokens=[["GUARD", "RING"], ["GR", "SPACING"], ["GR", "DOSE"]],
        goal_tags=["bv", "field"],
        primary_metrics=["breakdown_voltage_v", "max_electric_field_v_per_cm"],
        tradeoff_metrics=["active_area_um2", "leakage_abs_current_at_target_a", "specific_on_resistance_ohm_cm2"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["termination field spreads across rings", "BV bracket improves without leakage penalty"],
        stop_conditions=["termination peak worsens", "active area/process penalty is too large"],
        public_source_ids=["gts_tutorial_catalog", "genius_tcad_open"],
    ),
    MutationVocabularyEntry(
        class_id="oxide_thickness",
        display_name="Oxide thickness",
        target_kind="geometry_or_process_parameter",
        default_risk_level="high",
        requires_user_confirmation=True,
        variable_name_tokens=[["OXIDE"], ["TOX"], ["_TOX"]],
        goal_tags=["field", "leakage", "bv", "moscap_capacitance"],
        primary_metrics=["max_electric_field_v_per_cm", "leakage_abs_current_at_target_a", "oxide_capacitance_estimate_f_per_cm2"],
        tradeoff_metrics=["threshold_voltage_v", "capacitance_f_per_cm2", "subthreshold_swing_mv_dec"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["oxide/interface field or C-V Cox evidence moves consistently"],
        stop_conditions=["threshold/capacitance/process constraints are violated"],
        public_source_ids=["devsim_density_gradient", "silvaco_guide_tcad_examples"],
    ),
    MutationVocabularyEntry(
        class_id="implant_dose",
        display_name="Implant dose",
        target_kind="process_parameter",
        default_risk_level="high",
        requires_user_confirmation=True,
        variable_name_tokens=[["IMPLANT", "DOSE"], ["DOSE"], ["PPLUS", "DOSE"], ["NPLUS", "DOSE"]],
        goal_tags=["bv", "leakage", "ron"],
        primary_metrics=["breakdown_voltage_v", "leakage_abs_current_at_target_a", "specific_on_resistance_ohm_cm2"],
        tradeoff_metrics=["threshold_voltage_v", "max_electric_field_v_per_cm", "specific_on_resistance_ohm_cm2"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["dose perturbation identifies whether process dose controls the failure mode"],
        stop_conditions=["no Pareto improvement under constraints", "process window or junction placement becomes invalid"],
        public_source_ids=["gts_tutorial_catalog", "silvaco_guide_tcad_examples"],
    ),
    MutationVocabularyEntry(
        class_id="junction_depth",
        display_name="Junction depth",
        target_kind="process_geometry_parameter",
        default_risk_level="high",
        requires_user_confirmation=True,
        variable_name_tokens=[["JUNCTION", "DEPTH"], ["_XJ"], ["XJ"], ["JUNC", "DEPTH"]],
        goal_tags=["bv", "field", "leakage"],
        primary_metrics=["breakdown_voltage_v", "max_electric_field_v_per_cm"],
        tradeoff_metrics=["specific_on_resistance_ohm_cm2", "leakage_abs_current_at_target_a", "capacitance_f_per_cm2"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["field peak moves away from shallow junction edge"],
        stop_conditions=["leakage/Ron/process limits degrade", "field peak is not junction dominated"],
        public_source_ids=["gts_tutorial_catalog", "gts_power_ldmos_si_2d"],
    ),
    MutationVocabularyEntry(
        class_id="trench_corner_radius",
        display_name="Trench corner radius",
        target_kind="geometry_parameter",
        default_risk_level="high",
        requires_user_confirmation=True,
        variable_name_tokens=[["TRENCH", "RADIUS"], ["TRENCH", "CORNER"], ["CORNER", "RADIUS"]],
        goal_tags=["field", "bv"],
        primary_metrics=["max_electric_field_v_per_cm", "breakdown_voltage_v"],
        tradeoff_metrics=["specific_on_resistance_ohm_cm2", "capacitance_f_per_cm2", "leakage_abs_current_at_target_a"],
        semantic_patch_operations=["sentaurus_set_variable", "sentaurus_update_assignment"],
        expected_curve_evidence=["trench-corner field peak reduces in overlay comparison"],
        stop_conditions=["peak field is elsewhere", "geometry constraints dominate"],
        public_source_ids=["gts_tutorial_catalog", "genius_tcad_open"],
    ),
)


def list_mutation_vocabulary() -> list[dict[str, Any]]:
    return [entry.model_dump(mode="json") for entry in MUTATION_VOCABULARY]


def mutation_entry(class_id: str) -> MutationVocabularyEntry | None:
    for entry in MUTATION_VOCABULARY:
        if entry.class_id == class_id:
            return entry
    return None


def mutation_class_ids() -> list[str]:
    return [entry.class_id for entry in MUTATION_VOCABULARY]


def normalize_token_text(value: str) -> str:
    return value.upper().replace("-", "_")


def variable_matches_entry(variable_name: str, entry: MutationVocabularyEntry) -> bool:
    text = normalize_token_text(variable_name)
    for token_group in entry.variable_name_tokens:
        if all(normalize_token_text(token).strip("_") in text for token in token_group):
            return True
    return False


def classify_mutation_variable(variable_name: str) -> set[str]:
    classes = {entry.class_id for entry in MUTATION_VOCABULARY if variable_matches_entry(variable_name, entry)}
    if "region_specific_lifetime" in classes:
        classes.add("lifetime")
    return classes


def vocabulary_evidence_for(class_id: str) -> dict[str, Any]:
    entry = mutation_entry(class_id)
    if entry is None:
        return {"kind": "mutation_vocabulary", "class": class_id, "status": "unknown"}
    return {
        "kind": "mutation_vocabulary",
        "class": entry.class_id,
        "display_name": entry.display_name,
        "target_kind": entry.target_kind,
        "default_risk_level": entry.default_risk_level,
        "primary_metrics": entry.primary_metrics,
        "tradeoff_metrics": entry.tradeoff_metrics,
        "expected_curve_evidence": entry.expected_curve_evidence,
        "stop_conditions": entry.stop_conditions,
        "public_source_ids": entry.public_source_ids,
    }
