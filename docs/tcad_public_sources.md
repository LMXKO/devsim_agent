# Public TCAD Sources

`tcad_agent.public_sources` records public, non-proprietary references that can seed future TCAD runners, convergence recipes, and benchmark rules. It is a registry, not a vendored copy of third-party decks.

Use it from the CLI:

```bash
python3.11 -m tcad_agent.tools.device_templates sources
python3.11 -m tcad_agent.tools.device_templates sources --kind categories
python3.11 -m tcad_agent.tools.device_templates sources --kind sources
```

The registry intentionally keeps license and access notes close to each source. Open-source examples can be wrapped by adapters when the license is compatible. Vendor training pages and public PDFs are methodology references only unless their terms explicitly allow project import.

## Evidence Gate

`build_public_evidence_dossier(goal_text, simulator=..., template_ids=...)` turns the registry into a planning gate. The dossier is written into autonomous checkpoints as `checkpoint.public_evidence_dossier` and into Sentaurus patch plans as `public_evidence_dossier`.

The dossier contains:

- matched public source cards and category ids;
- convergence strategy, required models, expected metrics, and signoff boundaries;
- live lookup queries for operations not already covered by local deck IR or registry evidence;
- guardrails that forbid copying proprietary software, licenses, PDKs, calibrated model files, private decks, or treating fake backends as physics evidence.

This gate is intentionally registry-seeded and deterministic for tests. In real runs, if a requested simulator operation or device workflow is outside the local deck evidence plus registry coverage, the agent should perform live lookup or pause instead of inventing syntax or process semantics.

## Seven Seed Categories

| Category | Local template ids | Best public seeds | Current boundary |
| --- | --- | --- | --- |
| MOSFET Id-Vg / Id-Vd / DIBL | `mosfet_2d_id` | DEVSIM examples, DEVSIM 3D MOS, OpenSourceTCAD, Sentaurus quasistationary/Inspect training, Silvaco example taxonomy | Executable local 2D DEVSIM path exists; high-drain convergence and golden correlation still gate industrial signoff. |
| Diode / SBD Breakdown | `diode_breakdown_leakage`, `schottky_diode`, `sic_power_diode_bv_leakage` | DEVSIM diode examples, Genius-TCAD-Open high-field models, Sentaurus ramp/extraction training, Silvaco taxonomy | PN and Schottky are executable open-source evidence paths; avalanche/high-voltage signoff needs coupled impact ionization and field-mesh evidence. |
| LDMOS / IGBT Power Devices | `power_mosfet_bv_ron`, `igbt_output_turnoff` | GTS LDMOS application example, GTS tutorial catalog, Genius-TCAD-Open, Sentaurus quasistationary ramp pattern | Compact planning only; real high-voltage geometry runners are still needed. |
| GaN / AlGaN HEMT | `gan_hemt_id_bv` | GTS GaN/HEMT tutorial catalog, Genius-TCAD-Open model coverage, Charon manual | Planned only; needs heterojunction, polarization charge, traps/current-collapse, and high-field continuation. |
| BJT Gummel / Output | `bjt_gummel_output` | DEVSIM BJT publication example, OpenSourceTCAD Charon/Genius entries, Sentaurus Inspect extraction training | Compact planning only; DEVSIM BJT example is the clearest public promotion path. |
| FinFET / SOI Variability | `finfet_id_cv` | DEVSIM 3D MOS, DEVSIM density-gradient MOSCAP, GTS SOI/FinFET/variability examples | Planned only; needs 3D/advanced geometry, quantum correction, capacitance, and variability campaign support. |
| MOS Capacitor / Capacitance | `mos_capacitor_cv` | DEVSIM capacitor/MOSCAP examples, DEVSIM density-gradient MOSCAP, Silvaco C-V taxonomy | Executable local MOS C-V path exists; measured C-V correlation and optional AC/density-gradient qualification are future signoff gates. |

## Convergence Playbook

The category registry stores recommended convergence actions that can be reused by repair and long-horizon planning:

- MOSFET/DIBL: equilibrium first, ramp drain before gate sweep, split low/high drain Id-Vg, shrink gate step on threshold failures, save/load intermediate bias states.
- Breakdown/SBD: start from small reverse bias, refine near the threshold current, switch to current or resistor control after breakdown onset, watch field and power density, run temperature corners after room-temperature convergence.
- LDMOS/IGBT: separate off-state blocking from on-state output, ramp high voltage with small initial steps, use current/resistor control for snapback, reuse DC states for transient turn-off, refine field-plate and drift-junction mesh.
- GaN HEMT: converge heterojunction equilibrium with fixed polarization first, enable traps after DC convergence, split output sweeps by gate bias, run stress/recovery probes for dynamic Ron, limit high-field steps near gate/field plate.
- BJT: equilibrium first, base-emitter ramp, hold Vce for Gummel, sweep collector from saved base-bias states, extract gain only above the noise floor.
- FinFET/SOI: validate planar/2D surrogate before 3D, enable density-gradient after classical drift-diffusion, run nominal before random trap/dopant splits, reuse mesh across samples, sign off distributions rather than one point.
- MOSCAP: equilibrium first, sweep accumulation-to-inversion with step shrink, compare capacitance to analytic Cox, split oxide thickness/doping, enable density-gradient only after classical C-V is stable.

## Source Registry

The source ids currently stored in code are:

- `devsim_core_examples`: https://devsim.net/examples.html
- `devsim_github`: https://github.com/devsim/devsim
- `devsim_3dmos`: https://github.com/devsim/devsim_3dmos
- `devsim_bjt_example`: https://github.com/devsim/devsim_bjt_example
- `devsim_density_gradient`: https://github.com/devsim/devsim_density_gradient
- `opensource_tcad_bundle`: https://github.com/thesourcerer8/OpenSourceTCAD
- `genius_tcad_open`: https://github.com/cogenda/Genius-TCAD-Open
- `charon_user_manual`: https://www.sandia.gov/research/publications/details/charon-user-manual-v-2-2-revision1-2022-06-01/
- `sentaurus_quasistationary_training`: https://ghzphy.github.io/Sentaurus_Training/sd/sd_8.html
- `sentaurus_inspect_extraction`: https://kolegite.com/EE_library/books_and_lectures/%D0%90%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F%20%D0%BD%D0%B0%20%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B8%D1%80%D0%B0%D0%BD%D0%B5%D1%82%D0%BE%20%D0%B2%20%D0%95%D0%BB%D0%B5%D0%BA%D1%82%D1%80%D0%BE%D0%BD%D0%B8%D0%BA%D0%B0%D1%82%D0%B0/Sentaurus_Training/insp/insp_4.html
- `silvaco_guide_tcad_examples`: https://silvaco.com/wp-content/uploads/product/pdf/GuideTCAD.pdf
- `gts_tutorial_catalog`: https://www.globaltcad.com/resources/tutorials-application-examples/
- `gts_power_ldmos_si_2d`: https://www.globaltcad.com/download/power-ldmos-si-2d-application-example/

## Promotion Rules

Before a public source becomes an executable signoff route, add:

- a local runner or external adapter with explicit license handling;
- a smoke test that exercises the intended bias sequence;
- physical quality checks for units, monotonicity, model coupling, and field/thermal risk;
- repair actions tied to the known failure modes;
- benchmark evidence that labels compact/vendor-methodology references as conditional unless measured or golden correlation is present.
