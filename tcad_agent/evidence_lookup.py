from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from tcad_agent.public_sources import build_public_evidence_dossier, get_public_tcad_source, public_tcad_sources


Fetcher = Callable[[str, float], tuple[int, str]]


class PublicEvidenceFinding(BaseModel):
    source_id: str
    name: str
    url: str
    source_type: str
    access: str
    status: str
    live_checked: bool = False
    fetched_at: str | None = None
    http_status: int | None = None
    title: str | None = None
    snippet: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    methodology_claims: list[str] = Field(default_factory=list)
    license_note: str | None = None
    failure_reason: str | None = None


class PublicEvidenceLookupRequest(BaseModel):
    goal_text: str
    simulator: str | None = None
    template_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    max_sources: int = Field(default=6, ge=1, le=24)
    live: bool = False
    timeout_seconds: float = Field(default=8.0, gt=0)
    output_path: Path | None = None


class PublicEvidenceLookupResult(BaseModel):
    tool_name: str = "public_evidence_lookup"
    schema_version: str = "actsoft.tcad.public_evidence_lookup.v1"
    status: str
    goal_text: str
    simulator: str | None = None
    live: bool = False
    source_ids: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    findings: list[PublicEvidenceFinding] = Field(default_factory=list)
    verified_source_ids: list[str] = Field(default_factory=list)
    failed_source_ids: list[str] = Field(default_factory=list)
    evidence_gate: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[str] = Field(default_factory=list)
    output_path: str | None = None
    failure_reason: str | None = None


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_fetcher(url: str, timeout_seconds: float) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ActSoft-TCAD-Agent/1.0 public-evidence-lookup",
            "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 200))
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read(1_000_000).decode(charset, errors="replace")
    return status, body


def strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_title(raw: str) -> str | None:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    if match:
        return strip_html(match.group(1))[:240]
    heading = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", raw)
    return strip_html(heading.group(1))[:240] if heading else None


def text_snippet(text: str, terms: list[str], limit: int = 520) -> str:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms if term and lowered.find(term.lower()) >= 0]
    start = max(0, min(positions) - 120) if positions else 0
    return text[start : start + limit].strip()


def base_query_terms(goal_text: str, simulator: str | None) -> list[str]:
    terms = []
    lowered = goal_text.lower()
    for token in [
        "sentaurus",
        "devsim",
        "quasistationary",
        "transient",
        "inspect",
        "breakdown",
        "leakage",
        "field",
        "ron",
        "dibl",
        "mosfet",
        "gan",
        "hemt",
        "bjt",
        "finfet",
        "igbt",
        "oxide",
        "trap",
        "lifetime",
    ]:
        if token in lowered:
            terms.append(token)
    if simulator:
        terms.append(simulator.lower())
    return list(dict.fromkeys(terms or ["tcad", "simulation", "extraction", "convergence"]))


def methodology_claims(text: str) -> list[str]:
    lowered = text.lower()
    claims: list[str] = []
    claim_patterns = [
        ("sentaurus_command_sections", ["file", "electrode", "solve"]),
        ("sentaurus_math_controls", ["math", "iterations", "notdamped"]),
        ("sentaurus_quasistationary_step_control", ["quasistationary", "initialstep", "maxstep", "minstep"]),
        ("sentaurus_transient_step_control", ["transient", "initialtime", "finaltime", "maxstep"]),
        ("sentaurus_plot_field_outputs", ["plot", "electricfield", "impactionization"]),
        ("devsim_capacitor_examples", ["capacitance", "1d", "2d", "capacitor"]),
        ("devsim_diode_examples", ["diode", "1d", "2d", "3d"]),
        ("devsim_related_bjt_3dmos_density_gradient", ["bjt", "3dmos", "density gradient"]),
        ("metric_extraction_flow", ["vti", "gm", "ion", "ioff", "dibl"]),
    ]
    for claim, tokens in claim_patterns:
        if all(token in lowered for token in tokens):
            claims.append(claim)
    return claims


def allowed_public_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def source_ids_for_request(request: PublicEvidenceLookupRequest) -> list[str]:
    if request.source_ids:
        return request.source_ids[: request.max_sources]
    dossier = build_public_evidence_dossier(
        request.goal_text,
        simulator=request.simulator,
        template_ids=request.template_ids,
        max_categories=3,
    )
    ids = [card.source_id for card in dossier.source_cards]
    if ids:
        return ids[: request.max_sources]
    return [source.source_id for source in public_tcad_sources()[: request.max_sources]]


def lookup_one_source(
    source_id: str,
    *,
    request: PublicEvidenceLookupRequest,
    terms: list[str],
    fetcher: Fetcher,
) -> PublicEvidenceFinding:
    source = get_public_tcad_source(source_id)
    if source is None:
        return PublicEvidenceFinding(
            source_id=source_id,
            name=source_id,
            url="",
            source_type="unknown",
            access="unknown",
            status="failed",
            failure_reason="source_id is not in the public TCAD registry",
        )
    if not allowed_public_url(source.url):
        return PublicEvidenceFinding(
            source_id=source.source_id,
            name=source.name,
            url=source.url,
            source_type=source.source_type,
            access=source.access,
            status="skipped",
            license_note=source.license_note,
            failure_reason="source URL is not public HTTP(S)",
        )
    if not request.live:
        return PublicEvidenceFinding(
            source_id=source.source_id,
            name=source.name,
            url=source.url,
            source_type=source.source_type,
            access=source.access,
            status="registry_only",
            license_note=source.license_note,
            matched_terms=[term for term in terms if term in " ".join([source.name, source.runnable_seed or "", *source.notes]).lower()],
            methodology_claims=[],
        )
    try:
        status, raw = fetcher(source.url, request.timeout_seconds)
        text = strip_html(raw)
        matched = [term for term in terms if term.lower() in text.lower()]
        claims = methodology_claims(text)
        return PublicEvidenceFinding(
            source_id=source.source_id,
            name=source.name,
            url=source.url,
            source_type=source.source_type,
            access=source.access,
            status="verified" if 200 <= status < 400 and (matched or claims) else "fetched_no_match",
            live_checked=True,
            fetched_at=utc_timestamp(),
            http_status=status,
            title=extract_title(raw),
            snippet=text_snippet(text, [*terms, *claims]),
            matched_terms=matched,
            methodology_claims=claims,
            license_note=source.license_note,
        )
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError) as exc:
        return PublicEvidenceFinding(
            source_id=source.source_id,
            name=source.name,
            url=source.url,
            source_type=source.source_type,
            access=source.access,
            status="failed",
            live_checked=True,
            fetched_at=utc_timestamp(),
            license_note=source.license_note,
            failure_reason=str(exc),
        )


def run_public_evidence_lookup(
    request: PublicEvidenceLookupRequest,
    *,
    fetcher: Fetcher | None = None,
) -> PublicEvidenceLookupResult:
    actual_fetcher = fetcher or default_fetcher
    terms = base_query_terms(request.goal_text, request.simulator)
    source_ids = source_ids_for_request(request)
    findings = [
        lookup_one_source(source_id, request=request, terms=terms, fetcher=actual_fetcher)
        for source_id in source_ids
    ]
    verified = [item.source_id for item in findings if item.status in {"verified", "registry_only", "fetched_no_match"}]
    live_verified = [item.source_id for item in findings if item.status == "verified"]
    failed = [item.source_id for item in findings if item.status == "failed"]
    if request.live:
        status = "completed" if live_verified else "completed_with_lookup_gaps" if findings else "no_sources"
    else:
        status = "completed" if verified else "no_sources"
    result = PublicEvidenceLookupResult(
        status=status,
        goal_text=request.goal_text,
        simulator=request.simulator,
        live=request.live,
        source_ids=source_ids,
        query_terms=terms,
        findings=findings,
        verified_source_ids=live_verified if request.live else verified,
        failed_source_ids=failed,
        evidence_gate={
            "gate": "live_public_evidence_lookup",
            "mode": "live_fetch" if request.live else "registry_only",
            "passed": bool(live_verified) if request.live else bool(verified),
            "source_count": len(findings),
            "verified_count": len(live_verified) if request.live else len(verified),
            "failed_count": len(failed),
        },
        guardrails=[
            "Use fetched public pages as methodology or interface evidence only.",
            "Do not copy proprietary simulator files, PDKs, calibrated decks, or licensed model content.",
            "If live lookup has no verified source for a new operation, pause or require user confirmation.",
        ],
    )
    if request.output_path is not None:
        output_path = request.output_path.expanduser().resolve()
        result.output_path = str(output_path)
        write_json(output_path, result.model_dump(mode="json"))
    return result
