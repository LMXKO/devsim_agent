# Sentaurus Fixture Corpus

These fixtures are interface-contract examples for the ActSoft Sentaurus agent.
They are not copied proprietary decks and they are not calibrated device models.

The fixtures use publicly visible Sentaurus command-file concepts such as
`File`, `Electrode`, `Physics`, `Plot`, `Math`, `Solve`, `Quasistationary`,
`Goal`, and assignment syntax. They let the agent validate deck IR parsing,
semantic patching, CSV extraction contracts, artifact lineage, and long-run
control without requiring a local Sentaurus license.

The optional fake backend used by `tcad_agent.sentaurus_contract` produces
interface-only logs and CSV files. It is not a Sentaurus physics simulator.
