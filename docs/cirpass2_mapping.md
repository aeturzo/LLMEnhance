# CIRPASS-2 EU DPP Core Ontology Alignment

This document maps the terms in our hybrid DPP reasoning ontology
(`backend/ontologies/dpp_ontology.ttl`) to the **CIRPASS-2 EU DPP Core Ontology
Proposal** — *Ontology Requirements Specification v1.0*, March 12, 2025
(Maigre, Haav, Robal, Wolf, Danash; CIRPASS-2 Consortium;
DOI [10.5281/zenodo.14892666](https://doi.org/10.5281/zenodo.14892666);
CC-BY-4.0).

We treat the March 2025 document as the authoritative requirements-level
input for the EU DPP Core Ontology. The CIRPASS-2 consortium has indicated the
production-grade Core Ontology will be released as a follow-up; until then the
mapping below uses **placeholder IRIs in the `https://w3id.org/cirpass2/dpp/core#`
namespace** (matching the consortium's pre-announced namespace pattern). When
the formal TTL is released, the placeholder IRIs in `dpp_ontology.ttl` should
be replaced with the published ones in a single sed pass; the `skos:closeMatch`
links are designed to make that replacement low-risk.

## Why this alignment matters for the paper

A 2026 product-passport paper that does **not** reference CIRPASS-2 risks two
reviewer objections:

1. **Interoperability.** Reviewers will ask whether our OWL/RDF schema can be
   imported into a CIRPASS-2-aligned passport stack. Without an explicit
   mapping the answer is "unclear"; with the mapping below, the answer is
   "yes, modulo namespace re-targeting."
2. **Engineering relevance.** *Advanced Engineering Informatics* explicitly
   asks for industrial validation. CIRPASS-2 is the EU consortium that the
   European Commission has tasked with defining the cross-sector DPP
   information model. Showing that our schema lines up with the consortium's
   draft terms moves the paper from "ours alone" to "interoperable with the
   reference proposal."

## Term-by-term mapping

| Our term (`ex:`) | CIRPASS-2 placeholder (`cirpass2:`) | Mapping kind | Notes |
|---|---|---|---|
| `ex:Product` | `cirpass2:Product` | `skos:closeMatch` | Identity, identifiers (GTIN / serial / UID) and lifecycle linkage are aligned. CIRPASS-2 explicitly carries an `economicOperator` link that we treat as out-of-scope for the reasoning layer but expose in `dpp:hasManufacturer`. |
| `ex:Component` | `cirpass2:Component` | `skos:closeMatch` | Both represent physical-or-logical subparts that may carry their own passport-level claims. |
| `ex:Material` | `cirpass2:Material` | `skos:closeMatch` | CIRPASS-2 further distinguishes `Substance` (regulated chemicals) and `Material` (engineering material); we collapse these into one class because our reasoning rules only care about presence/absence of regulated substances at the component level. |
| `ex:ProcessStep` | `cirpass2:LifecycleProcess` | `skos:closeMatch` | Both describe stages such as design, manufacturing, use, repair, recycle, disposal. |
| `ex:Standard` | `cirpass2:Claim` | `skos:closeMatch` | CIRPASS-2 models a *Claim* as a verifiable assertion (compliance, warranty, certification). We expose compliance-target standards as `ex:Standard` and let the reasoner check whether the product `requiresCompliance` is satisfied by evidence; this is a Claim verification in CIRPASS-2 vocabulary. |
| `ex:requiresCompliance` | `cirpass2:hasClaim` (target = Claim) | `skos:closeMatch` | Property direction matches. |
| `ex:hasComponent` | `cirpass2:hasComponent` | `skos:closeMatch` |  |
| `ex:hasStep` | `cirpass2:hasLifecycleProcess` | `skos:closeMatch` |  |
| `ex:usesMaterial` | `cirpass2:hasMaterial` | `skos:closeMatch` |  |

## Carbon-ontology-side mapping

Our `carbon_ontology.ttl` introduces a parallel `carb:` namespace for the
carbon-footprint subsystem. CIRPASS-2's environmental-indicator submodel is
still being scoped, but the requirements document explicitly mentions
`CarbonFootprint` and `RecycledContent` as indicators carried by a DPP. The
following pre-emptive mapping is provided so the carbon subsystem can be
re-targeted with the same low-risk sed pass when the formal TTL appears:

| Our term (`carb:`) | CIRPASS-2 placeholder | Mapping kind |
|---|---|---|
| `carb:Product` | `cirpass2:Product` | `skos:exactMatch` |
| `carb:LifeCycleStage` | `cirpass2:LifecycleProcess` | `skos:closeMatch` |
| `carb:Result` (subclass `carb:TotalResult`) | `cirpass2:CarbonFootprintIndicator` | `skos:closeMatch` |
| `carb:EmissionFactor` | (no direct equivalent yet — CIRPASS-2 v1 does not model emission factors; they sit in the LCA-database layer) | `rdfs:seeAlso` only |
| `carb:Source` | `cirpass2:DataSource` | `skos:closeMatch` |
| `carb:DataStatus` | `cirpass2:DataQualityStatus` | `skos:closeMatch` |

## What is intentionally NOT mapped

These terms in our ontology have no CIRPASS-2 equivalent because they are
internal to the reliability-oriented reasoning layer and not part of the
passport itself:

- `ex:requiresStep`, `ex:hasStep` (auxiliary obligation reasoning)
- `carb:CalculationStep` (an audit-trail object for the deterministic carbon
  calculator)
- `carb:FactorSet` (a curated emission-factor table — an LCA-database object,
  not a passport object)

## How to upgrade when CIRPASS-2 publishes the formal TTL

```bash
# 1. Download the published CIRPASS-2 core ontology TTL.
# 2. Inspect its actual namespace (e.g. https://w3id.org/cirpass2/dpp/core#
#    or https://catalogue.cirpass2.eu/ontology/core#).
# 3. Replace the placeholder prefix in dpp_ontology.ttl and carbon_ontology.ttl:
sed -i 's#https://w3id.org/cirpass2/dpp/core#<actual published namespace>#g' \
    backend/ontologies/dpp_ontology.ttl backend/ontologies/carbon_ontology.ttl
# 4. Validate with rdflib + owlrl that the resulting graph still parses
#    and that the OWL-RL closure does not produce new contradictions.
# 5. Update this document with the actual term IRIs.
```

## How the paper should cite this

> "Our DPP ontology aligns with the CIRPASS-2 EU DPP Core Ontology Proposal
> (Maigre et al., 2025) at the level of the eight core terms identified in the
> requirements specification (Product, Component, Material, LifecycleProcess,
> Claim, hasComponent, hasMaterial, hasLifecycleProcess). The alignment is
> recorded with `skos:closeMatch` triples so that the schema can be re-targeted
> to the production CIRPASS-2 TTL with a single namespace rewrite."

## Source

Maigre, R., Haav, H.-M., Robal, T., Wolf, M.-A., Danash, F. (2025).
*Ontology Requirements Specification for an EU DPP Core Ontology Proposal*.
CIRPASS-2 Consortium. <https://doi.org/10.5281/zenodo.14892666>.
Published March 12, 2025. CC-BY-4.0.
