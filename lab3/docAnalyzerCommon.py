"""
Shared helpers for Intelligent Document Analysis lab
"""
import re
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

LAB_DIR = Path(__file__).resolve().parent
PDF_DIR = str(LAB_DIR / "10KsSmall")

# Original lab4 list, minus bare "clean" (matched Superfund "cleanup") and the
# hyphenated duplicates, plus transition-risk and climate-finance terms.
# Matching rules are in _keyword_pattern(): each keyword must start at a word
# boundary, and spaces and hyphens inside a keyword are interchangeable.
KEYWORDS = [
    "climate", "emission", "greenhouse", "ghg", "scope 1", "scope 2", "scope 3",
    "weather", "natural disaster", "act of god", "acts of god", "tornado",
    "hurricane", "earthquake", "wildfire", "drought", "flood", "temperature",
    "sea level", "carbon", "warming", "net zero",
    "methane", "precipitation", "co2", "green", "earth",
    "storm", "typhoon", "extreme heat", "heat wave",
    "clean energy", "clean fuel", "clean power", "clean technolog",
    "renewable energy", "renewable power", "energy transition",
    "regulatory credit", "zero-emission", "zev", "fuel economy",
    "sustainab", "esg", "environmental, social", "paris agreement", "tcfd",
]

# Acronyms that would hit unrelated words if matched as prefixes.
WHOLE_WORD_KEYWORDS = {"esg", "zev"}

EXPECTED_KEYS = [
    "climate_risks", "emissions", "targets_and_goals",
    "governance", "financials", "mitigation",
]

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False,
)


def list_pdfs(pdf_dir: str = PDF_DIR) -> List[str]:
    return sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))


def extract_text_from_pdf(file: str, pdf_dir: str = PDF_DIR) -> str:
    reader = PdfReader(os.path.join(pdf_dir, file))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str) -> List[str]:
    return textSplitter.split_text(text)


def _keyword_pattern(keyword: str) -> re.Pattern:
    words = re.split(r"[\s\-]+", keyword)
    body = r"[\s\-]+".join(re.escape(word) for word in words)
    end = r"s?\b" if keyword in WHOLE_WORD_KEYWORDS else ""
    return re.compile(r"\b" + body + end, re.IGNORECASE)


KEYWORD_PATTERNS = {kw: _keyword_pattern(kw) for kw in KEYWORDS}


def keyword_hit(chunk: str) -> bool:
    return any(pattern.search(chunk) for pattern in KEYWORD_PATTERNS.values())


def matched_keywords(chunk: str) -> List[str]:
    return [kw for kw, pattern in KEYWORD_PATTERNS.items() if pattern.search(chunk)]


class EvidenceFinding(BaseModel):
    """A finding paired with a verbatim quotation from the input chunk."""

    finding: str = Field(min_length=1, description="The extracted climate-related finding")
    evidence_quote: str = Field(
        min_length=1,
        description="An exact, contiguous quotation from the input text supporting the finding",
    )

    @field_validator("finding", "evidence_quote")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Findings and evidence quotes must not be blank")
        return value


class ClimateDisclosure(BaseModel):
    """Structured climate-disclosure extraction for a single document chunk."""

    climate_risks: List[EvidenceFinding] = Field(
        default_factory=list,
        description="Material climate-related threats to operations or financials",
    )
    emissions: List[EvidenceFinding] = Field(
        default_factory=list,
        description="GHG emissions disclosures (Scope 1/2/3), with numbers and units when present",
    )
    targets_and_goals: List[EvidenceFinding] = Field(
        default_factory=list,
        description="Reduction targets, net-zero commitments, or milestones, with dates",
    )
    governance: List[EvidenceFinding] = Field(
        default_factory=list,
        description="Board oversight and management responsibility for climate risk",
    )
    financials: List[EvidenceFinding] = Field(
        default_factory=list,
        description="Capital, costs, or R&D spending tied to climate initiatives or impacts",
    )
    mitigation: List[EvidenceFinding] = Field(
        default_factory=list,
        description="Transition plans, scenario analysis, frameworks, or technology investments",
    )

    @field_validator(
        "climate_risks", "emissions", "targets_and_goals",
        "governance", "financials", "mitigation",
        mode="after",
    )
    @classmethod
    def drop_parroted_description(
        cls, items: List[EvidenceFinding], info: ValidationInfo
    ) -> List[EvidenceFinding]:
        """Strips any list item that's just this field's own schema
        description echoed back as if it were extracted content.
 
        with_structured_output() puts each field's `description=` into the
        JSON schema shown to the model as guidance. A model that isn't
        confident anything in the chunk belongs in a category can parrot
        that description back as a "finding" instead of correctly returning
        an empty list -- that's exactly what happened on the costco10k run
        that prompted this validator. This is cleanup, not a retry trigger:
        it filters the bad item out and otherwise accepts the object. Raising
        here instead (to force extract_with_retry() to retry) would do
        nothing useful at temperature=0 -- the model would just parrot the
        same description again on every attempt.
        """
        field_description = cls.model_fields[info.field_name].description or ""
        target = _normalize(field_description)
        return [item for item in items if _normalize(item.finding) != target]

    def total_items(self) -> int:
        return sum(
            len(v)
            for v in (
                self.climate_risks, self.emissions, self.targets_and_goals,
                self.governance, self.financials, self.mitigation,
            )
        )

    def is_empty(self) -> bool:
        return self.total_items() == 0


def validate_evidence(disclosure: ClimateDisclosure, chunk: str) -> ClimateDisclosure:
    """Reject a response if any quotation is absent from its source chunk.

    Only whitespace is normalized: case, punctuation, numbers, and units
    must match. A matching quote establishes provenance, not that the
    model's interpretation or category is correct.
    """
    source = " ".join(chunk.split())
    for category in EXPECTED_KEYS:
        for index, item in enumerate(getattr(disclosure, category)):
            quote = " ".join(item.evidence_quote.split())
            if not quote or quote not in source:
                raise ValueError(
                    f"{category}[{index}].evidence_quote does not occur in the input chunk"
                )
    return disclosure


EVIDENCE_INSTRUCTIONS = """
Each category must be a list of objects with two fields: finding and evidence_quote.
For every finding, copy an exact, contiguous quotation from the input text into
evidence_quote. The quotation must support the finding; do not change its numbers,
units, spelling, or punctuation, and do not combine separate passages with ellipses.
The finding may be paraphrased, but its evidence_quote must be verbatim.
If no supporting quotation exists in this chunk, omit the finding. Empty lists
are valid. Never quote the instructions or schema descriptions as evidence.
"""

def _normalize(text: str) -> str:
    """Lowercases and collapses everything but letters/digits to single
    spaces, so trivial differences in punctuation/whitespace/case don't
    defeat an exact-match comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


EXTRACTION_INSTRUCTIONS = """You are a compliance analyst extracting climate-related data per the SEC's
March 2024 climate-related disclosure rule (Release Nos. 33-11275/34-99678).

Extract only information explicitly present in the text below. If a category has no
relevant data in this text, leave it empty. Use the detailed guidance below to decide
which category (if any) a given statement belongs in — many categories are easy to
confuse with each other, so the "not this" notes matter as much as the "this" notes.

climate_risks: material climate-related risks that have materially impacted, or are
reasonably likely to materially impact, the company's strategy, results of
operations, or financial condition. Covers physical risks (extreme weather, wildfire,
drought, flooding, sea-level rise affecting facilities, operations, or suppliers) and
transition risks (new regulation, carbon pricing, shifting demand, stranded assets,
reputational exposure), over both short-term (next 12 months) and long-term horizons.
Not this: routine risk-factor boilerplate that never reaches a material climate
impact; the actions taken in response (that's mitigation) or their dollar cost
(that's financials).

governance: board-level oversight of climate risk — which board committee or
subcommittee is responsible and how it is kept informed — plus management's role:
who is responsible, their relevant expertise, and how they report up to the board.
This is about oversight structure and reporting lines specifically, not the substance
of what risks exist or what's being done about them.

mitigation: the company's processes for identifying, assessing, and managing material
climate risks, and how those processes feed into its overall risk management system —
transition plans, scenario analysis (e.g. against a 1.5°C or 2°C warming scenario),
internal carbon pricing, adoption of a framework (TCFD, SBTi, GHG Protocol), or
specific technology or operational changes. Mitigation is the process and the action,
not a figure (emissions) or a dated commitment (targets_and_goals) — if a sentence is
really just an emissions number or a target date with no action described, it belongs
in one of those categories instead.

emissions: Scope 1 and Scope 2 greenhouse gas emissions, when material, expressed in
aggregate as CO2e in gross terms (excluding any purchased or generated offsets).
Scope 3 was dropped from the final rule entirely, and this requirement only applies to
Large Accelerated and Accelerated Filers — Smaller Reporting Companies and Emerging
Growth Companies are exempt, so don't expect (or invent) emissions figures for those
companies. Not this: a vague statement about reducing emissions with no figure or
methodology attached.

targets_and_goals: climate-related targets or goals that have materially affected, or
are reasonably likely to materially affect, the business, along with progress toward
them and material expenditures incurred pursuing them. A target only counts here if
it's a specific, material commitment — a date, a percentage, or a defined benchmark.
Not this: a vague aspirational statement ("committed to sustainability") — that
doesn't meet the rule's bar and usually isn't a finding for any category.

financials: the financial-statement effects the rule requires — capitalized costs,
expensed costs, charges, and losses from severe weather events and other natural
conditions, and the financial-statement effects of material climate-related targets
or transition activities, plus any resulting impact on financial estimates and
assumptions, all subject to the rule's materiality thresholds. This is strictly about
dollar impacts that flow through the financial statements, not a general cost or risk
mentioned elsewhere in the narrative disclosure.

Materiality is the operative threshold throughout the rule — if a statement wouldn't
clear a materiality bar on its own, it likely doesn't belong in any of these six
categories. If a single statement could reasonably fit more than one category, put it
in the most specific one that applies (a $50M investment in carbon capture technology
is mitigation, not financials, unless the sentence is really about the cost itself).

Never restate any of the guidance above as if it were a finding, in whole or in part —
not your own category's wording, and not another category's. Only include text that
actually appears (verbatim or closely paraphrased) in the chunk below. If nothing in
the text supports a category, leave it empty — an empty list is a correct, expected
answer, not a failure.

Text:
{text}
"""
