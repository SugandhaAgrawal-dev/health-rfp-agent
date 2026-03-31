# rfp_parser.py
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class RFPStructure:
    title: str
    deadline: Optional[str]
    sections: list[dict]
    evaluation_criteria: list[dict]
    certifications_required: list[str]
    page_limit: Optional[int]

class RFPParser:
    def parse(self, raw_text: str) -> RFPStructure:
        return RFPStructure(
            title=self._extract_title(raw_text),
            deadline=self._extract_deadline(raw_text),
            sections=self._extract_sections(raw_text),
            evaluation_criteria=self._extract_criteria(raw_text),
            certifications_required=self._find_certs(raw_text),
            page_limit=self._extract_page_limit(raw_text)
        )

    def _extract_deadline(self, text: str) -> Optional[str]:
        pattern = r'due\s+(by|on|date)[:\s]+([A-Z][a-z]+ \d{1,2},?\s+\d{4})'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(2) if match else None

    def _extract_criteria(self, text: str) -> list[dict]:
        # finds "Technical Approach: 40 points" style patterns
        pattern = r'([A-Za-z\s]+):\s*(\d+)\s*(points|%)'
        return [{"criterion": m.group(1).strip(), "weight": int(m.group(2))}
                for m in re.finditer(pattern, text)]
