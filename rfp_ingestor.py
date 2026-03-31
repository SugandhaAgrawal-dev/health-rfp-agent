# rfp_ingestor.py
import httpx, hashlib
from pathlib import Path

class RFPIngestor:
    async def fetch_sam_gov(self, naics_code="621"):
        """Pull health-sector RFPs from SAM.gov"""
        r = await httpx.AsyncClient().get(
            "https://api.sam.gov/opportunities/v2/search",
            params={"api_key": SAM_API_KEY, "ptype": "o", "ncode": naics_code}
        )
        return r.json()["opportunitiesData"]

    async def ingest_file(self, file_path: Path) -> str:
        """Hash + store file, return document_id"""
        doc_id = hashlib.md5(file_path.read_bytes()).hexdigest()
        # store to Cloud Storage / S3
        return doc_id
