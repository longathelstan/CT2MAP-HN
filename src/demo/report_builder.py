# -*- coding: utf-8 -*-
"""Report generation for CT2MAP-HN.

Builds structured HTML (and optionally PDF) reports containing case
information, representative slice overlays, suspicious region tables,
triage recommendations, and uncertainty notices.

Example:
    >>> from src.demo.report_builder import ReportBuilder
    >>> rb = ReportBuilder("CASE_001", "outputs/reports/")
    >>> rb.add_case_info("CASE_001", {"age": 62, "sex": "M"})
    >>> rb.add_prediction(heatmap, candidates, 0.72, 0.15)
    >>> rb.add_representative_slices(ct_vol, heatmap_vol, num_slices=5)
    >>> rb.save("outputs/reports/CASE_001_report.html")
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Inline Jinja2-style HTML template (no external file dependency)
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CT2MAP-HN Report — {{ case_id }}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f5f7fa; color: #333; padding: 20px;
  }
  .container { max-width: 960px; margin: 0 auto; }
  h1 { color: #2c3e50; margin-bottom: 6px; }
  h2 { color: #34495e; margin: 24px 0 12px; border-bottom: 2px solid #3498db; padding-bottom: 4px; }
  .disclaimer {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
    padding: 12px 16px; margin: 16px 0; font-size: 0.9em;
  }
  .disclaimer strong { color: #856404; }
  .info-table, .candidates-table {
    width: 100%; border-collapse: collapse; margin: 10px 0;
  }
  .info-table td, .candidates-table th, .candidates-table td {
    border: 1px solid #dee2e6; padding: 8px 12px;
  }
  .candidates-table th { background: #3498db; color: white; text-align: left; }
  .candidates-table tr:nth-child(even) { background: #f2f2f2; }
  .badge {
    display: inline-block; padding: 4px 12px; border-radius: 12px;
    color: white; font-weight: bold; font-size: 0.95em;
  }
  .badge-high { background: #e74c3c; }
  .badge-medium { background: #f39c12; }
  .badge-low { background: #2ecc71; }
  .slices-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
  .slices-grid img { max-width: 280px; border: 1px solid #ccc; border-radius: 4px; }
  .score-panel {
    display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
    margin: 10px 0;
  }
  .score-box {
    background: white; border-radius: 8px; padding: 16px 24px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center; min-width: 160px;
  }
  .score-box .label { font-size: 0.85em; color: #7f8c8d; }
  .score-box .value { font-size: 1.8em; font-weight: bold; margin-top: 4px; }
  .recommendation {
    background: white; border-left: 4px solid #3498db; padding: 12px 16px;
    margin: 10px 0; border-radius: 4px;
  }
  .footer {
    margin-top: 30px; padding-top: 12px; border-top: 1px solid #ddd;
    font-size: 0.8em; color: #95a5a6; text-align: center;
  }
</style>
</head>
<body>
<div class="container">

<h1>🧠 CT2MAP-HN Report</h1>
<p><strong>Case ID:</strong> {{ case_id }} &nbsp;|&nbsp; <strong>Generated:</strong> {{ timestamp }}</p>

<div class="disclaimer">
  <strong>⚠️ Disclaimer:</strong> Research-use only. Not for clinical diagnosis.
  <br>Chỉ dùng cho mục đích nghiên cứu. Không dùng để chẩn đoán lâm sàng.
</div>

{% if case_metadata %}
<h2>📋 Case Information</h2>
<table class="info-table">
{% for key, value in case_metadata.items() %}
  <tr><td style="font-weight:bold; width:200px;">{{ key }}</td><td>{{ value }}</td></tr>
{% endfor %}
</table>
{% endif %}

<h2>📊 Triage & Uncertainty</h2>
<div class="score-panel">
  <div class="score-box">
    <div class="label">Triage Score</div>
    <div class="value" style="color: {{ triage_color }};">{{ "%.3f"|format(triage_score) }}</div>
  </div>
  <div class="score-box">
    <div class="label">Uncertainty</div>
    <div class="value" style="color: {{ uncertainty_color }};">{{ "%.3f"|format(uncertainty_score) }}</div>
  </div>
  <div class="score-box">
    <div class="label">Risk Level</div>
    <div class="value"><span class="badge badge-{{ risk_level|lower }}">{{ risk_level }}</span></div>
  </div>
</div>

<div class="recommendation">
  <strong>Recommendation:</strong><br>
  {{ recommendation }}
</div>

{% if slice_images %}
<h2>🖼️ Representative Slices</h2>
<div class="slices-grid">
{% for img_b64 in slice_images %}
  <img src="data:image/png;base64,{{ img_b64 }}" alt="Slice">
{% endfor %}
</div>
{% endif %}

{% if lesion_candidates %}
<h2>🎯 Suspicious Regions</h2>
<table class="candidates-table">
  <thead>
    <tr>
      <th>#</th><th>Volume (voxels)</th><th>Centroid (z, y, x)</th>
      <th>Max Intensity</th><th>Mean Intensity</th>
    </tr>
  </thead>
  <tbody>
  {% for cand in lesion_candidates %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ cand.volume_voxels }}</td>
      <td>{{ cand.centroid }}</td>
      <td>{{ "%.3f"|format(cand.max_intensity) }}</td>
      <td>{{ "%.3f"|format(cand.get("mean_intensity", 0)) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<div class="footer">
  CT2MAP-HN v0.1.0 &mdash; Metabolic Risk Heatmap from Head-Neck CT<br>
  Generated {{ timestamp }}
</div>

</div>
</body>
</html>
"""


class ReportBuilder:
    """Builds an HTML report for a single inference case.

    Args:
        case_id: Unique case identifier.
        output_dir: Directory where report artefacts are saved.

    Example:
        >>> rb = ReportBuilder("CASE_001", "outputs/reports/")
        >>> rb.add_case_info("CASE_001", {"age": 62})
        >>> html = rb.build_html()
    """

    def __init__(self, case_id: str, output_dir: str) -> None:
        self.case_id = case_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._case_metadata: Dict[str, Any] = {}
        self._triage_score: float = 0.0
        self._uncertainty_score: float = 0.0
        self._lesion_candidates: List[Dict[str, Any]] = []
        self._slice_images: List[str] = []  # base64-encoded PNGs
        self._heatmap: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def add_case_info(
        self,
        case_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Add case-level metadata.

        Args:
            case_id: Case identifier (stored for consistency).
            metadata: Arbitrary key-value metadata (e.g. age, sex,
                scanner model).
        """
        self.case_id = case_id
        self._case_metadata.update(metadata)
        logger.debug("Added case info for %s", case_id)

    def add_prediction(
        self,
        heatmap: np.ndarray,
        lesion_candidates: List[Dict[str, Any]],
        triage_score: float,
        uncertainty_score: float,
    ) -> None:
        """Add prediction results.

        Args:
            heatmap: 3-D metabolic risk heatmap.
            lesion_candidates: List of candidate dicts.
            triage_score: Case-level triage score [0, 1].
            uncertainty_score: Overall uncertainty [0, 1].
        """
        self._heatmap = heatmap
        self._triage_score = triage_score
        self._uncertainty_score = uncertainty_score

        # Make candidates serialisable (remove slice objects)
        clean_candidates: List[Dict[str, Any]] = []
        for cand in lesion_candidates:
            cc = {}
            for k, v in cand.items():
                if k == "bounding_box":
                    # Convert slices to string
                    cc[k] = str(v)
                elif isinstance(v, tuple):
                    cc[k] = (
                        "("
                        + ", ".join(f"{c:.1f}" if isinstance(c, float) else str(c) for c in v)
                        + ")"
                    )
                else:
                    cc[k] = v
            clean_candidates.append(cc)
        self._lesion_candidates = clean_candidates
        logger.debug("Added prediction (triage=%.3f)", triage_score)

    def add_representative_slices(
        self,
        ct_volume: np.ndarray,
        heatmap_volume: np.ndarray,
        num_slices: int = 5,
        alpha: float = 0.4,
    ) -> None:
        """Generate and store representative slice overlays.

        Evenly-spaced axial slices are extracted, overlaid with the
        heatmap, and encoded as base64 PNG strings for embedding in HTML.

        Args:
            ct_volume: 3-D CT volume (D, H, W).
            heatmap_volume: 3-D heatmap (same shape).
            num_slices: Number of slices to include.
            alpha: Overlay blending factor.
        """
        from src.demo.viewer_utils import create_overlay

        depth = ct_volume.shape[0]
        indices = np.linspace(
            int(depth * 0.15),
            int(depth * 0.85),
            num_slices,
            dtype=int,
        )

        self._slice_images = []
        for idx in indices:
            ct_slc = ct_volume[idx, :, :]
            hm_slc = heatmap_volume[idx, :, :]
            overlay = create_overlay(ct_slc, hm_slc, alpha=alpha)
            b64 = self._image_to_base64(overlay)
            self._slice_images.append(b64)

        logger.debug("Added %d representative slices", len(self._slice_images))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_html(self) -> str:
        """Render the report as an HTML string.

        Uses Jinja2 if available, otherwise falls back to simple string
        replacement.

        Returns:
            Complete HTML document as a string.
        """
        # Determine risk level and colours
        if self._triage_score >= 0.7:
            risk_level = "HIGH"
            triage_color = "#e74c3c"
        elif self._triage_score >= 0.4:
            risk_level = "MEDIUM"
            triage_color = "#f39c12"
        else:
            risk_level = "LOW"
            triage_color = "#2ecc71"

        if self._uncertainty_score >= 0.6:
            uncertainty_color = "#e74c3c"
        elif self._uncertainty_score >= 0.3:
            uncertainty_color = "#f39c12"
        else:
            uncertainty_color = "#2ecc71"

        from src.inference.postprocess import compute_triage_recommendation

        recommendation = compute_triage_recommendation(
            self._triage_score, self._uncertainty_score
        )

        try:
            from jinja2 import Template  # type: ignore[import-untyped]

            template = Template(_HTML_TEMPLATE)
            html = template.render(
                case_id=self.case_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                case_metadata=self._case_metadata,
                triage_score=self._triage_score,
                triage_color=triage_color,
                uncertainty_score=self._uncertainty_score,
                uncertainty_color=uncertainty_color,
                risk_level=risk_level,
                recommendation=recommendation.replace("\n", "<br>"),
                slice_images=self._slice_images,
                lesion_candidates=self._lesion_candidates,
            )
        except ImportError:
            logger.warning(
                "Jinja2 not found; using basic string replacement for report"
            )
            html = self._build_html_fallback(
                risk_level,
                triage_color,
                uncertainty_color,
                recommendation,
            )

        return html

    def _build_html_fallback(
        self,
        risk_level: str,
        triage_color: str,
        uncertainty_color: str,
        recommendation: str,
    ) -> str:
        """Build a simplified HTML report without Jinja2.

        Args:
            risk_level: Risk level string.
            triage_color: CSS colour for triage score.
            uncertainty_color: CSS colour for uncertainty.
            recommendation: Recommendation text.

        Returns:
            HTML string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Metadata rows
        meta_rows = ""
        for k, v in self._case_metadata.items():
            meta_rows += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>\n"

        # Candidate rows
        cand_rows = ""
        for i, c in enumerate(self._lesion_candidates, 1):
            cand_rows += (
                f"<tr><td>{i}</td>"
                f"<td>{c.get('volume_voxels', '')}</td>"
                f"<td>{c.get('centroid', '')}</td>"
                f"<td>{c.get('max_intensity', 0):.3f}</td>"
                f"<td>{c.get('mean_intensity', 0):.3f}</td></tr>\n"
            )

        # Slice images
        slice_html = ""
        for img_b64 in self._slice_images:
            slice_html += (
                f'<img src="data:image/png;base64,{img_b64}" '
                f'alt="Slice" style="max-width:280px;border:1px solid #ccc;">\n'
            )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>CT2MAP-HN Report — {self.case_id}</title></head>
<body style="font-family:sans-serif;padding:20px;max-width:960px;margin:auto;">
<h1>CT2MAP-HN Report</h1>
<p><b>Case:</b> {self.case_id} | <b>Generated:</b> {timestamp}</p>
<div style="background:#fff3cd;padding:10px;border:1px solid #ffc107;margin:10px 0;">
<b>⚠️</b> Research-use only. Not for clinical diagnosis.
</div>
<h2>Case Info</h2><table border="1" cellpadding="6">{meta_rows}</table>
<h2>Triage</h2>
<p>Score: <b style="color:{triage_color}">{self._triage_score:.3f}</b> |
Uncertainty: <b style="color:{uncertainty_color}">{self._uncertainty_score:.3f}</b> |
Risk: <b>{risk_level}</b></p>
<pre>{recommendation}</pre>
<h2>Slices</h2><div style="display:flex;flex-wrap:wrap;gap:8px;">{slice_html}</div>
<h2>Suspicious Regions</h2>
<table border="1" cellpadding="6">
<tr><th>#</th><th>Volume</th><th>Centroid</th><th>Max Int.</th><th>Mean Int.</th></tr>
{cand_rows}</table>
<hr><p style="font-size:0.8em;color:#999;">CT2MAP-HN v0.1.0 — {timestamp}</p>
</body></html>"""
        return html

    def build_pdf(self, output_path: str) -> None:
        """Generate a PDF report.

        Attempts to use ``weasyprint``.  If unavailable, saves the HTML
        report instead and logs a warning.

        Args:
            output_path: Destination PDF (or HTML fallback) path.
        """
        html = self.build_html()

        try:
            from weasyprint import HTML as WeasyprintHTML  # type: ignore[import-untyped]

            pdf_path = Path(output_path)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            WeasyprintHTML(string=html).write_pdf(str(pdf_path))
            logger.info("Saved PDF report to %s", pdf_path)

        except ImportError:
            logger.warning(
                "weasyprint not installed; saving HTML instead of PDF"
            )
            html_path = Path(output_path).with_suffix(".html")
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html, encoding="utf-8")
            logger.info("Saved HTML report (PDF fallback) to %s", html_path)

    def save(self, output_path: str) -> None:
        """Save the report to the specified path.

        Format is determined by file extension:
            - ``.html`` → HTML
            - ``.pdf`` → PDF (via ``build_pdf``)

        Args:
            output_path: Destination file path.

        Example:
            >>> rb.save("outputs/reports/CASE_001.html")
        """
        ext = Path(output_path).suffix.lower()

        if ext == ".pdf":
            self.build_pdf(output_path)
        else:
            html = self.build_html()
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            logger.info("Saved HTML report to %s", out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_base64(image: np.ndarray) -> str:
        """Encode a numpy RGB image as a base64 PNG string.

        Args:
            image: RGB image (H×W×3) as ``np.uint8``.

        Returns:
            Base64-encoded PNG string.
        """
        from PIL import Image  # type: ignore[import-untyped]

        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
