# -*- coding: utf-8 -*-
"""CT2MAP-HN Streamlit Dashboard.

Interactive web application for metabolic risk heatmap inference and
visualization from head-neck CT scans.

Run with:
    $ streamlit run src/demo/app_streamlit.py -- --config configs/demo.yaml

Sections:
    1. Sidebar: Upload / select case, model settings
    2. CT Slice Viewer: Axial / Coronal / Sagittal with slider
    3. Heatmap Overlay: Blended view with opacity control
    4. Triage & Uncertainty Panel: Score gauges, risk badge, recommendation
    5. Report: Summary table + download button
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    """Streamlit app entry point."""
    import streamlit as st

    # ----- Page config -----
    st.set_page_config(
        page_title="CT2MAP-HN Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----- Disclaimer banner -----
    st.warning(
        "⚠️ **Chỉ dùng cho mục đích nghiên cứu. "
        "Không dùng để chẩn đoán lâm sàng.**\n\n"
        "*Research-use only. Not for clinical diagnosis.*"
    )

    st.title("🧠 CT2MAP-HN Dashboard")
    st.markdown(
        "Bản đồ nguy cơ chuyển hóa từ ảnh CT đầu-cổ — *Metabolic risk "
        "heatmap inference from head-neck CT*"
    )

    # ==================================================================
    # Sidebar
    # ==================================================================
    with st.sidebar:
        st.header("⚙️ Cài đặt / Settings")

        # Config path
        config_path = st.text_input(
            "Đường dẫn cấu hình (Config path)",
            value="configs/demo.yaml",
            help="Đường dẫn tới file YAML cấu hình / Path to YAML config",
        )

        # Checkpoint path
        checkpoint_path = st.text_input(
            "Đường dẫn checkpoint (Checkpoint path)",
            value="outputs/checkpoints/best.pt",
            help="Đường dẫn tới file checkpoint mô hình / Model checkpoint path",
        )

        # Device
        device = st.selectbox(
            "Thiết bị (Device)",
            options=["cuda", "cuda:0", "cuda:1", "cpu"],
            index=0,
        )

        # MC Dropout
        use_mc_dropout = st.checkbox(
            "Sử dụng MC Dropout (Enable MC Dropout)",
            value=False,
        )
        mc_passes = 20
        if use_mc_dropout:
            mc_passes = st.slider(
                "Số lần forward pass (MC passes)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )

        st.divider()

        # ----- Input source -----
        st.subheader("📁 Dữ liệu đầu vào / Input Data")
        input_mode = st.radio(
            "Chọn phương thức nhập / Input mode",
            options=["Tải lên file (Upload)", "Demo mẫu (Demo case)"],
            index=1,
        )

        ct_path: Optional[str] = None

        if input_mode == "Tải lên file (Upload)":
            uploaded = st.file_uploader(
                "Tải file CT (.nii.gz)",
                type=["nii", "nii.gz", "gz"],
            )
            if uploaded is not None:
                # Save to temp file
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".nii.gz", delete=False
                )
                tmp.write(uploaded.read())
                tmp.close()
                ct_path = tmp.name
                st.success(f"Đã tải file: {uploaded.name}")
        else:
            # Demo cases
            demo_dir = Path("data/demo")
            if demo_dir.exists():
                cases = sorted(
                    [p.name for p in demo_dir.iterdir() if p.is_dir()]
                )
                if cases:
                    selected = st.selectbox("Chọn ca mẫu / Select demo case", cases)
                    ct_file = demo_dir / selected / "ct.nii.gz"
                    if ct_file.exists():
                        ct_path = str(ct_file)
                    else:
                        st.warning(f"Không tìm thấy ct.nii.gz trong {demo_dir / selected}")
                else:
                    st.info("Không có ca mẫu trong data/demo/")
            else:
                st.info(
                    "Thư mục data/demo/ chưa tồn tại. "
                    "Hãy tạo thư mục và đặt dữ liệu mẫu vào đó."
                )

        st.divider()
        run_button = st.button(
            "🚀 Chạy suy luận (Run Inference)",
            type="primary",
            disabled=(ct_path is None),
        )

    # ==================================================================
    # Main content
    # ==================================================================

    if ct_path is None:
        st.info(
            "👈 Hãy chọn hoặc tải lên file CT ở thanh bên trái.\n\n"
            "*Select or upload a CT file from the sidebar.*"
        )
        return

    # ------------------------------------------------------------------
    # Load model (cached)
    # ------------------------------------------------------------------
    @st.cache_resource
    def _load_inferencer(
        _ckpt: str, _cfg_path: str, _device: str
    ) -> Any:
        """Load and cache the CaseInferencer.

        Args:
            _ckpt: Checkpoint path.
            _cfg_path: Config YAML path.
            _device: Torch device string.

        Returns:
            CaseInferencer instance.
        """
        from src.utils.config import load_config
        from src.inference.infer_case import CaseInferencer

        cfg = load_config(_cfg_path)
        return CaseInferencer(_ckpt, cfg, device=_device)

    # ------------------------------------------------------------------
    # Run inference on button click
    # ------------------------------------------------------------------
    if run_button:
        with st.spinner("Đang xử lý... / Processing..."):
            try:
                inferencer = _load_inferencer(
                    checkpoint_path, config_path, device
                )

                if use_mc_dropout:
                    results = inferencer.infer_with_mc_dropout(
                        ct_path, num_passes=mc_passes
                    )
                else:
                    results = inferencer.infer(ct_path)

                st.session_state["results"] = results
                st.session_state["ct_path"] = ct_path
                st.success(
                    f"✅ Hoàn tất! Thời gian: {results['elapsed_seconds']:.2f}s"
                )

            except Exception as exc:
                st.error(f"❌ Lỗi: {exc}")
                logger.error("Inference error: %s", exc, exc_info=True)
                return

    # Check if we have results to display
    if "results" not in st.session_state:
        if not run_button:
            st.info("Nhấn **Chạy suy luận** để bắt đầu / Press **Run Inference** to start.")
        return

    results = st.session_state["results"]
    heatmap: np.ndarray = results["heatmap"]
    uncertainty_map: Optional[np.ndarray] = results.get("uncertainty_map")

    # Load CT for visualization
    try:
        import SimpleITK as sitk  # type: ignore[import-untyped]

        ct_image = sitk.ReadImage(st.session_state["ct_path"])
        ct_volume = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    except Exception as exc:
        st.error(f"Không thể tải CT để hiển thị: {exc}")
        return

    # ==================================================================
    # Section 2: CT Slice Viewer
    # ==================================================================
    st.header("🔍 Xem lát cắt CT / CT Slice Viewer")

    from src.demo.viewer_utils import create_slice_viewer, create_overlay

    col_ax, col_cor, col_sag = st.columns(3)

    with col_ax:
        st.subheader("Axial")
        ax_idx = st.slider(
            "Lát cắt Axial",
            0,
            ct_volume.shape[0] - 1,
            ct_volume.shape[0] // 2,
            key="axial_slider",
        )
        ax_slice = create_slice_viewer(ct_volume, "axial", ax_idx)
        st.image(
            _normalize_for_display(ax_slice),
            caption=f"Axial slice {ax_idx}",
            use_container_width=True,
        )

    with col_cor:
        st.subheader("Coronal")
        cor_idx = st.slider(
            "Lát cắt Coronal",
            0,
            ct_volume.shape[1] - 1,
            ct_volume.shape[1] // 2,
            key="coronal_slider",
        )
        cor_slice = create_slice_viewer(ct_volume, "coronal", cor_idx)
        st.image(
            _normalize_for_display(cor_slice),
            caption=f"Coronal slice {cor_idx}",
            use_container_width=True,
        )

    with col_sag:
        st.subheader("Sagittal")
        sag_idx = st.slider(
            "Lát cắt Sagittal",
            0,
            ct_volume.shape[2] - 1,
            ct_volume.shape[2] // 2,
            key="sagittal_slider",
        )
        sag_slice = create_slice_viewer(ct_volume, "sagittal", sag_idx)
        st.image(
            _normalize_for_display(sag_slice),
            caption=f"Sagittal slice {sag_idx}",
            use_container_width=True,
        )

    # ==================================================================
    # Section 3: Heatmap Overlay
    # ==================================================================
    st.header("🌡️ Bản đồ nguy cơ / Heatmap Overlay")

    overlay_alpha = st.slider(
        "Độ trong suốt heatmap / Heatmap opacity",
        0.0,
        1.0,
        0.4,
        step=0.05,
        key="overlay_alpha",
    )

    # Match heatmap to CT slices (handle potential shape mismatch)
    try:
        col_o1, col_o2, col_o3 = st.columns(3)

        with col_o1:
            hm_ax = create_slice_viewer(heatmap, "axial", ax_idx) if ax_idx < heatmap.shape[0] else np.zeros_like(ax_slice)
            overlay_ax = create_overlay(ax_slice, hm_ax, alpha=overlay_alpha)
            st.image(overlay_ax, caption=f"Axial overlay {ax_idx}", use_container_width=True)

        with col_o2:
            hm_cor = create_slice_viewer(heatmap, "coronal", cor_idx) if cor_idx < heatmap.shape[1] else np.zeros_like(cor_slice)
            overlay_cor = create_overlay(cor_slice, hm_cor, alpha=overlay_alpha)
            st.image(overlay_cor, caption=f"Coronal overlay {cor_idx}", use_container_width=True)

        with col_o3:
            hm_sag = create_slice_viewer(heatmap, "sagittal", sag_idx) if sag_idx < heatmap.shape[2] else np.zeros_like(sag_slice)
            overlay_sag = create_overlay(sag_slice, hm_sag, alpha=overlay_alpha)
            st.image(overlay_sag, caption=f"Sagittal overlay {sag_idx}", use_container_width=True)

    except Exception as exc:
        st.warning(f"Không thể tạo overlay: {exc}")

    # ==================================================================
    # Section 4: Triage & Uncertainty Panel
    # ==================================================================
    st.header("📊 Phân loại & Độ bất định / Triage & Uncertainty")

    from src.inference.postprocess import compute_triage_recommendation

    triage_score = results["triage_score"]
    uncertainty_score = results["uncertainty_score"]

    # Risk level
    if triage_score >= 0.7:
        risk_level = "HIGH"
        risk_color = "red"
        risk_emoji = "🔴"
    elif triage_score >= 0.4:
        risk_level = "MEDIUM"
        risk_color = "orange"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "green"
        risk_emoji = "🟢"

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        # Triage gauge via plotly
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=triage_score,
                    title={"text": "Điểm phân loại / Triage Score"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.4], "color": "#2ecc71"},
                            {"range": [0.4, 0.7], "color": "#f39c12"},
                            {"range": [0.7, 1], "color": "#e74c3c"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": triage_score,
                        },
                    },
                )
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.metric("Triage Score", f"{triage_score:.3f}")

    with col_t2:
        st.metric("Độ bất định / Uncertainty", f"{uncertainty_score:.3f}")
        # Uncertainty bar
        try:
            from src.demo.viewer_utils import plot_uncertainty_bar

            fig_unc = plot_uncertainty_bar(uncertainty_score)
            st.pyplot(fig_unc)
        except Exception:
            pass

    with col_t3:
        st.markdown(f"### Mức nguy cơ / Risk Level")
        st.markdown(
            f"<h1 style='text-align:center;color:{risk_color};'>"
            f"{risk_emoji} {risk_level}</h1>",
            unsafe_allow_html=True,
        )

    # Recommendation
    recommendation = compute_triage_recommendation(triage_score, uncertainty_score)
    st.info(recommendation)

    # ==================================================================
    # Section 5: Report
    # ==================================================================
    st.header("📝 Báo cáo / Report")

    # Summary table
    candidates = results.get("lesion_candidates", [])
    if candidates:
        import pandas as pd  # type: ignore[import-untyped]

        rows = []
        for i, c in enumerate(candidates, 1):
            rows.append(
                {
                    "#": i,
                    "Thể tích (voxels)": c.get("volume_voxels", ""),
                    "Trọng tâm (z,y,x)": str(c.get("centroid", "")),
                    "Cường độ tối đa": f"{c.get('max_intensity', 0):.3f}",
                    "Cường độ trung bình": f"{c.get('mean_intensity', 0):.3f}",
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Không phát hiện vùng nghi ngờ / No suspicious regions detected.")

    # Download report
    st.subheader("📥 Tải báo cáo / Download Report")

    try:
        from src.demo.report_builder import ReportBuilder

        rb = ReportBuilder(
            case_id=Path(st.session_state.get("ct_path", "unknown")).stem,
            output_dir=tempfile.gettempdir(),
        )
        rb.add_case_info(
            rb.case_id,
            {
                "Triage Score": f"{triage_score:.3f}",
                "Uncertainty": f"{uncertainty_score:.3f}",
                "Risk Level": risk_level,
                "Num Candidates": len(candidates),
            },
        )
        rb.add_prediction(heatmap, candidates, triage_score, uncertainty_score)

        html_report = rb.build_html()
        st.download_button(
            label="📄 Tải báo cáo HTML / Download HTML Report",
            data=html_report,
            file_name=f"{rb.case_id}_report.html",
            mime="text/html",
        )
    except Exception as exc:
        st.warning(f"Không thể tạo báo cáo: {exc}")


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalise a 2-D image to [0, 255] uint8 for Streamlit display.

    Args:
        image: Input 2-D array.

    Returns:
        Normalised uint8 array.
    """
    img = image.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 0:
        img = (img - vmin) / (vmax - vmin) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


if __name__ == "__main__":
    main()
