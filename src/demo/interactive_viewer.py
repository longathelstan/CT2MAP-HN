# -*- coding: utf-8 -*-
"""Real-time HTML5/Canvas Medical CT Viewer for CT-OMNICARE.

Provides zero-latency, 60 FPS client-side slice scrubbing (Axial, Coronal, Sagittal),
mouse wheel scroll (PACS style), Cine play/pause loops, and transparent 3D multi-organ
mask overlay with dynamic opacity controls using pre-rendered sprite sheets.
"""

from __future__ import annotations

import base64
import colorsys
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import SimpleITK as sitk
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

# Standard high-contrast medical color map for key anatomical structures
# RGBA format with alpha ~ 210 for clean overlay blending
PRESET_COLOR_MAP: Dict[int, Tuple[int, int, int, int]] = {
    1: (34, 197, 94, 220),    # liver - Emerald Green (#22c55e)
    2: (14, 165, 233, 220),   # kidney - Sky Blue (#0ea5e9)
    3: (168, 85, 247, 220),   # spleen - Purple (#a855f7)
    4: (234, 179, 8, 220),    # pancreas - Amber/Yellow (#eab308)
    5: (249, 115, 22, 220),   # right kidney - Orange (#f97316)
    6: (239, 68, 68, 220),    # aorta - Red (#ef4444)
    7: (59, 130, 246, 220),   # inferior vena cava - Blue (#3b82f6)
    8: (244, 114, 182, 220),  # right adrenal gland - Pink (#f472b6)
    9: (236, 72, 153, 220),   # left adrenal gland - Hot Pink (#ec4899)
    10: (132, 204, 22, 220),  # gallbladder - Lime (#84cc16)
    11: (161, 161, 170, 220), # esophagus - Zinc Gray (#a1a1aa)
    12: (244, 63, 94, 220),   # stomach - Rose (#f43f5e)
    13: (251, 146, 60, 220),  # duodenum - Light Orange (#fb923c)
    14: (6, 182, 212, 220),   # left kidney - Cyan (#06b6d4)
    15: (99, 102, 241, 220),  # bladder - Indigo (#6366f1)
    16: (192, 132, 252, 220), # prostate or uterus - Violet (#c084fc)
    17: (56, 189, 248, 220),  # portal & splenic vein - Sky (#38bdf8)
    18: (168, 162, 158, 220), # rectum - Stone (#a8a29e)
    19: (251, 191, 36, 220),  # small bowel - Amber (#fbbf24)
    20: (56, 189, 248, 220),  # lung - Sky (#38bdf8)
    21: (226, 232, 240, 220), # bone - Light Slate (#e2e8f0)
    23: (239, 68, 68, 220),   # lung tumor - Red
    24: (239, 68, 68, 220),   # pancreatic tumor - Red
    26: (239, 68, 68, 220),   # hepatic tumor - Red
    28: (56, 189, 248, 220),  # left lung upper lobe - Sky Blue
    29: (14, 165, 233, 220),  # left lung lower lobe - Deep Sky
    30: (2, 132, 199, 220),   # right lung upper lobe - Blue
    31: (3, 105, 161, 220),   # right lung middle lobe - Darker Blue
    32: (12, 74, 110, 220),   # right lung lower lobe - Navy Blue
    57: (20, 184, 166, 220),  # trachea - Teal (#14b8a6)
    62: (245, 158, 11, 220),  # colon - Amber (#f59e0b)
    115: (225, 29, 72, 220),  # heart - Crimson (#e11d48)
    116: (16, 185, 129, 220), # left kidney cyst - Emerald
    117: (16, 185, 129, 220), # right kidney cyst - Emerald
    129: (220, 38, 38, 220),  # kidney mass - Red
    130: (220, 38, 38, 220),  # liver tumor - Red
}


def _get_color_for_label(label_id: int) -> Tuple[int, int, int, int]:
    """Return RGBA color tuple for any label ID."""
    if label_id in PRESET_COLOR_MAP:
        return PRESET_COLOR_MAP[label_id]
    # Deterministic golden-ratio HSL distribution for unmapped IDs
    h = (label_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255), 220


@st.cache_data(show_spinner=False)
def get_volume_sprites(
    file_path: str,
    vmin: float = -160.0,
    vmax: float = 240.0,
    max_slices: int = 400,
) -> Optional[Dict[str, Any]]:
    """Generate high-performance sprite sheets for Axial, Coronal, Sagittal planes.

    Cached by file_path so disk read and image encoding happen only once per CT scan.
    """
    if not file_path or not Path(file_path).is_file():
        return None

    try:
        sitk_img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        arr = np.nan_to_num(arr, nan=-1000.0)

        # Standard soft tissue window
        arr_norm = np.clip(arr, vmin, vmax)
        arr_u8 = ((arr_norm - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

        Z, Y, X = arr_u8.shape

        cols = 16
        tile_w = 180

        # Helper to encode plane into JPEG sprite sheet
        def _make_sprite(plane_arr: np.ndarray, t_w: int, t_h: int) -> Tuple[str, int, int, int]:
            n = plane_arr.shape[0]
            r_count = (n + cols - 1) // cols
            sprite = Image.new("L", (cols * t_w, r_count * t_h))
            for i in range(n):
                sl = Image.fromarray(plane_arr[i]).resize((t_w, t_h), Image.BILINEAR)
                r, c = divmod(i, cols)
                sprite.paste(sl, (c * t_w, r * t_h))
            buf = io.BytesIO()
            sprite.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}", n, cols, r_count

        # 1. Axial (Z slices)
        ax_uri, ax_n, ax_c, ax_r = _make_sprite(arr_u8, tile_w, tile_w)

        # 2. Coronal (Y slices) - aspect ratio Z / X
        t_h_cor = max(100, min(240, int(tile_w * (Z / X))))
        cor_arr = np.transpose(arr_u8, (1, 0, 2))
        cor_uri, cor_n, cor_c, cor_r = _make_sprite(cor_arr, tile_w, t_h_cor)

        # 3. Sagittal (X slices) - aspect ratio Z / Y
        t_h_sag = max(100, min(240, int(tile_w * (Z / Y))))
        sag_arr = np.transpose(arr_u8, (2, 0, 1))
        sag_uri, sag_n, sag_c, sag_r = _make_sprite(sag_arr, tile_w, t_h_sag)

        return {
            "axial": {"uri": ax_uri, "count": ax_n, "cols": ax_c, "tw": tile_w, "th": tile_w},
            "coronal": {"uri": cor_uri, "count": cor_n, "cols": cor_c, "tw": tile_w, "th": t_h_cor},
            "sagittal": {"uri": sag_uri, "count": sag_n, "cols": sag_c, "tw": tile_w, "th": t_h_sag},
            "shape": (Z, Y, X),
        }
    except Exception as exc:
        logger.error("Failed to generate sprites for %s: %s", file_path, exc)
        return None


@st.cache_data(show_spinner=False)
def get_mask_sprites(mask_path: str, target_shape: Optional[Tuple[int, int, int]] = None) -> Optional[Dict[str, Any]]:
    """Generate transparent PNG sprite sheets for 3D segmentation mask (VISTA3D).

    Encodes organ masks into RGBA PNG with Nearest-Neighbor interpolation to preserve
    crisp boundary edges. Includes organ metadata and color tags for UI legend.
    """
    if not mask_path or not Path(mask_path).is_file():
        return None

    try:
        sitk_img = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.int32)
        Z, Y, X = arr.shape

        unique_labels = sorted([int(x) for x in np.unique(arr) if x > 0])
        if not unique_labels:
            return None

        # Build color lookup table for fast array mapping
        max_id = max(unique_labels)
        lut = np.zeros((max_id + 1, 4), dtype=np.uint8)
        legend_info = []

        for lbl in unique_labels:
            rgba = _get_color_for_label(lbl)
            lut[lbl] = rgba
            hex_color = f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}"
            legend_info.append({
                "id": lbl,
                "hex": hex_color,
                "rgba": list(rgba),
            })

        # Map entire 3D volume to RGBA
        arr_clipped = np.clip(arr, 0, max_id)
        rgba_vol = lut[arr_clipped]  # Shape: (Z, Y, X, 4)

        cols = 16
        tile_w = 180

        def _make_mask_sprite(plane_rgba: np.ndarray, t_w: int, t_h: int) -> Tuple[str, int, int, int]:
            n = plane_rgba.shape[0]
            r_count = (n + cols - 1) // cols
            sprite = Image.new("RGBA", (cols * t_w, r_count * t_h), (0, 0, 0, 0))
            for i in range(n):
                # Only process slices that contain any non-zero mask pixels
                if np.any(plane_rgba[i, ..., 3] > 0):
                    sl = Image.fromarray(plane_rgba[i], mode="RGBA").resize((t_w, t_h), Image.NEAREST)
                    r, c = divmod(i, cols)
                    sprite.paste(sl, (c * t_w, r * t_h))
            buf = io.BytesIO()
            sprite.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}", n, cols, r_count

        # 1. Axial mask
        ax_uri, ax_n, ax_c, ax_r = _make_mask_sprite(rgba_vol, tile_w, tile_w)

        # 2. Coronal mask
        t_h_cor = max(100, min(240, int(tile_w * (Z / X))))
        cor_rgba = np.transpose(rgba_vol, (1, 0, 2, 3))
        cor_uri, cor_n, cor_c, cor_r = _make_mask_sprite(cor_rgba, tile_w, t_h_cor)

        # 3. Sagittal mask
        t_h_sag = max(100, min(240, int(tile_w * (Z / Y))))
        sag_rgba = np.transpose(rgba_vol, (2, 0, 1, 3))
        sag_uri, sag_n, sag_c, sag_r = _make_mask_sprite(sag_rgba, tile_w, t_h_sag)

        return {
            "axial": {"uri": ax_uri, "count": ax_n, "cols": ax_c, "tw": tile_w, "th": tile_w},
            "coronal": {"uri": cor_uri, "count": cor_n, "cols": cor_c, "tw": tile_w, "th": t_h_cor},
            "sagittal": {"uri": sag_uri, "count": sag_n, "cols": sag_c, "tw": tile_w, "th": t_h_sag},
            "labels": legend_info,
        }
    except Exception as exc:
        logger.error("Failed to generate mask sprites for %s: %s", mask_path, exc)
        return None


def render_mpr_interactive_viewer(
    file_path: str,
    mask_path: Optional[str] = None,
    height: int = 560,
    label_dict: Optional[Dict[int, str]] = None,
) -> None:
    """Render 3-panel real-time MPR viewer (Axial, Coronal, Sagittal) with optional mask overlay."""
    sprites = get_volume_sprites(file_path)
    if not sprites:
        st.warning("Không thể tạo dữ liệu xem 3D cho file này.")
        return

    mask_sprites = get_mask_sprites(mask_path) if mask_path else None
    has_mask = mask_sprites is not None

    # Prepare legend tags
    legend_html = ""
    if has_mask and mask_sprites.get("labels"):
        tags = []
        for item in mask_sprites["labels"]:
            lbl_id = item["id"]
            name = (label_dict.get(lbl_id) if label_dict else None) or f"ID {lbl_id}"
            tags.append(
                f'<span style="display:inline-flex; align-items:center; gap:5px; background:#1e293b; '
                f'border:1px solid #475569; padding:2px 8px; border-radius:4px; font-size:11px; margin:2px 4px;">'
                f'<span style="width:10px; height:10px; border-radius:50%; background:{item["hex"]};"></span>'
                f'<span>{name}</span></span>'
            )
        legend_html = (
            f'<div style="display:flex; flex-wrap:wrap; align-items:center; margin-bottom:8px; gap:4px;">'
            f'<span style="font-size:11px; color:#94a3b8; font-weight:600; margin-right:4px;">🧩 Nhãn tạng:</span>'
            f'{"".join(tags)}</div>'
        )

    # Encode mask config for JS
    mask_json = json.dumps(mask_sprites) if has_mask else "null"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: #090d16;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        overflow: hidden;
      }}
      .top-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 8px;
      }}
      .opacity-control {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 12px;
        color: #38bdf8;
      }}
      .opacity-control input[type=range] {{
        width: 140px;
        accent-color: #10b981;
      }}
      .mpr-container {{
        display: flex;
        gap: 12px;
        justify-content: space-between;
        width: 100%;
      }}
      .pane {{
        flex: 1;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
      }}
      .pane-header {{
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #38bdf8;
      }}
      .badge {{
        background: #1e293b;
        border: 1px solid #475569;
        color: #f1f5f9;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 11px;
      }}
      .btn-cine {{
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 4px;
        font-size: 11px;
        cursor: pointer;
        padding: 2px 6px;
        transition: all 0.2s;
      }}
      .btn-cine:hover {{
        background: #38bdf8;
        color: #0f172a;
      }}
      .canvas-wrap {{
        width: 100%;
        height: 310px;
        background: #000;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        cursor: ns-resize;
        user-select: none;
      }}
      canvas {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 4px;
        image-rendering: pixelated;
      }}
      .slider-bar {{
        width: 100%;
        margin-top: 8px;
      }}
      input[type=range] {{
        width: 100%;
        accent-color: #38bdf8;
        cursor: pointer;
        margin: 4px 0;
      }}
      .hint {{
        font-size: 10px;
        color: #64748b;
        margin-top: 2px;
      }}
    </style>
    </head>
    <body>
      <div class="top-bar">
        <div style="font-size:12px; color:#94a3b8;">
          <span>⚡ Viewer 60 FPS | Cuộn chuột để lướt lát cắt</span>
        </div>
        {"<div class='opacity-control'><span>Độ mờ mặt nạ (Mask): <b id='txt_opacity'>65%</b></span><input type='range' id='sl_mask_opacity' min='0' max='100' value='65'></div>" if has_mask else "<span style='font-size:11px; color:#64748b;'>CT Gốc (Chưa có Mask phân vùng)</span>"}
      </div>

      {legend_html}

      <div class="mpr-container">
        <!-- AXIAL -->
        <div class="pane">
          <div class="pane-header">
            <span>Axial Plane</span>
            <div>
              <button class="btn-cine" id="cine_ax" title="Tự động chạy lát cắt">▶ Play</button>
              <span class="badge" id="badge_ax">0 / {sprites['axial']['count']}</span>
            </div>
          </div>
          <div class="canvas-wrap" id="wrap_ax" title="Cuộn chuột để đổi lát cắt tức thì">
            <canvas id="cv_ax"></canvas>
          </div>
          <div class="slider-bar">
            <input type="range" id="sl_ax" min="0" max="{sprites['axial']['count'] - 1}" value="{sprites['axial']['count'] // 2}">
            <div class="hint">🖱️ Cuộn chuột hoặc kéo thanh trượt (60 FPS)</div>
          </div>
        </div>

        <!-- CORONAL -->
        <div class="pane">
          <div class="pane-header">
            <span>Coronal Plane</span>
            <div>
              <button class="btn-cine" id="cine_cor" title="Tự động chạy lát cắt">▶ Play</button>
              <span class="badge" id="badge_cor">0 / {sprites['coronal']['count']}</span>
            </div>
          </div>
          <div class="canvas-wrap" id="wrap_cor" title="Cuộn chuột để đổi lát cắt tức thì">
            <canvas id="cv_cor"></canvas>
          </div>
          <div class="slider-bar">
            <input type="range" id="sl_cor" min="0" max="{sprites['coronal']['count'] - 1}" value="{sprites['coronal']['count'] // 2}">
            <div class="hint">🖱️ Cuộn chuột hoặc kéo thanh trượt (60 FPS)</div>
          </div>
        </div>

        <!-- SAGITTAL -->
        <div class="pane">
          <div class="pane-header">
            <span>Sagittal Plane</span>
            <div>
              <button class="btn-cine" id="cine_sag" title="Tự động chạy lát cắt">▶ Play</button>
              <span class="badge" id="badge_sag">0 / {sprites['sagittal']['count']}</span>
            </div>
          </div>
          <div class="canvas-wrap" id="wrap_sag" title="Cuộn chuột để đổi lát cắt tức thì">
            <canvas id="cv_sag"></canvas>
          </div>
          <div class="slider-bar">
            <input type="range" id="sl_sag" min="0" max="{sprites['sagittal']['count'] - 1}" value="{sprites['sagittal']['count'] // 2}">
            <div class="hint">🖱️ Cuộn chuột hoặc kéo thanh trượt (60 FPS)</div>
          </div>
        </div>
      </div>

      <script>
        const config = {{
          axial: {sprites['axial']},
          coronal: {sprites['coronal']},
          sagittal: {sprites['sagittal']}
        }};

        const maskConfig = {mask_json};
        let globalMaskOpacity = 0.65;

        const opacitySl = document.getElementById('sl_mask_opacity');
        const opacityTxt = document.getElementById('txt_opacity');
        if (opacitySl) {{
          opacitySl.addEventListener('input', (e) => {{
            globalMaskOpacity = parseFloat(e.target.value) / 100.0;
            if (opacityTxt) opacityTxt.innerText = `${{e.target.value}}%`;
            if (window.redrawAll) window.redrawAll();
          }});
        }}

        const viewRedraws = [];
        window.redrawAll = () => {{
          viewRedraws.forEach(fn => fn());
        }};

        function setupViewer(key, cvId, slId, badgeId, cineId, wrapId) {{
          const cfg = config[key];
          const mCfg = (maskConfig && maskConfig[key]) ? maskConfig[key] : null;

          const cv = document.getElementById(cvId);
          const ctx = cv.getContext('2d');
          const sl = document.getElementById(slId);
          const badge = document.getElementById(badgeId);
          const cineBtn = document.getElementById(cineId);
          const wrap = document.getElementById(wrapId);

          cv.width = cfg.tw;
          cv.height = cfg.th;

          const img = new Image();
          let imgLoaded = false;
          img.src = cfg.uri;

          const maskImg = new Image();
          let maskLoaded = false;
          if (mCfg) {{
            maskImg.src = mCfg.uri;
            maskImg.onload = () => {{
              maskLoaded = true;
              drawSlice(parseInt(sl.value));
            }};
          }}

          function drawSlice(idx) {{
            if (idx < 0) idx = 0;
            if (idx >= cfg.count) idx = cfg.count - 1;
            const r = Math.floor(idx / cfg.cols);
            const c = idx % cfg.cols;

            // 1. Draw base CT image
            ctx.globalAlpha = 1.0;
            ctx.drawImage(img, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);

            // 2. Draw mask overlay if available
            if (mCfg && maskLoaded && globalMaskOpacity > 0.0) {{
              ctx.globalAlpha = globalMaskOpacity;
              ctx.drawImage(maskImg, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);
              ctx.globalAlpha = 1.0;
            }}

            badge.innerText = `${{idx + 1}} / ${{cfg.count}}`;
          }}

          viewRedraws.push(() => {{
            drawSlice(parseInt(sl.value));
          }});

          img.onload = () => {{
            imgLoaded = true;
            drawSlice(parseInt(sl.value));
          }};

          // Instant real-time slider
          sl.addEventListener('input', (e) => {{
            drawSlice(parseInt(e.target.value));
          }});

          // Mouse wheel scrub (PACS Workstation style)
          wrap.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const delta = Math.sign(e.deltaY);
            let val = parseInt(sl.value) + delta;
            if (val < 0) val = 0;
            if (val >= cfg.count) val = cfg.count - 1;
            sl.value = val;
            drawSlice(val);
          }}, {{ passive: false }});

          // Cine loop
          let cineInterval = null;
          cineBtn.addEventListener('click', () => {{
            if (cineInterval) {{
              clearInterval(cineInterval);
              cineInterval = null;
              cineBtn.innerText = '▶ Play';
            }} else {{
              cineBtn.innerText = '⏸ Pause';
              cineInterval = setInterval(() => {{
                let val = parseInt(sl.value) + 1;
                if (val >= cfg.count) val = 0;
                sl.value = val;
                drawSlice(val);
              }}, 45); // ~22 FPS
            }}
          }});
        }}

        setupViewer('axial', 'cv_ax', 'sl_ax', 'badge_ax', 'cine_ax', 'wrap_ax');
        setupViewer('coronal', 'cv_cor', 'sl_cor', 'badge_cor', 'cine_cor', 'wrap_cor');
        setupViewer('sagittal', 'cv_sag', 'sl_sag', 'badge_sag', 'cine_sag', 'wrap_sag');
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


def render_axial_interactive_viewer(
    file_path: str,
    mask_path: Optional[str] = None,
    height: int = 440,
    label_dict: Optional[Dict[int, str]] = None,
) -> None:
    """Render single Axial real-time interactive viewer with optional mask overlay."""
    sprites = get_volume_sprites(file_path)
    if not sprites:
        st.warning("Không thể hiển thị lát cắt.")
        return

    mask_sprites = get_mask_sprites(mask_path) if mask_path else None
    has_mask = mask_sprites is not None

    legend_html = ""
    if has_mask and mask_sprites.get("labels"):
        tags = []
        for item in mask_sprites["labels"]:
            lbl_id = item["id"]
            name = (label_dict.get(lbl_id) if label_dict else None) or f"ID {lbl_id}"
            tags.append(
                f'<span style="display:inline-flex; align-items:center; gap:4px; background:#1e293b; '
                f'border:1px solid #475569; padding:1px 6px; border-radius:3px; font-size:10px; margin:2px 3px;">'
                f'<span style="width:8px; height:8px; border-radius:50%; background:{item["hex"]};"></span>'
                f'<span>{name}</span></span>'
            )
        legend_html = (
            f'<div style="display:flex; flex-wrap:wrap; align-items:center; margin-bottom:6px; gap:2px;">'
            f'<span style="font-size:10px; color:#94a3b8; font-weight:600;">🧩 Nhãn:</span>'
            f'{"".join(tags)}</div>'
        )

    ax_data = sprites["axial"]
    ax_mask_data = mask_sprites["axial"] if has_mask else None
    mask_json = json.dumps(ax_mask_data) if has_mask else "null"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: #090d16;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        overflow: hidden;
      }}
      .pane {{
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
      }}
      .pane-header {{
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 13px;
        font-weight: 600;
        color: #38bdf8;
      }}
      .badge {{
        background: #1e293b;
        border: 1px solid #475569;
        color: #f1f5f9;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 11px;
      }}
      .btn-cine {{
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 4px;
        font-size: 11px;
        cursor: pointer;
        padding: 2px 6px;
        transition: all 0.2s;
        margin-right: 6px;
      }}
      .btn-cine:hover {{
        background: #38bdf8;
        color: #0f172a;
      }}
      .canvas-wrap {{
        width: 100%;
        height: 290px;
        background: #000;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        cursor: ns-resize;
        user-select: none;
      }}
      canvas {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 4px;
        image-rendering: pixelated;
      }}
      .slider-bar {{
        width: 100%;
        margin-top: 8px;
      }}
      input[type=range] {{
        width: 100%;
        accent-color: #38bdf8;
        cursor: pointer;
      }}
      .hint {{
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748b;
        margin-top: 3px;
      }}
    </style>
    </head>
    <body>
      <div class="pane">
        <div class="pane-header">
          <span>Axial Slice Preview (60 FPS)</span>
          <div>
            <button class="btn-cine" id="cine_ax" title="Tự động phát lát cắt liên tục">▶ Play</button>
            <span class="badge" id="badge_ax">0 / {ax_data['count']}</span>
          </div>
        </div>

        {legend_html}

        <div class="canvas-wrap" id="wrap_ax" title="Kéo chuột hoặc cuộn con lăn (Wheel) để xem lát cắt">
          <canvas id="cv_ax"></canvas>
        </div>
        <div class="slider-bar">
          <input type="range" id="sl_ax" min="0" max="{ax_data['count'] - 1}" value="{ax_data['count'] // 2}">
          <div class="hint">
            <span>🖱️ Cuộn chuột / Kéo mượt (60 FPS)</span>
            <span>Lát: <b><span id="txt_curr">{ax_data['count'] // 2}</span></b></span>
          </div>
          {"<div style='display:flex; justify-content:space-between; align-items:center; margin-top:4px; font-size:11px; color:#38bdf8;'><span>Độ mờ Mask: <b id='txt_opacity'>65%</b></span><input type='range' id='sl_mask_opacity' min='0' max='100' value='65' style='width:120px; accent-color:#10b981;'></div>" if has_mask else ""}
        </div>
      </div>

      <script>
        const cfg = {ax_data};
        const mCfg = {mask_json};
        let maskOpacity = 0.65;

        const cv = document.getElementById('cv_ax');
        const ctx = cv.getContext('2d');
        const sl = document.getElementById('sl_ax');
        const badge = document.getElementById('badge_ax');
        const txtCurr = document.getElementById('txt_curr');
        const cineBtn = document.getElementById('cine_ax');
        const wrap = document.getElementById('wrap_ax');

        const opSl = document.getElementById('sl_mask_opacity');
        const opTxt = document.getElementById('txt_opacity');
        if (opSl) {{
          opSl.addEventListener('input', (e) => {{
            maskOpacity = parseFloat(e.target.value) / 100.0;
            if (opTxt) opTxt.innerText = `${{e.target.value}}%`;
            drawSlice(parseInt(sl.value));
          }});
        }}

        cv.width = cfg.tw;
        cv.height = cfg.th;

        const img = new Image();
        let imgLoaded = false;
        img.src = cfg.uri;

        const maskImg = new Image();
        let maskLoaded = false;
        if (mCfg) {{
          maskImg.src = mCfg.uri;
          maskImg.onload = () => {{
            maskLoaded = true;
            drawSlice(parseInt(sl.value));
          }};
        }}

        function drawSlice(idx) {{
          if (idx < 0) idx = 0;
          if (idx >= cfg.count) idx = cfg.count - 1;
          const r = Math.floor(idx / cfg.cols);
          const c = idx % cfg.cols;

          // 1. Draw base CT
          ctx.globalAlpha = 1.0;
          ctx.drawImage(img, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);

          // 2. Draw mask overlay
          if (mCfg && maskLoaded && maskOpacity > 0.0) {{
            ctx.globalAlpha = maskOpacity;
            ctx.drawImage(maskImg, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);
            ctx.globalAlpha = 1.0;
          }}

          badge.innerText = `${{idx + 1}} / ${{cfg.count}}`;
          txtCurr.innerText = `${{idx}}`;
        }}

        img.onload = () => {{
          imgLoaded = true;
          drawSlice(parseInt(sl.value));
        }};

        // Real-time scrubbing (0ms lag)
        sl.addEventListener('input', (e) => {{
          drawSlice(parseInt(e.target.value));
        }});

        // Mouse wheel scroll (PACS style)
        wrap.addEventListener('wheel', (e) => {{
          e.preventDefault();
          const delta = Math.sign(e.deltaY);
          let val = parseInt(sl.value) + delta;
          if (val < 0) val = 0;
          if (val >= cfg.count) val = cfg.count - 1;
          sl.value = val;
          drawSlice(val);
        }}, {{ passive: false }});

        // Cine auto-player
        let cineInterval = null;
        cineBtn.addEventListener('click', () => {{
          if (cineInterval) {{
            clearInterval(cineInterval);
            cineInterval = null;
            cineBtn.innerText = '▶ Play';
          }} else {{
            cineBtn.innerText = '⏸ Pause';
            cineInterval = setInterval(() => {{
              let val = parseInt(sl.value) + 1;
              if (val >= cfg.count) val = 0;
              sl.value = val;
              drawSlice(val);
            }}, 40);
          }}
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


def render_omniseg_interactive_canvas(
    file_path: str,
    mask_path: Optional[str] = None,
    height: int = 680,
    label_dict: Optional[Dict[int, str]] = None,
    initial_points: Optional[List[Dict[str, Any]]] = None,
    allow_interactive_clicks: bool = True,
    propagate_slices: Optional[List[int]] = None,
) -> None:
    """Render full interactive Canvas for OmniSeg-CT with positive/negative prompt clicks.

    Supports:
      - Left Click: Positive prompt (+) marked in Green/Red
      - Right Click / Shift+Click: Negative prompt (-) marked in Cyan/Blue
      - Toggle Display Mode: Overlay Mask, Contour Outline, or CT Only
      - Opacity Slider (0% - 100%)
      - 3D Propagation view indicators
    """
    sprites = get_volume_sprites(file_path)
    if not sprites:
        st.warning("Không thể tạo dữ liệu xem 3D cho file này.")
        return

    mask_sprites = get_mask_sprites(mask_path) if mask_path else None
    has_mask = mask_sprites is not None
    mask_json = json.dumps(mask_sprites) if has_mask else "null"
    points_json = json.dumps(initial_points or [])

    # Prepare legend tags
    legend_html = ""
    if has_mask and mask_sprites.get("labels"):
        tags = []
        for item in mask_sprites["labels"]:
            lbl_id = item["id"]
            name = (label_dict.get(lbl_id) if label_dict else None) or f"ID {lbl_id}"
            tags.append(
                f'<span style="display:inline-flex; align-items:center; gap:5px; background:#1e293b; '
                f'border:1px solid #475569; padding:2px 8px; border-radius:4px; font-size:11px; margin:2px 4px;">'
                f'<span style="width:10px; height:10px; border-radius:50%; background:{item["hex"]};"></span>'
                f'<span>{name}</span></span>'
            )
        legend_html = (
            f'<div style="display:flex; flex-wrap:wrap; align-items:center; margin-bottom:8px; gap:4px;">'
            f'<span style="font-size:11px; color:#94a3b8; font-weight:600; margin-right:4px;">🧩 Nhãn nhận diện:</span>'
            f'{"".join(tags)}</div>'
        )

    init_ax = (initial_points[0].get("z") if initial_points else None)
    if init_ax is None or init_ax >= sprites['axial']['count']:
        init_ax = sprites['axial']['count'] // 2

    init_cor = (initial_points[0].get("y") if initial_points else None)
    if init_cor is None or init_cor >= sprites['coronal']['count']:
        init_cor = sprites['coronal']['count'] // 2

    init_sag = (initial_points[0].get("x") if initial_points else None)
    if init_sag is None or init_sag >= sprites['sagittal']['count']:
        init_sag = sprites['sagittal']['count'] // 2

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: #090d16;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        overflow: hidden;
      }}
      .control-panel {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
      }}
      .btn-mode {{
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #475569;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        cursor: pointer;
        transition: all 0.2s;
        margin-right: 4px;
      }}
      .btn-mode.active {{
        background: #0284c7;
        color: #ffffff;
        border-color: #38bdf8;
        font-weight: bold;
      }}
      .point-legend {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 11px;
      }}
      .dot-pos {{
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border: 2px solid #ffffff;
        border-radius: 50%;
        margin-right: 4px;
      }}
      .dot-neg {{
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #06b6d4;
        border: 2px solid #ffffff;
        border-radius: 50%;
        margin-right: 4px;
      }}
      .slider-box {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        color: #94a3b8;
      }}
      input[type=range] {{
        accent-color: #38bdf8;
        cursor: pointer;
      }}
      .mpr-container {{
        display: flex;
        gap: 12px;
        justify-content: space-between;
        width: 100%;
      }}
      .pane {{
        flex: 1;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
      }}
      .pane-header {{
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #38bdf8;
      }}
      .badge {{
        background: #1e293b;
        border: 1px solid #475569;
        color: #f1f5f9;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
      }}
      .canvas-wrap {{
        width: 100%;
        height: 340px;
        background: #000;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        cursor: crosshair;
        user-select: none;
      }}
      canvas {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 4px;
      }}
      .slice-bar {{
        width: 100%;
        margin-top: 8px;
      }}
      .point-info-bar {{
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 6px 12px;
        margin-top: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: #cbd5e1;
      }}
      .btn-clear {{
        background: #475569;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 3px 8px;
        cursor: pointer;
        font-size: 10px;
      }}
      .btn-clear:hover {{
        background: #ef4444;
      }}
    </style>
    </head>
    <body>
      <div class="control-panel">
        <div style="display:flex; align-items:center;">
          <button class="btn-mode active" id="btn_mode_overlay">Phủ màu (Mask Overlay)</button>
          <button class="btn-mode" id="btn_mode_contour">Đường viền (Contour)</button>
          <button class="btn-mode" id="btn_mode_raw">Chỉ ảnh CT</button>
        </div>

        <div class="point-legend">
          <span><span class="dot-pos"></span><b>Click Trái:</b> Điểm dương (+)</span>
          <span><span class="dot-neg"></span><b>Click Phải / Shift:</b> Điểm âm (-)</span>
        </div>

        <div class="slider-box">
          <span>Độ mờ Mask: <b id="txt_opacity">65%</b></span>
          <input type="range" id="sl_opacity" min="0" max="100" value="65">
        </div>
      </div>

      {legend_html}

      <div class="mpr-container">
        <!-- AXIAL -->
        <div class="pane">
          <div class="pane-header">
            <span>Axial Plane (Mặt cắt ngang)</span>
            <span class="badge" id="badge_ax">0 / {sprites['axial']['count']}</span>
          </div>
          <div class="canvas-wrap" id="wrap_ax" title="Click Trái: (+) Thêm | Click Phải: (-) Xóa | Cuộn chuột để chuyển lát">
            <canvas id="cv_ax"></canvas>
          </div>
          <div class="slice-bar">
            <input type="range" id="sl_ax" min="0" max="{sprites['axial']['count'] - 1}" value="{init_ax}" style="width:100%;">
          </div>
        </div>

        <!-- CORONAL -->
        <div class="pane">
          <div class="pane-header">
            <span>Coronal Plane (Mặt cắt trán)</span>
            <span class="badge" id="badge_cor">0 / {sprites['coronal']['count']}</span>
          </div>
          <div class="canvas-wrap" id="wrap_cor" title="Click Trái: (+) Thêm | Click Phải: (-) Xóa | Cuộn chuột để chuyển lát">
            <canvas id="cv_cor"></canvas>
          </div>
          <div class="slice-bar">
            <input type="range" id="sl_cor" min="0" max="{sprites['coronal']['count'] - 1}" value="{init_cor}" style="width:100%;">
          </div>
        </div>

        <!-- SAGITTAL -->
        <div class="pane">
          <div class="pane-header">
            <span>Sagittal Plane (Mặt cắt đứng dọc)</span>
            <span class="badge" id="badge_sag">0 / {sprites['sagittal']['count']}</span>
          </div>
          <div class="canvas-wrap" id="wrap_sag" title="Click Trái: (+) Thêm | Click Phải: (-) Xóa | Cuộn chuột để chuyển lát">
            <canvas id="cv_sag"></canvas>
          </div>
          <div class="slice-bar">
            <input type="range" id="sl_sag" min="0" max="{sprites['sagittal']['count'] - 1}" value="{init_sag}" style="width:100%;">
          </div>
        </div>
      </div>

      <div class="point-info-bar">
        <div>
          <span>📌 Danh sách điểm nhấp hiện tại: <b id="txt_point_count">0 điểm</b></span>
          <span id="txt_point_list" style="margin-left: 10px; color: #38bdf8;"></span>
        </div>
        <div>
          <button class="btn-clear" id="btn_clear_points">Xóa toàn bộ điểm</button>
        </div>
      </div>

      <script>
        const config = {{
          axial: {sprites['axial']},
          coronal: {sprites['coronal']},
          sagittal: {sprites['sagittal']},
          shape: {list(sprites['shape'])}
        }};

        const maskConfig = {mask_json};
        let userPoints = {points_json};
        let viewMode = 'overlay'; // 'overlay', 'contour', 'raw'
        let globalOpacity = 0.65;

        // UI Mode Buttons
        document.getElementById('btn_mode_overlay').onclick = () => setMode('overlay');
        document.getElementById('btn_mode_contour').onclick = () => setMode('contour');
        document.getElementById('btn_mode_raw').onclick = () => setMode('raw');

        function setMode(mode) {{
          viewMode = mode;
          document.querySelectorAll('.btn-mode').forEach(b => b.classList.remove('active'));
          if (mode === 'overlay') document.getElementById('btn_mode_overlay').classList.add('active');
          if (mode === 'contour') document.getElementById('btn_mode_contour').classList.add('active');
          if (mode === 'raw') document.getElementById('btn_mode_raw').classList.add('active');
          redrawAll();
        }}

        // Opacity Slider
        document.getElementById('sl_opacity').oninput = (e) => {{
          globalOpacity = parseFloat(e.target.value) / 100.0;
          document.getElementById('txt_opacity').innerText = `${{e.target.value}}%`;
          redrawAll();
        }};

        // Clear Points
        document.getElementById('btn_clear_points').onclick = () => {{
          userPoints = [];
          updatePointInfo();
          redrawAll();
        }};

        function updatePointInfo() {{
          document.getElementById('txt_point_count').innerText = `${{userPoints.length}} điểm`;
          const desc = userPoints.map(p => `${{p.type === 1 ? '(+)' : '(-)'}} [${{p.z}},${{p.y}},${{p.x}}]`).join(' | ');
          document.getElementById('txt_point_list').innerText = desc ? desc : 'Chưa có điểm nhấp';
        }}
        updatePointInfo();

        const redrawFns = [];
        function redrawAll() {{
          redrawFns.forEach(fn => fn());
        }}

        function setupInteractivePlane(key, cvId, slId, badgeId, wrapId, axisDim) {{
          const cfg = config[key];
          const mCfg = (maskConfig && maskConfig[key]) ? maskConfig[key] : null;

          const cv = document.getElementById(cvId);
          const ctx = cv.getContext('2d');
          const sl = document.getElementById(slId);
          const badge = document.getElementById(badgeId);
          const wrap = document.getElementById(wrapId);

          cv.width = cfg.tw;
          cv.height = cfg.th;

          const origZ = config.shape[0];
          const origY = config.shape[1];
          const origX = config.shape[2];

          const img = new Image();
          let imgLoaded = false;
          img.src = cfg.uri;
          img.onload = () => {{
            imgLoaded = true;
            draw();
          }};

          const maskImg = new Image();
          let maskLoaded = false;
          if (mCfg) {{
            maskImg.src = mCfg.uri;
            maskImg.onload = () => {{
              maskLoaded = true;
              draw();
            }};
          }}

          function draw() {{
            const idx = parseInt(sl.value);
            const r = Math.floor(idx / cfg.cols);
            const c = idx % cfg.cols;

            // 1. Draw base CT
            ctx.globalAlpha = 1.0;
            ctx.drawImage(img, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);

            // 2. Draw mask / contour
            if (mCfg && maskLoaded && viewMode !== 'raw' && globalOpacity > 0.0) {{
              if (viewMode === 'overlay') {{
                ctx.globalAlpha = globalOpacity;
                ctx.drawImage(maskImg, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);
                ctx.globalAlpha = 1.0;
              }} else if (viewMode === 'contour') {{
                ctx.globalAlpha = 0.9;
                ctx.shadowColor = '#38bdf8';
                ctx.shadowBlur = 4;
                ctx.drawImage(maskImg, c * cfg.tw, r * cfg.th, cfg.tw, cfg.th, 0, 0, cv.width, cv.height);
                ctx.shadowBlur = 0;
                ctx.globalAlpha = 1.0;
              }}
            }}

            // 3. Draw Points that lie on or near this slice (+/- 2 slices) with proper coordinate scaling
            userPoints.forEach(p => {{
              let pSlice = 0;
              let ptCanvasX = 0, ptCanvasY = 0;

              if (key === 'axial') {{
                pSlice = p.z;
                ptCanvasX = p.x * (cv.width / origX);
                ptCanvasY = p.y * (cv.height / origY);
              }} else if (key === 'coronal') {{
                pSlice = p.y;
                ptCanvasX = p.x * (cv.width / origX);
                ptCanvasY = p.z * (cv.height / origZ);
              }} else if (key === 'sagittal') {{
                pSlice = p.x;
                ptCanvasX = p.y * (cv.width / origY);
                ptCanvasY = p.z * (cv.height / origZ);
              }}

              const sliceDist = Math.abs(idx - pSlice);
              if (sliceDist <= 2) {{
                // Draw point circle
                ctx.beginPath();
                const radius = sliceDist === 0 ? 5 : 3;
                ctx.arc(ptCanvasX, ptCanvasY, radius, 0, 2 * Math.PI);
                ctx.fillStyle = p.type === 1 ? '#22c55e' : '#06b6d4';
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();

                // Draw + or -
                if (sliceDist === 0) {{
                  ctx.fillStyle = '#ffffff';
                  ctx.font = 'bold 9px sans-serif';
                  ctx.fillText(p.type === 1 ? '+' : '-', ptCanvasX + 6, ptCanvasY - 4);
                }}
              }}
            }});

            badge.innerText = `${{idx + 1}} / ${{cfg.count}}`;
          }}

          redrawFns.push(draw);

          // Slider & wheel
          sl.oninput = () => draw();
          wrap.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const delta = Math.sign(e.deltaY);
            let val = parseInt(sl.value) + delta;
            val = Math.max(0, Math.min(cfg.count - 1, val));
            sl.value = val;
            draw();
          }}, {{ passive: false }});

          // Canvas Clicks with inverse coordinate scaling to Voxel space
          cv.addEventListener('click', (e) => handleCanvasClick(e, false));
          cv.addEventListener('contextmenu', (e) => {{
            e.preventDefault();
            handleCanvasClick(e, true);
          }});

          function handleCanvasClick(e, isRightClick) {{
            const rect = cv.getBoundingClientRect();
            const clickCanvasX = (e.clientX - rect.left) * (cv.width / rect.width);
            const clickCanvasY = (e.clientY - rect.top) * (cv.height / rect.height);
            const curSlice = parseInt(sl.value);

            let voxelZ = 0, voxelY = 0, voxelX = 0;
            if (key === 'axial') {{
              voxelZ = curSlice;
              voxelX = Math.round(clickCanvasX * (origX / cv.width));
              voxelY = Math.round(clickCanvasY * (origY / cv.height));
            }} else if (key === 'coronal') {{
              voxelY = curSlice;
              voxelX = Math.round(clickCanvasX * (origX / cv.width));
              voxelZ = Math.round(clickCanvasY * (origZ / cv.height));
            }} else if (key === 'sagittal') {{
              voxelX = curSlice;
              voxelY = Math.round(clickCanvasX * (origY / cv.width));
              voxelZ = Math.round(clickCanvasY * (origZ / cv.height));
            }}

            const pointType = (isRightClick || e.shiftKey) ? 0 : 1; // 1 = Pos (+), 0 = Neg (-)
            userPoints.push({{ z: voxelZ, y: voxelY, x: voxelX, type: pointType }});
            updatePointInfo();
            redrawAll();
          }}
        }}

        setupInteractivePlane('axial', 'cv_ax', 'sl_ax', 'badge_ax', 'wrap_ax', 0);
        setupInteractivePlane('coronal', 'cv_cor', 'sl_cor', 'badge_cor', 'wrap_cor', 1);
        setupInteractivePlane('sagittal', 'cv_sag', 'sl_sag', 'badge_sag', 'wrap_sag', 2);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)
