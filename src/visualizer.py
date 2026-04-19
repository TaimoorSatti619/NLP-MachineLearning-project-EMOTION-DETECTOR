# src/visualizer.py
from .config import EMOTION_PALETTE


def render_probability_bars(probs: dict, top_emotion: str) -> str:
    """Return HTML for probability bar chart."""
    html = ""
    for emo, p in sorted(probs.items(), key=lambda x: -x[1]):
        cfg = EMOTION_PALETTE.get(emo, {"color": "#5b6ef5", "emoji": "❓"})
        pct = p * 100
        bold = "font-weight:600;" if emo == top_emotion else "opacity:.65;"
        html += (
            f'<div class="prob-row">'
            f'<div class="prob-emo" style="{bold}color:{cfg["color"]}">{cfg["emoji"]} {emo}</div>'
            f'<div class="prob-bg"><div class="prob-fill" style="width:{pct:.1f}%;background:{cfg["color"]};"></div></div>'
            f'<div class="prob-val">{pct:.1f}%</div>'
            f'</div>'
        )
    return html


def render_result_card(emotion: str, confidence: float, theme: str = "dark") -> str:
    """Return HTML for the main result card.
    
    Args:
        emotion: predicted emotion label
        confidence: float 0-1
        theme: 'dark' or 'light' — controls gradient end color
    """
    cfg = EMOTION_PALETTE.get(emotion, {"emoji": "❓", "color": "#5b6ef5", "bg": "rgba(91,110,245,0.08)"})
    if confidence > 0.80:
        lvl = "Very High"
        bb, bc = ("rgba(52,217,195,0.15)", "#34d9c3")
    elif confidence > 0.60:
        lvl = "High"
        bb, bc = ("rgba(245,197,66,0.15)", "#f5c542")
    elif confidence > 0.40:
        lvl = "Moderate"
        bb, bc = ("rgba(249,125,168,0.15)", "#f97da8")
    else:
        lvl = "Low"
        bb, bc = ("rgba(107,120,158,0.15)", "#6b789e")

    # BUG 2 FIX: was hardcoded #0f1626 (near-black) which looks broken in light mode.
    # Use a theme-aware neutral end color instead.
    gradient_end = "#0f1626" if theme == "dark" else "#f5f7fc"
    text_color   = "#eef2fb" if theme == "dark" else "#1a2639"
    border_alpha = "22"

    return (
        f'<div class="result-box" style="border-color:{cfg["color"]}{border_alpha};'
        f'background:linear-gradient(160deg,{cfg["bg"]},{gradient_end});">'
        f'<div class="result-emoji" style="font-size:3.5rem;line-height:1;margin-bottom:12px;">{cfg["emoji"]}</div>'
        f'<div class="result-emo" style="font-size:1.8rem;font-weight:600;text-transform:capitalize;color:{cfg["color"]};margin-bottom:6px;">{emotion}</div>'
        f'<div class="result-conf" style="font-size:0.85rem;color:#7a8aa3;">Confidence &nbsp;{confidence*100:.1f}%</div>'
        f'<div class="conf-badge" style="display:inline-block;border-radius:20px;padding:4px 14px;'
        f'font-size:12px;font-weight:500;margin-top:10px;background:{bb};color:{bc};">{lvl} Confidence</div>'
        f'</div>'
    )