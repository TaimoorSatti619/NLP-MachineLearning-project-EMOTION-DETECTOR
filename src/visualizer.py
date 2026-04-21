import matplotlib.pyplot as plt
import matplotlib
from .config import EMOTION_COLORS, EMOTION_EMOJIS


def get_emotion_color(emotion: str) -> str:
    """Return hex color for a given emotion label."""
    return EMOTION_COLORS.get(emotion, '#666666')


def get_emotion_emoji(emotion: str) -> str:
    """Return emoji for a given emotion label."""
    return EMOTION_EMOJIS.get(emotion, '❓')


def plot_confidence_scores(probabilities: dict, title: str = "Emotion Confidence Distribution"):
    """
    Create a bar chart of emotion probabilities.

    ── BUG-8 FIX: old code used plt.subplots() with default white figure background.
    In Streamlit dark mode this produced a jarring white rectangle around the chart.
    Fix: set both the figure and axes face-color to 'none' (transparent) so the chart
    blends into whatever background color Streamlit uses (dark or light).
    Also set all text / spine colors to a neutral mid-gray that works in both themes.
    """
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions  = [item[0].title() for item in sorted_items]
    scores    = [item[1] * 100  for item in sorted_items]
    colors    = [get_emotion_color(item[0]) for item in sorted_items]

    # ── Transparent figure & axes so dark-mode Streamlit shows through
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('none')          # transparent figure background
    ax.set_facecolor('none')                 # transparent axes background

    bars = ax.bar(emotions, scores, color=colors, edgecolor='white', linewidth=1.5)

    # Neutral text color visible in both light and dark backgrounds
    TEXT_COLOR = '#CCCCCC'

    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.set_ylabel('Confidence Score (%)', fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=TEXT_COLOR)
    ax.set_ylim(0, 110)   # extra room for value labels
    ax.grid(axis='y', alpha=0.25, linestyle='--', color=TEXT_COLOR)

    # Value labels on top of each bar
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height + 1.5,
            f'{score:.1f}%',
            ha='center', va='bottom',
            fontweight='bold', fontsize=10, color=TEXT_COLOR
        )

    # Make spines and tick labels match the neutral color
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    plt.xticks(rotation=0, ha='center')

    plt.tight_layout()
    return fig
