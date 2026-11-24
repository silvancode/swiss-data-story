from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Farbrollen nach Datenjob (validierte Referenzpalette):
# eine Serie -> Slot 1; Emphasis -> Slot 1 + Grau; mehrere Serien -> feste Slot-Reihenfolge
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
ACCENT = "#2a78d6"
DEEMPHASIS = "#c6c4bc"
SEQ_LOW, SEQ_HIGH = "#cde2fb", "#0d366b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

MAX_SERIES = 6
MAX_BARS = 14


def build_chart(df: pd.DataFrame, chart_plan: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_type = chart_plan.get("chart_type", "bar")

    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    renderers = {
        "line": _render_line,
        "area": _render_area,
        "bar": _render_bar,
        "horizontal_bar": _render_horizontal_bar,
        "lollipop": _render_lollipop,
        "grouped_bar": _render_grouped_bar,
        "stacked_bar": _render_stacked_bar,
        "scatter": _render_scatter,
        "histogram": _render_histogram,
        "heatmap": _render_heatmap,
    }
    renderer = renderers.get(chart_type, _render_bar)
    plotted = renderer(ax, df.copy(), chart_plan)

    _apply_chrome(fig, ax, chart_plan, chart_type)
    if plotted is not None:
        _draw_reference_line(ax, plotted, chart_plan, chart_type)
        _draw_annotations(ax, plotted, chart_plan, chart_type)

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------- chart types


def _render_line(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y, category = plan.get("x_column"), plan.get("y_column"), plan.get("category_column")
    if category and category in df.columns and category != x:
        grouped = _aggregate_by_category(df, x, y, category, plan)
        series_names = list(dict.fromkeys(grouped[category]))
        for idx, name in enumerate(series_names[:MAX_SERIES]):
            part = grouped[grouped[category] == name]
            ax.plot(
                _axis_str(part[x]), part[y],
                color=CATEGORICAL[idx % len(CATEGORICAL)], linewidth=2.0,
                solid_joinstyle="round", solid_capstyle="round",
                marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=str(name),
            )
        ax.legend(loc="best", fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
        result = grouped
    else:
        grouped = _aggregate(df, x, y, plan).sort_values(x)
        ax.plot(
            _axis_str(grouped[x]), grouped[y],
            color=ACCENT, linewidth=2.2, solid_joinstyle="round", solid_capstyle="round",
            marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.2,
        )
        _mark_line_highlights(ax, grouped, x, y, plan)
        result = grouped
    _set_labels(ax, plan, category_default=x, value_default=y)
    _thin_x_labels(ax, result[x].drop_duplicates())
    return result


def _render_area(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y = plan.get("x_column"), plan.get("y_column")
    grouped = _aggregate(df, x, y, plan).sort_values(x)
    xs = _axis_str(grouped[x])
    ax.plot(xs, grouped[y], color=ACCENT, linewidth=2.2, solid_capstyle="round")
    ax.fill_between(range(len(xs)), grouped[y], color=ACCENT, alpha=0.10, zorder=1)
    ax.set_ylim(bottom=min(0, float(grouped[y].min())))
    _mark_line_highlights(ax, grouped, x, y, plan)
    _set_labels(ax, plan, category_default=x, value_default=y)
    _thin_x_labels(ax, grouped[x])
    return grouped


def _render_bar(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y = plan.get("x_column"), plan.get("y_column")
    grouped = _aggregate(df, x, y, plan, default_sort="value_desc").head(MAX_BARS)
    colors, highlight_idx = _emphasis_colors(grouped, y, plan)
    ax.bar(_axis_str(grouped[x]), grouped[y], color=colors, width=0.62)
    _label_extremes_vertical(ax, grouped, y, highlight_idx)
    _set_labels(ax, plan, category_default=x, value_default=y)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return grouped


def _render_horizontal_bar(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y = plan.get("x_column"), plan.get("y_column")
    key = plan.get("category_column") or x
    grouped = _aggregate(df, key, y, plan, default_sort="value_desc").head(MAX_BARS).iloc[::-1].reset_index(drop=True)
    colors, highlight_idx = _emphasis_colors(grouped, y, plan, key_column=key)
    ax.barh(_axis_str(grouped[key]), grouped[y], color=colors, height=0.62)
    _label_extremes_horizontal(ax, grouped, y, highlight_idx)
    _set_labels(ax, plan, category_default=key, value_default=y, horizontal=True)
    return grouped


def _render_lollipop(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y = plan.get("x_column"), plan.get("y_column")
    key = plan.get("category_column") or x
    grouped = _aggregate(df, key, y, plan, default_sort="value_desc").head(MAX_BARS).iloc[::-1].reset_index(drop=True)
    colors, highlight_idx = _emphasis_colors(grouped, y, plan, key_column=key)
    labels = _axis_str(grouped[key])
    ax.hlines(labels, 0, grouped[y], color=colors, linewidth=2.0)
    ax.scatter(grouped[y], labels, color=colors, s=64, zorder=3, edgecolors=SURFACE, linewidths=1.4)
    _label_extremes_horizontal(ax, grouped, y, highlight_idx, offset_factor=0.02)
    _set_labels(ax, plan, category_default=key, value_default=y, horizontal=True)
    return grouped


def _render_grouped_bar(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y, category = plan.get("x_column"), plan.get("y_column"), plan.get("category_column")
    pivot = _pivot(df, x, y, category, plan, max_columns=4, max_rows=10)
    if pivot is None or pivot.empty:
        return _render_bar(ax, df, plan)
    positions = range(len(pivot.index))
    n = len(pivot.columns)
    width = 0.72 / n
    for idx, column in enumerate(pivot.columns):
        offsets = [p + (idx - (n - 1) / 2) * width for p in positions]
        ax.bar(offsets, pivot[column], width=width * 0.92, color=CATEGORICAL[idx % len(CATEGORICAL)], label=str(column))
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(v) for v in pivot.index])
    ax.legend(loc="best", fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    _set_labels(ax, plan, category_default=x, value_default=y)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return None


def _render_stacked_bar(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y, category = plan.get("x_column"), plan.get("y_column"), plan.get("category_column")
    pivot = _pivot(df, x, y, category, plan, max_columns=MAX_SERIES, max_rows=16, fold_tail=True)
    if pivot is None or pivot.empty:
        return _render_bar(ax, df, plan)
    bottom = pd.Series(0.0, index=pivot.index)
    for idx, column in enumerate(pivot.columns):
        ax.bar(
            [str(v) for v in pivot.index], pivot[column], bottom=bottom,
            color=CATEGORICAL[idx % len(CATEGORICAL)], width=0.62,
            edgecolor=SURFACE, linewidth=1.4, label=str(column),
        )
        bottom = bottom + pivot[column].fillna(0)
    ax.legend(loc="best", fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    _set_labels(ax, plan, category_default=x, value_default=y)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return None


def _render_scatter(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y = plan.get("x_column"), plan.get("y_column")
    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    ax.scatter(xs, ys, color=ACCENT, alpha=0.75, s=42, edgecolors=SURFACE, linewidths=1.0)
    ax.xaxis.set_major_formatter(NUMBER_FORMATTER)
    _set_labels(ax, plan, category_default=x, value_default=y)
    return None


def _render_histogram(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    y = plan.get("y_column")
    values = pd.to_numeric(df[y], errors="coerce").dropna()
    bins = min(24, max(8, int(values.nunique() ** 0.5)))
    ax.hist(values, bins=bins, color=ACCENT, edgecolor=SURFACE, linewidth=1.2)
    ax.xaxis.set_major_formatter(NUMBER_FORMATTER)
    value_label = str(plan.get("y_label") or y)
    unit = str(plan.get("unit") or "").strip()
    if unit and unit.lower() not in value_label.lower():
        value_label = f"{value_label} ({unit})"
    ax.set_xlabel(fill(value_label, 60), fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Anzahl Beobachtungen", fontsize=10, color=INK_SECONDARY)
    return None


def _render_heatmap(ax: plt.Axes, df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    x, y, category = plan.get("x_column"), plan.get("y_column"), plan.get("category_column")
    pivot = _pivot(df, x, y, category, plan, max_columns=24, max_rows=15, columns_from_x=True)
    if pivot is None or pivot.empty:
        return _render_bar(ax, df, plan)
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SEQ_LOW, SEQ_HIGH])
    mesh = ax.pcolormesh(pivot.values, cmap=cmap, edgecolors=SURFACE, linewidth=1.2)
    ax.set_xticks([i + 0.5 for i in range(len(pivot.columns))])
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=35, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(pivot.index))])
    ax.set_yticklabels([str(v) for v in pivot.index])
    colorbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02, format=NUMBER_FORMATTER)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
    value_label = str(plan.get("y_label") or y)
    unit = str(plan.get("unit") or "").strip()
    if unit and unit.lower() not in value_label.lower():
        value_label = f"{value_label} ({unit})"
    colorbar.set_label(fill(value_label, 40), fontsize=9, color=INK_SECONDARY)
    ax.grid(False)
    ax.set_xlabel(fill(str(plan.get("x_label") or x), 60), fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel(fill(str(category), 60), fontsize=10, color=INK_SECONDARY)
    return None


# ---------------------------------------------------------- data preparation


def _aggregate(df: pd.DataFrame, x: str, y: str, plan: dict[str, Any], default_sort: str = "x_asc") -> pd.DataFrame:
    work = df[[x, y]].copy().dropna()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[y])
    if work.empty:
        return pd.DataFrame({x: [], y: []})

    grouped = work.groupby(x, dropna=True)[y]
    aggregation = plan.get("aggregation", "mean")
    if aggregation == "sum":
        result = grouped.sum()
    elif aggregation == "median":
        result = grouped.median()
    elif aggregation == "count":
        result = grouped.count()
    else:
        result = grouped.mean()
    result = result.reset_index()

    sort = plan.get("sort") or default_sort
    if sort == "value_desc":
        result = result.sort_values(y, ascending=False)
    elif sort == "value_asc":
        result = result.sort_values(y, ascending=True)
    elif sort == "x_asc":
        result = result.sort_values(x)

    top_n = plan.get("top_n")
    if isinstance(top_n, (int, float)) and int(top_n) >= 3:
        result = result.head(int(top_n))
    return result.reset_index(drop=True)


def _aggregate_by_category(df: pd.DataFrame, x: str, y: str, category: str, plan: dict[str, Any]) -> pd.DataFrame:
    work = df[[category, x, y]].copy().dropna(subset=[category, x])
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[y])
    if work.empty:
        return pd.DataFrame({category: [], x: [], y: []})
    keep = (
        work.groupby(category, dropna=True)[y].max().sort_values(ascending=False).head(MAX_SERIES).index
    )
    work = work[work[category].isin(keep)]
    aggregation = plan.get("aggregation", "mean")
    grouped = work.groupby([category, x], dropna=True)[y]
    result = {"sum": grouped.sum, "median": grouped.median, "count": grouped.count}.get(aggregation, grouped.mean)()
    return result.reset_index().sort_values([category, x])


def _pivot(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str | None,
    plan: dict[str, Any],
    max_columns: int,
    max_rows: int,
    fold_tail: bool = False,
    columns_from_x: bool = False,
) -> pd.DataFrame | None:
    if not category or category not in df.columns or category == x:
        return None
    work = df[[x, category, y]].copy().dropna(subset=[x, category])
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.dropna(subset=[y])
    if work.empty:
        return None

    aggregation = plan.get("aggregation", "mean")
    agg_name = aggregation if aggregation in {"sum", "median", "count", "mean"} else "mean"
    column_key, row_key = (x, category) if columns_from_x else (category, x)

    top_columns = work.groupby(column_key)[y].sum().abs().sort_values(ascending=False).head(max_columns).index
    if fold_tail and work[column_key].nunique() > max_columns:
        work[column_key] = work[column_key].where(work[column_key].isin(top_columns), "Übrige")
    else:
        work = work[work[column_key].isin(top_columns)]

    top_rows = work.groupby(row_key)[y].sum().abs().sort_values(ascending=False).head(max_rows).index
    work = work[work[row_key].isin(top_rows)]

    pivot = work.pivot_table(index=row_key, columns=column_key, values=y, aggfunc=agg_name)
    try:
        pivot = pivot.sort_index()
    except TypeError:
        pass
    return pivot


# ------------------------------------------------------- emphasis & labeling


def _emphasis_colors(
    grouped: pd.DataFrame, value_column: str, plan: dict[str, Any], key_column: str | None = None
) -> tuple[list[str], set[int]]:
    highlight = plan.get("highlight") or {}
    key = key_column or plan.get("x_column")
    highlight_idx = _highlight_indices(grouped, value_column, key, highlight)
    if not highlight_idx:
        return [ACCENT] * len(grouped), set()
    # Emphasis-Form: die Story-Balken in der Akzentfarbe, der Rest zurueckhaltend grau
    colors = [ACCENT if idx in highlight_idx else DEEMPHASIS for idx in grouped.index]
    return colors, highlight_idx


def _highlight_indices(
    grouped: pd.DataFrame, value_column: str, key_column: str | None, highlight: dict[str, Any]
) -> set[int]:
    mode = str(highlight.get("mode", "none")).lower()
    if mode == "none" or grouped.empty or value_column not in grouped.columns:
        return set()

    if mode == "category":
        wanted = {str(v).strip().lower() for v in highlight.get("values", []) if str(v).strip()}
        if not wanted or not key_column or key_column not in grouped.columns:
            return set()
        keys = grouped[key_column].astype(str).str.strip().str.lower()
        return set(grouped.index[keys.isin(wanted)])

    n = int(highlight.get("n", 1) or 0)
    if n <= 0:
        return set()
    values = pd.to_numeric(grouped[value_column], errors="coerce")
    usable = grouped.loc[values.notna()].copy()
    usable[value_column] = values[values.notna()]
    if usable.empty:
        return set()
    if mode in {"max", "top"}:
        return set(usable.nlargest(n, value_column).index)
    if mode in {"min", "bottom"}:
        return set(usable.nsmallest(n, value_column).index)
    return set()


def _mark_line_highlights(ax: plt.Axes, grouped: pd.DataFrame, x: str, y: str, plan: dict[str, Any]) -> None:
    highlight = plan.get("highlight") or {}
    highlight_idx = _highlight_indices(grouped, y, x, highlight)
    if not highlight_idx:
        return
    marked = grouped.loc[sorted(highlight_idx)]
    positions = {value: pos for pos, value in enumerate(_axis_str(grouped[x]))}
    xs = [positions.get(value, 0) for value in _axis_str(marked[x])]
    ax.scatter(xs, marked[y], color=ACCENT, s=72, zorder=4, edgecolors=SURFACE, linewidths=1.6)
    for xpos, yval in zip(xs, marked[y]):
        ax.annotate(
            _format_number(float(yval)), (xpos, yval), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=9, color=INK, fontweight="bold",
        )


def _label_extremes_vertical(ax: plt.Axes, grouped: pd.DataFrame, y: str, highlight_idx: set[int]) -> None:
    for bar, idx in zip(ax.patches, grouped.index):
        if idx not in highlight_idx:
            continue
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height, _format_number(height),
            va="bottom", ha="center", fontsize=9, color=INK, fontweight="bold",
        )


def _label_extremes_horizontal(
    ax: plt.Axes, grouped: pd.DataFrame, y: str, highlight_idx: set[int], offset_factor: float = 0.01
) -> None:
    if not highlight_idx:
        return
    span = float(pd.to_numeric(grouped[y], errors="coerce").max() or 1)
    for idx in highlight_idx:
        value = float(grouped.loc[idx, y])
        ax.text(
            value + span * offset_factor, idx, f" {_format_number(value)}",
            va="center", ha="left", fontsize=9, color=INK, fontweight="bold",
        )


# ---------------------------------------------------- annotations & overlays


def _draw_reference_line(ax: plt.Axes, plotted: pd.DataFrame, plan: dict[str, Any], chart_type: str) -> None:
    reference = plan.get("reference_line")
    if not isinstance(reference, dict) or chart_type in {"heatmap", "stacked_bar", "grouped_bar"}:
        return
    y = plan.get("y_column")
    mode = str(reference.get("mode", "mean")).lower()
    if mode == "value":
        try:
            value = float(reference.get("value"))
        except (TypeError, ValueError):
            return
    else:
        values = pd.to_numeric(plotted.get(y), errors="coerce").dropna() if y in plotted.columns else pd.Series(dtype=float)
        if values.empty:
            return
        value = float(values.mean())

    label = str(reference.get("label") or ("Durchschnitt" if mode != "value" else "")).strip()
    horizontal = chart_type in {"horizontal_bar", "lollipop"}
    if horizontal:
        ax.axvline(value, color=INK_SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        if label:
            ax.text(value, ax.get_ylim()[1], f" {label}: {_format_number(value)}", fontsize=8, color=INK_SECONDARY, va="bottom", ha="left")
    else:
        ax.axhline(value, color=INK_SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        if label:
            ax.text(ax.get_xlim()[1], value, f"{label}: {_format_number(value)} ", fontsize=8, color=INK_SECONDARY, va="bottom", ha="right")


def _draw_annotations(ax: plt.Axes, plotted: pd.DataFrame, plan: dict[str, Any], chart_type: str) -> None:
    annotations = plan.get("annotations")
    if not isinstance(annotations, list) or chart_type in {"heatmap", "stacked_bar", "grouped_bar", "scatter", "histogram"}:
        return
    x = plan.get("x_column")
    key = plan.get("category_column") or x if chart_type in {"horizontal_bar", "lollipop"} else x
    y = plan.get("y_column")
    if key not in plotted.columns or y not in plotted.columns:
        return

    keys = _axis_str(plotted[key]).str.strip().str.lower()
    horizontal = chart_type in {"horizontal_bar", "lollipop"}
    for annotation in annotations[:2]:
        if not isinstance(annotation, dict):
            continue
        target = str(annotation.get("x", "")).strip().lower()
        text = str(annotation.get("text", "")).strip()
        if not target or not text:
            continue
        match = plotted.index[keys == target]
        if match.empty:
            continue
        idx = match[0]
        position = list(plotted.index).index(idx)
        value = float(pd.to_numeric(plotted.loc[idx, y], errors="coerce"))
        point = (value, position) if horizontal else (position, value)
        ax.annotate(
            fill(text[:90], 28), point,
            textcoords="offset points", xytext=(18, 18), fontsize=8.5, color=INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9, "shrinkA": 2, "shrinkB": 4},
            ha="left", va="bottom",
        )


# ------------------------------------------------------------------- chrome


def _apply_chrome(fig: plt.Figure, ax: plt.Axes, plan: dict[str, Any], chart_type: str) -> None:
    title = plan.get("title", "Swiss Data Story")
    subtitle = plan.get("subtitle", "")
    source = plan.get("source_note", "opendata.swiss")
    highlight = plan.get("highlight") or {}

    # Titel und Untertitel in Figur-Koordinaten setzen und den Platz dafuer aus der
    # Zeilenzahl berechnen - sonst bleibt bei einzeiligen Titeln eine grosse Luecke.
    title_text, subtitle_text = fill(title, 68), fill(subtitle, 110) if subtitle else ""
    fig_height = fig.get_figheight()
    title_lines = title_text.count("\n") + 1
    subtitle_lines = subtitle_text.count("\n") + 1 if subtitle_text else 0
    margin, title_line, subtitle_line, gap = 0.12, 0.34, 0.2, 0.08
    fig.text(0.06, 1 - margin / fig_height, title_text, ha="left", va="top",
             fontsize=17, fontweight="bold", color=INK)
    if subtitle_text:
        subtitle_top = margin + title_lines * title_line + gap
        fig.text(0.06, 1 - subtitle_top / fig_height, subtitle_text, ha="left", va="top",
                 fontsize=10, color=INK_SECONDARY)
    header = margin + title_lines * title_line + gap + subtitle_lines * subtitle_line + 0.25
    ax.text(
        0, -0.17, fill(_source_and_highlight_note(source, highlight), 135),
        transform=ax.transAxes, fontsize=8, color=MUTED, va="top",
    )

    if chart_type != "heatmap":
        horizontal = chart_type in {"horizontal_bar", "lollipop"}
        ax.grid(axis="x" if horizontal else "y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        if horizontal:
            ax.xaxis.set_major_formatter(NUMBER_FORMATTER)
        else:
            ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1 - header / fig_height))


def _set_labels(
    ax: plt.Axes, plan: dict[str, Any], category_default: Any, value_default: Any, horizontal: bool = False
) -> None:
    # x_label beschreibt immer die Kategorien-/Zeitachse, y_label immer die Kennzahl -
    # bei horizontalen Charts tauschen die matplotlib-Achsen, die Bedeutung nicht.
    category_label = str(plan.get("x_label") or category_default or "")
    value_label = str(plan.get("y_label") or value_default or "")
    unit = str(plan.get("unit") or "").strip()
    if unit and unit.lower() not in value_label.lower():
        value_label = f"{value_label} ({unit})" if value_label else unit
    x_label, y_label = (value_label, category_label) if horizontal else (category_label, value_label)
    ax.set_xlabel(fill(_nice_label(x_label), 60), fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel(fill(_nice_label(y_label), 60), fontsize=10, color=INK_SECONDARY)


def _nice_label(label: str) -> str:
    """Rohe Spaltennamen lesbarer machen: 'anzahl_total' -> 'Anzahl total'."""
    text = label.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def _source_and_highlight_note(source: str, highlight: dict[str, Any]) -> str:
    reason = str(highlight.get("reason", "")).strip()
    if reason and str(highlight.get("mode", "none")).lower() != "none":
        return f"Quelle: {source} | Hervorhebung: {reason}"
    return f"Quelle: {source}"


# ------------------------------------------------------------------ helpers


def _axis_str(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if len(numeric) and numeric.notna().all() and (numeric % 1 == 0).all():
        return numeric.astype(int).astype(str)
    return values.astype(str)


def _format_number(value: float, _pos: Any = None) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "'")
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


NUMBER_FORMATTER = FuncFormatter(_format_number)


def _thin_x_labels(ax: plt.Axes, values: pd.Series) -> None:
    if len(values) <= 12:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        return
    keep_every = max(1, len(values) // 10)
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_visible(idx % keep_every == 0)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
