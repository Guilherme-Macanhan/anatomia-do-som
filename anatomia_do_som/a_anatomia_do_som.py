#!/usr/bin/env python3
"""A Anatomia do Som - infografico reproduzivel a partir do Spotify Tracks Dataset.

Uso:
    python a_anatomia_do_som.py
    python a_anatomia_do_som.py --csv /caminho/spotify-tracks-dataset-detailed.csv
    python a_anatomia_do_som.py --csv dados.csv --output-dir saida

Dependencias: pandas, numpy e matplotlib.

Tratamento e metodos:
- remove apenas linhas inteiras exatamente duplicadas;
- para estatisticas globais, resume cada track_id em uma observacao e usa a
  mediana de popularidade quando o mesmo ID tem mais de um valor no arquivo;
- para comparacoes entre generos, preserva cada associacao distinta faixa-genero;
- usa correlacao de Spearman para associacoes com popularidade e correlacao de
  Pearson entre as medias de energia e acousticness por genero;
- nao interpreta correlacoes como causalidade nem popularidade como qualidade.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Paleta editorial: creme, verde-escuro e laranja, com verde suave para apoio.
BG = "#F6F4ED"
INK = "#202D25"
GREEN = "#267A53"
GREEN_DARK = "#185A3D"
GREEN_PALE = "#C8DFCE"
LIME = "#DDEF79"
ORANGE = "#E97543"
ORANGE_PALE = "#F1C6AD"
MUTED = "#68746B"
GRID = "#D9DED5"
WHITE = "#FFFEFA"

PAGE_W, PAGE_H = 11.69, 8.27  # A4 horizontal, polegadas


def fmt_int_pt(value: int | float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def fmt_dec(value: float, digits: int = 1, sign: bool = False) -> str:
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    return fmt.format(float(value)).replace(".", ",")


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{fmt_dec(value * 100, digits)}%"


def fmt_duration(ms: float) -> str:
    seconds = int(round(float(ms) / 1000))
    return f"{seconds // 60}:{seconds % 60:02d}"


def axis_in(fig: plt.Figure, x: float, y: float, w: float, h: float):
    """Cria um eixo usando coordenadas em polegadas a partir do canto inferior esquerdo."""
    return fig.add_axes([x / PAGE_W, y / PAGE_H, w / PAGE_W, h / PAGE_H])


def fig_text(fig: plt.Figure, x: float, y: float, text: str, *, size=8,
             color=INK, weight="normal", ha="left", va="baseline",
             family="DejaVu Sans", linespacing=1.22, rotation=0, zorder=20):
    return fig.text(x / PAGE_W, y / PAGE_H, text, fontsize=size, color=color,
                    fontweight=weight, ha=ha, va=va, family=family,
                    linespacing=linespacing, rotation=rotation, zorder=zorder)


def style_axis(ax, *, grid_axis="y"):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelsize=6.2, length=2.2, width=0.6, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho via ranks, sem dependencia de scipy."""
    valid = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid["a"].rank(method="average").corr(valid["b"].rank(method="average")))


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    """Efeito rank-biserial/Cliff via U calculado das posicoes de ranking."""
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    n1, n0 = len(a), len(b)
    if n1 == 0 or n0 == 0:
        return float("nan")
    all_vals = pd.concat([a.rename("x"), b.rename("x")], ignore_index=True)
    ranks = all_vals.rank(method="average").iloc[:n1].sum()
    u = ranks - n1 * (n1 + 1) / 2
    return float(2 * u / (n1 * n0) - 1)


def load_and_analyze(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Nao encontrei o CSV: {csv_path}\n"
            "Coloque 'spotify-tracks-dataset-detailed.csv' na mesma pasta do script "
            "ou informe o caminho com --csv."
        )
    raw = pd.read_csv(csv_path, low_memory=False)
    required = {
        "track_id", "popularity", "duration_ms", "explicit", "danceability",
        "energy", "acousticness", "valence", "speechiness", "instrumentalness",
        "loudness", "tempo", "track_genre"
    }
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Colunas obrigatorias ausentes no CSV: {', '.join(missing_cols)}")
    if raw.empty:
        raise ValueError("O CSV nao contem registros.")

    exact_duplicates = int(raw.duplicated().sum())
    clean = raw.drop_duplicates().copy()
    clean["popularity"] = pd.to_numeric(clean["popularity"], errors="coerce")
    numeric_features = [
        "duration_ms", "danceability", "energy", "acousticness", "valence",
        "speechiness", "instrumentalness", "loudness", "tempo", "liveness"
    ]
    numeric_features = [c for c in numeric_features if c in clean.columns]
    for c in numeric_features:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean["explicit"] = clean["explicit"].astype("boolean")

    # Uma observacao por track_id nas estatisticas globais. Os atributos sonoros
    # sao constantes por ID nesta base; popularidade pode variar em poucos casos.
    aggregations = {c: "first" for c in numeric_features if c != "duration_ms"}
    aggregations.update({"popularity": "median", "duration_ms": "median", "explicit": "first"})
    tracks = clean.groupby("track_id", as_index=False, sort=False).agg(aggregations)

    # Comparacoes entre generos: uma observacao por par track_id x track_genre.
    genre_rows = clean.drop_duplicates(["track_id", "track_genre"]).copy()
    genre_stats = genre_rows.groupby("track_genre").agg(
        n=("track_id", "size"),
        mean_popularity=("popularity", "mean"),
        median_popularity=("popularity", "median"),
        zero_share=("popularity", lambda s: float((s == 0).mean())),
        mean_energy=("energy", "mean"),
        mean_acousticness=("acousticness", "mean"),
        mean_danceability=("danceability", "mean"),
        mean_valence=("valence", "mean"),
    ).sort_values("mean_popularity", ascending=False)

    correlations = {
        "Dancabilidade": spearman_corr(tracks["popularity"], tracks["danceability"]),
        "Energia": spearman_corr(tracks["popularity"], tracks["energy"]),
        "Acustica": spearman_corr(tracks["popularity"], tracks["acousticness"]),
        "Valencia": spearman_corr(tracks["popularity"], tracks["valence"]),
        "Fala": spearman_corr(tracks["popularity"], tracks["speechiness"]),
        "Instrumental": spearman_corr(tracks["popularity"], tracks["instrumentalness"]),
        "Intensidade sonora": spearman_corr(tracks["popularity"], tracks["loudness"]),
        "Tempo": spearman_corr(tracks["popularity"], tracks["tempo"]),
    }
    # Ordena os atributos por magnitude da associacao para revelar o que mais se destaca.
    correlations = dict(sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True))

    popularity = tracks["popularity"].dropna()
    explicit_false = tracks.loc[tracks["explicit"] == False, "popularity"].dropna()  # noqa: E712
    explicit_true = tracks.loc[tracks["explicit"] == True, "popularity"].dropna()  # noqa: E712

    return {
        "raw": raw,
        "clean": clean,
        "tracks": tracks,
        "genre_rows": genre_rows,
        "genre_stats": genre_stats,
        "exact_duplicates": exact_duplicates,
        "n_original": len(raw),
        "n_clean": len(clean),
        "n_tracks": int(tracks["track_id"].nunique()),
        "n_genres": int(genre_rows["track_genre"].nunique()),
        "median_popularity": float(popularity.median()),
        "mean_popularity": float(popularity.mean()),
        "share_zero": float((popularity == 0).mean()),
        "median_duration_ms": float(tracks["duration_ms"].median()),
        "correlations": correlations,
        "explicit_false": explicit_false,
        "explicit_true": explicit_true,
        "explicit_delta": cliffs_delta(explicit_true, explicit_false),
        "median_explicit": float(explicit_true.median()),
        "median_nonexplicit": float(explicit_false.median()),
        "genre_feature_corr": float(genre_stats["mean_energy"].corr(genre_stats["mean_acousticness"])),
        "n_explicit": int(len(explicit_true)),
        "n_nonexplicit": int(len(explicit_false)),
    }


def add_panel_heading(fig, x, y, title, subtitle, explanation, wrap_width=67):
    """Título, contexto e uma leitura em linguagem comum para cada gráfico."""
    fig_text(fig, x, y, title, size=8.45, color=INK, weight="bold")
    fig_text(fig, x, y - 0.18, subtitle, size=6.2, color=MUTED)
    explanation_lines = textwrap.fill(explanation, width=wrap_width)
    fig_text(fig, x, y - 0.36, explanation_lines, size=6.1, color=INK,
             linespacing=1.05, va="top")


def draw_popularity_distribution(fig, data):
    x, y, w, h = 0.46, 4.02, 3.34, 1.10
    ax = axis_in(fig, x, y, w, h)
    scores = data["tracks"]["popularity"].dropna().clip(0, 100)
    labels = ["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
    counts = [int((scores == 0).sum())]
    for lo in range(1, 101, 10):
        hi = min(lo + 9, 100)
        counts.append(int(((scores >= lo) & (scores <= hi)).sum()))
    pcts = np.array(counts, dtype=float) / len(scores) * 100
    xpos = np.arange(len(labels))
    colors = [ORANGE] + [GREEN] * (len(labels) - 1)
    bars = ax.bar(xpos, pcts, color=colors, width=.72, zorder=3)
    style_axis(ax, grid_axis="y")
    ax.set_ylim(0, max(pcts) * 1.2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=5.5)
    ax.set_ylabel("Faixas (%)", color=MUTED, fontsize=6.2, labelpad=2)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}%"))
    for label in ax.get_yticklabels():
        label.set_fontsize(5.6)
    ax.annotate(fmt_pct(data["share_zero"], 1),
                xy=(0, pcts[0]), xytext=(0.25, pcts[0] + max(pcts)*.16),
                textcoords="data", fontsize=6.4, weight="bold", color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=.7), ha="left")


def draw_correlations(fig, data):
    x, y, w, h = 4.17, 4.02, 3.39, 1.10
    ax = axis_in(fig, x, y, w, h)
    corr = data["correlations"]
    labels = list(corr.keys())[:8]
    values = np.array([corr[k] for k in labels])
    positions = np.arange(len(labels))
    colors = [GREEN if v >= 0 else ORANGE for v in values]
    ax.barh(positions, values, color=colors, height=.57, zorder=3)
    ax.axvline(0, color=INK, linewidth=.8, zorder=2)
    style_axis(ax, grid_axis="x")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(-.16, .11)
    ax.set_xticks([-.15, -.10, -.05, 0, .05, .10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: (f"{v:+.2f}").replace(".", ",")))
    for label in ax.get_xticklabels():
        label.set_fontsize(5.4)
    for yi, val in zip(positions, values):
        if val >= 0:
            ax.text(val + .004, yi, fmt_dec(val, 3, sign=True), va="center", ha="left",
                    fontsize=5.6, color=INK, clip_on=False)
        else:
            ax.text(val - .004, yi, fmt_dec(val, 3, sign=True), va="center", ha="right",
                    fontsize=5.6, color=INK, clip_on=False)


def draw_explicit_boxplots(fig, data):
    x, y, w, h = 7.92, 4.02, 3.31, 1.10
    ax = axis_in(fig, x, y, w, h)
    samples = [data["explicit_false"].values, data["explicit_true"].values]
    bp = ax.boxplot(samples, vert=False, positions=[0, 1], widths=.48,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.35),
                    whiskerprops=dict(color=MUTED, linewidth=.8),
                    capprops=dict(color=MUTED, linewidth=.8),
                    boxprops=dict(edgecolor=INK, linewidth=.8))
    for patch, color in zip(bp["boxes"], [GREEN_PALE, ORANGE_PALE]):
        patch.set_facecolor(color)
    style_axis(ax, grid_axis="x")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sem explícito", "Explícito"], fontsize=6.1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Popularidade (0-100)", fontsize=6.0, color=MUTED, labelpad=1)
    for label in ax.get_xticklabels():
        label.set_fontsize(5.6)
    # Marca as medianas de forma explicita, sem confundir a sobreposicao dos intervalos.
    for yy, med, col in [(0, data["median_nonexplicit"], GREEN_DARK),
                          (1, data["median_explicit"], ORANGE)]:
        ax.annotate(f"Md {med:.0f}".replace(".", ","), xy=(med, yy),
                    xytext=(med + 5, yy + .15), fontsize=5.6, color=col,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=.55))


def draw_genre_ranking(fig, data):
    x, y, w, h = 0.80, 1.50, 4.78, 1.43
    ax = axis_in(fig, x, y, w, h)
    stats = data["genre_stats"]
    top = stats.sort_values("mean_popularity", ascending=False).head(5)
    bottom = stats.sort_values("mean_popularity", ascending=True).head(5).sort_values("mean_popularity", ascending=False)
    # Dez faixas para destacar os extremos; o espaco entre os grupos e intencional.
    rows = []
    for idx, (name, row) in enumerate(top.iterrows()):
        rows.append((name, float(row["mean_popularity"]), 9 - idx, "top"))
    for idx, (name, row) in enumerate(bottom.iterrows()):
        rows.append((name, float(row["mean_popularity"]), 3 - idx, "bottom"))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    positions = [r[2] for r in rows]
    colors = [GREEN if r[3] == "top" else ORANGE for r in rows]
    ax.barh(positions, vals, color=colors, height=.59, zorder=3)
    style_axis(ax, grid_axis="x")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=6.1)
    ax.set_xlim(0, 68)
    ax.set_xticks([0, 20, 40, 60])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}"))
    for label in ax.get_xticklabels():
        label.set_fontsize(5.8)
    ax.set_ylim(-1.65, 9.65)
    # Separador entre os cinco maiores e cinco menores.
    ax.axhline(4, color=GRID, linewidth=.9, linestyle=(0, (2, 2)), zorder=1)
    for value, yy in zip(vals, positions):
        ax.text(value + 1.0, yy, fmt_dec(value, 1), va="center", ha="left",
                fontsize=5.9, color=INK, weight="bold")


def draw_sonic_map(fig, data):
    x, y, w, h = 5.91, 1.50, 5.32, 1.43
    ax = axis_in(fig, x, y, w, h)
    stats = data["genre_stats"].dropna(subset=["mean_energy", "mean_acousticness"])
    highlighted = set(stats.sort_values("mean_popularity", ascending=False).head(2).index)
    highlighted |= set(stats.sort_values("mean_energy", ascending=False).head(2).index)
    highlighted |= set(stats.sort_values("mean_acousticness", ascending=False).head(2).index)
    normal = stats.loc[~stats.index.isin(highlighted)]
    focus = stats.loc[stats.index.isin(highlighted)]
    ax.scatter(normal["mean_energy"], normal["mean_acousticness"], s=15,
               c=GREEN, alpha=.65, edgecolors=BG, linewidths=.35, zorder=3)
    # Codifica com laranja os extremos de som e com verde-lima os dois maiores scores medios.
    feature_extremes = set(stats.sort_values("mean_energy", ascending=False).head(2).index) | set(
        stats.sort_values("mean_acousticness", ascending=False).head(2).index)
    for name, row in focus.iterrows():
        color = LIME if name in set(stats.sort_values("mean_popularity", ascending=False).head(2).index) else ORANGE
        ax.scatter([row["mean_energy"]], [row["mean_acousticness"]], s=25,
                   c=color, edgecolors=INK, linewidths=.45, zorder=5)
    style_axis(ax, grid_axis="both")
    ax.set_xlim(0.12, 1.0)
    ax.set_ylim(-.02, 1.0)
    ax.set_xlabel("Energia média (0-1)", fontsize=6.0, color=MUTED, labelpad=1)
    ax.set_ylabel("Acústica média (0-1)", fontsize=6.0, color=MUTED, labelpad=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(5.5)
    corr = data["genre_feature_corr"]
    ax.text(.985, .97, f"r = {fmt_dec(corr, 2, sign=True)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.7,
            color=GREEN_DARK, weight="bold",
            bbox=dict(boxstyle="round,pad=.20", facecolor=BG, edgecolor=GRID, linewidth=.4))

    # Rotulos em pontos que ajudam a ler os extremos sem transformar o mapa em uma lista.
    label_specs = {
        "classical": (7, -7, "left", "center"),
        "romance": (7, 5, "left", "center"),
        "death-metal": (-60, 32, "left", "center"),
        "grindcore": (-60, 15, "left", "center"),
        "pop-film": (7, 8, "left", "center"),
        "k-pop": (7, -10, "left", "center"),
    }
    for name, (dx, dy, ha, va) in label_specs.items():
        if name in stats.index:
            row = stats.loc[name]
            ax.annotate(name, (row["mean_energy"], row["mean_acousticness"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=5.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=.45, shrinkA=1, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=.15", facecolor=BG,
                                  edgecolor="none", alpha=.88), zorder=8)


def write_summary(data, out_dir: Path):
    """Gera tabela de apoio para conferir os numeros usados no infografico."""
    records = [
        ("Registros originais", data["n_original"], "linhas do CSV"),
        ("Duplicatas exatas removidas", data["exact_duplicates"], "linhas"),
        ("Registros faixa-genero apos limpeza", data["n_clean"], "linhas"),
        ("IDs unicos", data["n_tracks"], "track_id distintos"),
        ("Generos", data["n_genres"], "rotulos distintos"),
        ("Popularidade media por ID", data["mean_popularity"], "escala 0-100; mediana por ID antes da media geral"),
        ("Popularidade mediana por ID", data["median_popularity"], "escala 0-100"),
        ("Participacao com popularidade zero", data["share_zero"], "proporcao de IDs"),
        ("Duracao mediana", data["median_duration_ms"], "milissegundos"),
        ("Correlacao energia-acustica entre medias de genero", data["genre_feature_corr"], "Pearson r"),
        ("Mediana popularidade - sem explicito", data["median_nonexplicit"], "escala 0-100"),
        ("Mediana popularidade - explicito", data["median_explicit"], "escala 0-100"),
        ("Cliff delta - explicito vs sem explicito", data["explicit_delta"], "efeito de ordenacao"),
    ]
    for label, value in data["correlations"].items():
        records.append((f"Spearman popularidade x {label}", value, "rho por track_id"))
    for column, count in data["raw"].isna().sum().items():
        if int(count) > 0:
            records.append((f"Valores ausentes - {column}", int(count),
                            "metadado descritivo; nao usado nos graficos" if column in {"artists", "album_name", "track_name"}
                            else "verificar antes de reutilizar a base"))
    pd.DataFrame(records, columns=["estatistica", "valor", "metodo_unidade"]).to_csv(
        out_dir / "a_anatomia_estatisticas.csv", index=False, encoding="utf-8-sig")
    data["genre_stats"].rename_axis("genero").to_csv(
        out_dir / "a_anatomia_perfil_generos.csv", encoding="utf-8-sig")


def create_infographic(data, pdf_path: Path, png_path: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.facecolor": BG,
    })
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)

    # Cabecalho editorial compacto: o espaco lateral vira uma chamada de descoberta.
    fig_text(fig, .46, 7.90, "MAPA SONORO  /  SPOTIFY TRACKS DATASET", size=7.2,
             color=GREEN, weight="bold")
    fig_text(fig, .46, 7.40, "A ANATOMIA DO SOM", size=27.0,
             color=INK, weight="bold")
    fig_text(fig, .48, 7.01,
             "Investigo como a popularidade se relaciona aos atributos sonoros, aos gêneros e ao conteúdo explícito.",
             size=8.1, color=MUTED)
    # O bloco superior direito explica o objetivo para leitores sem contexto estatístico.
    # A marca de vinil foi retirada para liberar espaço para esta nota editorial.
    fig.add_artist(plt.Line2D([7.19 / PAGE_W, 7.19 / PAGE_W],
                              [6.84 / PAGE_H, 7.67 / PAGE_H],
                              transform=fig.transFigure, color=ORANGE, linewidth=2.0))
    fig_text(fig, 7.36, 7.63, "SOBRE ESTE ESTUDO", size=6.1,
             color=GREEN, weight="bold")
    study_note = (
        f"Neste trabalho, analiso {fmt_int_pt(data['n_tracks'])} faixas distintas para investigar "
        "como a popularidade se relaciona com atributos sonoros e gêneros. Também comparo faixas com e sem conteúdo explícito."
    )
    fig_text(fig, 7.36, 7.40, textwrap.fill(study_note, width=72), size=6.5,
             color=INK, linespacing=1.08, va="top")
    dance_rho = data["correlations"].get("Dancabilidade", float("nan"))
    fig_text(fig, 7.36, 6.99,
             f"O QUE ENCONTREI  Dançabilidade x popularidade: ρ = {fmt_dec(dance_rho, 3, sign=True)} (associação muito fraca)",
             size=5.9, color=ORANGE, weight="bold")
    fig.add_artist(plt.Line2D([.46 / PAGE_W, 11.23 / PAGE_W],
                              [6.77 / PAGE_H, 6.77 / PAGE_H],
                              transform=fig.transFigure, color=GREEN, linewidth=2.0))

    # Indicadores-chave calculados a partir do CSV.
    kpis = [
        (fmt_int_pt(data["n_tracks"]), "IDs de faixa distintos"),
        (fmt_int_pt(data["n_genres"]), "gêneros no dataset"),
        (f"{fmt_dec(data['median_popularity'], 0)}/100", "mediana de popularidade"),
        (fmt_pct(data["share_zero"], 1), "com score igual a zero"),
        (fmt_duration(data["median_duration_ms"]), "duração mediana"),
    ]
    left, total_w, kpi_w = .46, 10.77, 10.77 / 5
    for i, (value, label) in enumerate(kpis):
        cx = left + i * kpi_w
        fig_text(fig, cx + .02, 6.46, value, size=18.8 if i != 2 else 16.8,
                 color=ORANGE if i == 3 else GREEN_DARK, weight="bold")
        fig_text(fig, cx + .03, 6.22, label, size=6.2, color=MUTED)
        if i > 0:
            fig.add_artist(plt.Line2D([cx / PAGE_W, cx / PAGE_W],
                                      [6.17 / PAGE_H, 6.68 / PAGE_H],
                                      transform=fig.transFigure, color=GRID, linewidth=.7))
    fig.add_artist(plt.Line2D([.46 / PAGE_W, 11.23 / PAGE_W],
                              [5.99 / PAGE_H, 5.99 / PAGE_H],
                              transform=fig.transFigure, color=GRID, linewidth=.8))

    # Primeira faixa: distribuições e associações, agora acompanhadas de instruções de leitura.
    add_panel_heading(
        fig, .46, 5.76, "01 / A CAUDA DA POPULARIDADE",
        "Distribuição do score entre IDs distintos",
        "Neste gráfico, mostro a porcentagem de faixas em cada intervalo de popularidade; a barra laranja destaca os scores zero.",
        wrap_width=62)
    add_panel_heading(
        fig, 4.17, 5.76, "02 / POPULARIDADE × SOM",
        f"Spearman ρ; dançabilidade = {fmt_dec(dance_rho, 3, sign=True)}",
        "Comparei oito atributos com a popularidade. À direita, a associação é positiva; à esquerda, negativa. Perto de zero, ela é fraca.",
        wrap_width=64)
    diff_med = data["median_explicit"] - data["median_nonexplicit"]
    add_panel_heading(
        fig, 7.92, 5.76, "03 / CONTEÚDO EXPLÍCITO",
        f"Mediana: {fmt_dec(data['median_nonexplicit'], 0)} vs {fmt_dec(data['median_explicit'], 0)} ({fmt_dec(diff_med, 0, sign=True)} pts)",
        "Comparei os dois grupos. A linha marca a mediana e a caixa, a metade central dos scores; a sobreposição indica distribuições semelhantes.",
        wrap_width=63)
    draw_popularity_distribution(fig, data)
    draw_correlations(fig, data)
    draw_explicit_boxplots(fig, data)
    dlt = data["explicit_delta"]
    effect_txt = "pequeno" if abs(dlt) < .15 else ("moderado" if abs(dlt) < .33 else "forte")

    # Segunda faixa: extremos de popularidade e paisagem sonora média.
    top = data["genre_stats"].sort_values("mean_popularity", ascending=False).iloc[0]
    bottom = data["genre_stats"].sort_values("mean_popularity", ascending=True).iloc[0]
    spread = float(top["mean_popularity"] - bottom["mean_popularity"])
    add_panel_heading(
        fig, .46, 3.55, "04 / O RANKING MUDA MUITO POR GÊNERO",
        "Score médio (0-100); verde = 5 maiores médias · laranja = 5 menores",
        "Aqui comparo as cinco maiores e menores médias desta base. O ranking descreve esta amostra, não a qualidade musical universal.",
        wrap_width=82)
    add_panel_heading(
        fig, 5.91, 3.55, "05 / MAPA DE IDENTIDADE SONORA",
        f"Energia e acústica médias por gênero; Pearson r = {fmt_dec(data['genre_feature_corr'], 2, sign=True)}",
        "Cada ponto representa um gênero. À direita há mais energia; acima, mais acústica. Encontrei uma associação negativa entre as médias.",
        wrap_width=90)
    draw_genre_ranking(fig, data)
    draw_sonic_map(fig, data)

    # Rodape: conclusoes e transparência metodologica.
    fig.add_artist(plt.Line2D([.46 / PAGE_W, 11.23 / PAGE_W],
                              [1.12 / PAGE_H, 1.12 / PAGE_H],
                              transform=fig.transFigure, color=GREEN, linewidth=1.7))
    fig_text(fig, .46, .94, "O QUE ENCONTREI", size=7.3, color=GREEN, weight="bold")
    corr_inst = data["correlations"].get("Instrumental", 0.0)
    corr_dance = data["correlations"].get("Dancabilidade", 0.0)
    left_note = (
        f"Encontrei uma associação muito fraca entre popularidade e dançabilidade (ρ = {fmt_dec(corr_dance, 3, sign=True)}). "
        f"A associação mais forte com a popularidade foi a instrumentalidade (ρ = {fmt_dec(corr_inst, 3, sign=True)}), mas também é fraca. "
        f"Entre os gêneros, as médias variam de {fmt_dec(bottom['mean_popularity'], 1)} ({bottom.name}) a {fmt_dec(top['mean_popularity'], 1)} ({top.name})."
    )
    left_lines = textwrap.fill(left_note, width=116)
    fig_text(fig, .46, .74, left_lines, size=5.95, color=INK, linespacing=1.20, va="top")

    fig_text(fig, 5.91, .94, "COMO ANALISEI OS DADOS", size=7.3, color=GREEN, weight="bold")
    right_paragraphs = [
        f"Removi {fmt_int_pt(data['exact_duplicates'])} duplicatas exatas. Na análise geral, contei {fmt_int_pt(data['n_tracks'])} IDs; por gênero, mantive {fmt_int_pt(data['n_clean'])} relações faixa-gênero. Uma faixa pode aparecer em mais de um gênero.",
        "Interpreto a popularidade como um score da plataforma, não como medida universal de qualidade; correlação não prova causalidade.",
        f"Na comparação de conteúdo explícito, a diferença entre medianas foi de {fmt_dec(diff_med, 0, sign=True)} pontos; o efeito é {effect_txt} (δ = {fmt_dec(dlt, 2, sign=True)}).",
        "Fonte: Spotify Tracks Dataset fornecido. Fiz a limpeza, a análise e a visualização em Python."
    ]
    right_note = "\n".join(textwrap.fill(paragraph, width=120) for paragraph in right_paragraphs)
    fig_text(fig, 5.91, .74, right_note, size=5.7, color=INK, linespacing=1.15, va="top")
    fig_text(fig, .46, .19,
             "A ANATOMIA DO SOM  /  ANÁLISE E VISUALIZAÇÃO FEITAS POR MIM EM PYTHON  /  BASE NÃO REPRESENTATIVA DE TODA A MÚSICA",
             size=5.0, color=MUTED, weight="bold")

    # O PDF permanece vetorial; o PNG e exportado em 300 dpi para apresentacao/impressao.
    fig.savefig(pdf_path, format="pdf", facecolor=BG, dpi=300, metadata={
        "Title": "A Anatomia do Som - Mapa Sonoro do Spotify",
        "Subject": "Infografico editorial de analise descritiva do Spotify Tracks Dataset",
        "Author": "Análise e visualização próprias em Python",
    })
    fig.savefig(png_path, format="png", facecolor=BG, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gera o infografico A Anatomia do Som a partir do CSV.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Caminho do spotify-tracks-dataset-detailed.csv (padrao: mesma pasta do script).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Pasta de saida (padrao: mesma pasta do script).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = (args.csv if args.csv is not None else script_dir / "spotify-tracks-dataset-detailed.csv").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else script_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_analyze(csv_path)
    pdf_path = output_dir / "a_anatomia_do_som.pdf"
    png_path = output_dir / "a_anatomia_do_som.png"
    create_infographic(data, pdf_path, png_path)
    write_summary(data, output_dir)

    print("Infografico gerado com sucesso:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"  Estatisticas: {output_dir / 'a_anatomia_estatisticas.csv'}")
    print(f"  Perfil por genero: {output_dir / 'a_anatomia_perfil_generos.csv'}")
    print(f"  Registros originais: {data['n_original']:,}".replace(",", "."))
    print(f"  Duplicatas exatas removidas: {data['exact_duplicates']:,}".replace(",", "."))
    print(f"  IDs unicos: {data['n_tracks']:,}".replace(",", "."))
    print(f"  Generos: {data['n_genres']}")
    print(f"  Spearman popularidade x dancabilidade: {data['correlations']['Dancabilidade']:+.4f}")
    print(f"  Pearson energia x acustica (medias de genero): {data['genre_feature_corr']:+.4f}")


if __name__ == "__main__":
    main()
