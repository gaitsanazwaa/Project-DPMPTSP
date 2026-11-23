"""Generate a publication-ready three-sector average LQ map for Surabaya."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter


YEARS = tuple(range(2021, 2026))
CATEGORIES = (
    "Sektor Basis dan Tumbuh",
    "Sektor Basis tetapi Tidak Tumbuh",
    "Sektor Non-Basis tetapi Tumbuh",
)
CATEGORY_LABELS = {
    "Sektor Basis dan Tumbuh": "Basis dan tumbuh",
    "Sektor Basis tetapi Tidak Tumbuh": "Basis tetapi tidak tumbuh",
    "Sektor Non-Basis tetapi Tumbuh": "Non-basis tetapi tumbuh",
}


def _normalize_district_name(name: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_name.casefold())


def _read_inputs(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = project_dir / "output"
    summary_path = output_dir / "analisis_gabungan_lq_pertumbuhan_sektor.csv"
    lq_path = output_dir / "hasil_lq_kecamatan_kbli.csv"
    for path in (summary_path, lq_path):
        if not path.is_file():
            raise FileNotFoundError(f"File hasil analisis tidak ditemukan: {path}")

    summary = pd.read_csv(summary_path, dtype={"KBLI_2digit": "string"})
    lq = pd.read_csv(lq_path, dtype={"KBLI_2digit": "string"})
    required_summary = {
        "KBLI_2digit",
        "jumlah_kegiatan_2021",
        "jumlah_kegiatan_2025",
        "rata_rata_lq_2021_2025",
        "jumlah_tahun_lq_tersedia",
        "cagr_2021_2025_persen",
        "kategori_gabungan",
        "status_data",
        "Judul_KBLI_5_digit_terbanyak",
    }
    required_lq = {"kecamatan_usaha", "KBLI_2digit", "Tahun", "LQ"}
    missing_summary = required_summary.difference(summary.columns)
    missing_lq = required_lq.difference(lq.columns)
    if missing_summary or missing_lq:
        raise KeyError(
            f"Kolom ringkasan hilang: {sorted(missing_summary)}; "
            f"kolom LQ hilang: {sorted(missing_lq)}"
        )

    summary["KBLI_2digit"] = summary["KBLI_2digit"].astype("string").str.strip().str.zfill(2)
    lq["KBLI_2digit"] = lq["KBLI_2digit"].astype("string").str.strip().str.zfill(2)
    if summary["KBLI_2digit"].duplicated().any():
        raise ValueError("Ringkasan analisis gabungan memiliki KBLI 2 digit duplikat.")
    for column in (
        "jumlah_kegiatan_2021",
        "jumlah_kegiatan_2025",
        "rata_rata_lq_2021_2025",
        "jumlah_tahun_lq_tersedia",
        "cagr_2021_2025_persen",
    ):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    lq["Tahun"] = pd.to_numeric(lq["Tahun"], errors="raise").astype(int)
    lq["LQ"] = pd.to_numeric(lq["LQ"], errors="coerce")
    lq["_district_key"] = lq["kecamatan_usaha"].map(
        lambda value: _normalize_district_name(value) if pd.notna(value) else ""
    )
    lq = lq.loc[lq["Tahun"].isin(YEARS)].copy()
    return summary, lq


def select_lq_map_sectors(
    project_dir: str | Path,
    min_activities_2025: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all candidate rows, one selected sector per available category, and alternatives."""
    project_dir = Path(project_dir).resolve()
    summary, _ = _read_inputs(project_dir)
    candidates = summary.loc[summary["kategori_gabungan"].isin(CATEGORIES)].copy()
    candidates["memenuhi_status_lengkap"] = candidates["status_data"].eq("Lengkap")
    candidates["memenuhi_lima_tahun_lq"] = candidates["jumlah_tahun_lq_tersedia"].eq(5)
    candidates["memenuhi_minimum_kegiatan_2025"] = (
        candidates["jumlah_kegiatan_2025"].ge(min_activities_2025)
    )
    candidates["memiliki_cagr"] = np.isfinite(candidates["cagr_2021_2025_persen"])
    candidates["layak_dipilih"] = candidates[
        [
            "memenuhi_status_lengkap",
            "memenuhi_lima_tahun_lq",
            "memenuhi_minimum_kegiatan_2025",
            "memiliki_cagr",
        ]
    ].all(axis=1)
    candidates["alasan_kelayakan"] = np.select(
        [
            ~candidates["memenuhi_status_lengkap"],
            ~candidates["memenuhi_lima_tahun_lq"],
            ~candidates["memenuhi_minimum_kegiatan_2025"],
            ~candidates["memiliki_cagr"],
        ],
        [
            "Status data tidak lengkap",
            "LQ sektor tidak tersedia untuk lima tahun",
            f"Kegiatan 2025 di bawah batas {min_activities_2025}",
            "CAGR tidak tersedia",
        ],
        default="Memenuhi seluruh kriteria",
    )
    candidates["skor_representatif"] = np.nan
    candidates["terpilih"] = False
    candidates["alasan_pemilihan"] = ""

    selected_rows = []
    alternatives = []
    for category in CATEGORIES:
        category_rows = candidates.loc[candidates["kategori_gabungan"].eq(category)]
        eligible = category_rows.loc[category_rows["layak_dipilih"]].copy()
        if eligible.empty:
            alternative_pool = category_rows.sort_values(
                [
                    "memenuhi_status_lengkap",
                    "memenuhi_lima_tahun_lq",
                    "jumlah_kegiatan_2025",
                    "cagr_2021_2025_persen",
                ],
                ascending=[False, False, False, False],
                na_position="last",
            )
            if not alternative_pool.empty:
                alternative = alternative_pool.iloc[0].copy()
                alternative["kategori_diminta"] = category
                alternative["alasan_alternatif"] = (
                    "Tidak dipilih karena tidak memenuhi seluruh kriteria; "
                    + str(alternative["alasan_kelayakan"])
                )
                alternatives.append(alternative)
            continue

        eligible["skor_representatif"] = (
            eligible["jumlah_kegiatan_2025"].rank(method="average", pct=True)
            + eligible["cagr_2021_2025_persen"].rank(method="average", pct=True)
        ) / 2
        eligible = eligible.sort_values(
            ["skor_representatif", "jumlah_kegiatan_2025", "cagr_2021_2025_persen", "KBLI_2digit"],
            ascending=[False, False, False, True],
        )
        candidates.loc[eligible.index, "skor_representatif"] = eligible["skor_representatif"]
        selected = eligible.iloc[0]
        selected_mask = candidates["KBLI_2digit"].eq(selected["KBLI_2digit"])
        candidates.loc[selected_mask, "skor_representatif"] = selected["skor_representatif"]
        candidates.loc[selected_mask, "terpilih"] = True
        candidates.loc[selected_mask, "alasan_pemilihan"] = (
            "Skor tertinggi dari rata-rata persentil jumlah kegiatan 2025 dan CAGR "
            "(bobot sama) di antara kandidat yang memenuhi kriteria."
        )
        selected_rows.append(selected)

    selected_sectors = pd.DataFrame(selected_rows).reset_index(drop=True)
    alternative_table = pd.DataFrame(alternatives)
    candidates = candidates.sort_values(
        ["kategori_gabungan", "layak_dipilih", "skor_representatif", "jumlah_kegiatan_2025"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return candidates, selected_sectors, alternative_table


def _load_geojson(project_dir: Path) -> tuple[list[dict], dict[str, str]]:
    geojson_path = project_dir / "data_raw" / "geojson_kecamatan_surabaya_merged.geojson"
    if not geojson_path.is_file():
        raise FileNotFoundError(f"GeoJSON kecamatan tidak ditemukan: {geojson_path}")
    with geojson_path.open(encoding="utf-8") as geojson_file:
        geojson = json.load(geojson_file)
    features = geojson.get("features", [])
    if geojson.get("type") != "FeatureCollection" or len(features) != 31:
        raise ValueError(f"GeoJSON harus memuat tepat 31 kecamatan: {geojson_path}")

    geo_names_by_key = {}
    for feature in features:
        properties = feature.get("properties") or {}
        if "K" not in properties or not properties["K"]:
            raise ValueError("Terdapat feature GeoJSON tanpa nama kecamatan pada property 'K'.")
        key = _normalize_district_name(properties["K"])
        if key in geo_names_by_key:
            raise ValueError(f"Nama kecamatan GeoJSON duplikat setelah normalisasi: {properties['K']}")
        geo_names_by_key[key] = str(properties["K"])
    return features, geo_names_by_key


def _project_ring(ring: object, longitude_center: float, latitude_center: float) -> np.ndarray:
    points = np.asarray(ring, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] < 3:
        raise ValueError("Ring GeoJSON tidak memiliki setidaknya tiga titik longitude-latitude.")
    points = points[:, :2]
    if not np.isfinite(points).all():
        raise ValueError("Koordinat GeoJSON mengandung nilai non-finite.")
    if (np.abs(points[:, 0]) > 180).any() or (np.abs(points[:, 1]) > 90).any():
        raise ValueError("Koordinat GeoJSON berada di luar rentang longitude-latitude.")
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    longitude_scale = float(np.cos(np.deg2rad(latitude_center)))
    projected = np.column_stack(
        (
            (points[:, 0] - longitude_center) * longitude_scale,
            points[:, 1] - latitude_center,
        )
    )
    if len(np.unique(projected, axis=0)) < 4:
        raise ValueError("Ring GeoJSON tidak membentuk polygon dengan titik valid yang cukup.")
    return projected


def generate_lq_map_panels(
    project_dir: str | Path,
    selected_sectors: pd.DataFrame | None = None,
    min_activities_2025: int = 20,
) -> tuple[Path, Path, Path, dict]:
    """Create common-scale LQ maps and a district-level coverage table."""
    project_dir = Path(project_dir).resolve()
    output_dir = project_dir / "output"
    visual_dir = output_dir / "visualisasi"
    visual_dir.mkdir(parents=True, exist_ok=True)
    _, lq = _read_inputs(project_dir)
    if selected_sectors is None:
        _, selected_sectors, alternatives = select_lq_map_sectors(
            project_dir, min_activities_2025=min_activities_2025
        )
    else:
        alternatives = pd.DataFrame()
        selected_sectors = selected_sectors.copy()
        selected_sectors["KBLI_2digit"] = (
            selected_sectors["KBLI_2digit"].astype("string").str.strip().str.zfill(2)
        )
    if len(selected_sectors) != 3 or set(selected_sectors["kategori_gabungan"]) != set(CATEGORIES):
        raise ValueError(
            "Peta memerlukan tepat satu sektor terpilih per kategori; "
            f"kategori tersedia: {selected_sectors.get('kategori_gabungan', pd.Series(dtype=str)).tolist()}"
        )
    selected_sectors = selected_sectors.drop_duplicates("KBLI_2digit")
    if len(selected_sectors) != 3:
        raise ValueError("Sektor terpilih harus memiliki tiga kode KBLI yang berbeda.")

    features, geo_names_by_key = _load_geojson(project_dir)
    selected_codes = set(selected_sectors["KBLI_2digit"])
    selected_lq = lq.loc[
        lq["KBLI_2digit"].isin(selected_codes) & lq["Tahun"].isin(YEARS)
    ].copy()
    if selected_lq["_district_key"].eq("").any():
        raise ValueError("Terdapat nama kecamatan kosong pada hasil LQ terpilih.")
    unknown_districts = sorted(
        selected_lq.loc[
            ~selected_lq["_district_key"].isin(geo_names_by_key), "kecamatan_usaha"
        ].dropna().unique().tolist()
    )
    if unknown_districts:
        raise ValueError(f"Nama kecamatan hasil LQ tidak cocok dengan GeoJSON: {unknown_districts}")
    if selected_lq.duplicated(["KBLI_2digit", "_district_key", "Tahun"]).any():
        raise ValueError("Ada baris LQ duplikat pada kombinasi KBLI-kecamatan-tahun terpilih.")

    lq_by_district = selected_lq.groupby(
        ["KBLI_2digit", "_district_key"], as_index=False
    ).agg(
        rata_rata_lq=("LQ", "mean"),
        tahun_lq_tersedia=("LQ", "count"),
    )
    coverage_rows = []
    for sector in selected_sectors.itertuples(index=False):
        for district_key, district_name in geo_names_by_key.items():
            values = lq_by_district.loc[
                lq_by_district["KBLI_2digit"].eq(sector.KBLI_2digit)
                & lq_by_district["_district_key"].eq(district_key)
            ]
            if len(values) > 1:
                raise ValueError(
                    f"Nilai ringkasan LQ duplikat untuk KBLI {sector.KBLI_2digit}, "
                    f"kecamatan {district_name}."
                )
            if values.empty:
                mean_lq, year_count = np.nan, 0
            else:
                mean_lq = values["rata_rata_lq"].iloc[0]
                year_count = int(values["tahun_lq_tersedia"].iloc[0])
            coverage_rows.append(
                {
                    "KBLI_2digit": sector.KBLI_2digit,
                    "kecamatan_usaha": district_name,
                    "rata_rata_lq_2021_2025": mean_lq,
                    "jumlah_tahun_lq_tersedia": year_count,
                    "tahun_pengamatan": "2021–2025",
                }
            )
    district_coverage = pd.DataFrame(coverage_rows)
    if len(district_coverage) != 93:
        raise ValueError(f"Tabel cakupan diharapkan 93 baris, aktual {len(district_coverage)}.")

    valid_lq = district_coverage["rata_rata_lq_2021_2025"].dropna()
    if valid_lq.empty:
        raise ValueError("Tidak ada rata-rata LQ valid untuk divisualisasikan.")
    actual_min = float(valid_lq.min())
    actual_max = float(valid_lq.max())
    vmin = actual_min
    vmax = max(1.05, float(valid_lq.quantile(0.98)))
    if not vmin < 1.0 < vmax:
        raise ValueError("Skala warna tidak dapat menempatkan nilai batas LQ=1.")
    lq_norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    lq_cmap = LinearSegmentedColormap.from_list(
        "lq_nonbasis_basis", ["#2166ac", "#ffffbf", "#b2182b"], N=256
    )
    lq_cmap.set_over("#7f0000")
    missing_color = "#bdbdbd"

    prepared_features = []
    all_projected_rings = []
    raw_coordinates = []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"Fitur {properties['K']} tidak memiliki geometri valid.")
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            raise ValueError(f"Tipe geometri tidak didukung: {geometry_type}")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Koordinat kosong/tidak valid untuk kecamatan {properties['K']}.")
        for polygon in polygons:
            for ring in polygon:
                raw_coordinates.extend(np.asarray(ring, dtype=float)[:, :2].tolist())
    raw_coordinates = np.asarray(raw_coordinates, dtype=float)
    if raw_coordinates.ndim != 2 or raw_coordinates.shape[1] != 2:
        raise ValueError("GeoJSON tidak memiliki koordinat polygon yang valid.")
    longitude_center = float(raw_coordinates[:, 0].mean())
    latitude_center = float(raw_coordinates[:, 1].mean())

    for feature in features:
        properties = feature["properties"]
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]
        polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
        projected_polygons = []
        for polygon in polygons:
            projected_rings = [
                _project_ring(ring, longitude_center, latitude_center)
                for ring in polygon
            ]
            all_projected_rings.extend(projected_rings)
            projected_polygons.append(projected_rings)
        prepared_features.append(
            {
                "district_key": _normalize_district_name(properties["K"]),
                "projected_polygons": projected_polygons,
            }
        )

    projected_coordinates = np.vstack(all_projected_rings)
    xmin, ymin = projected_coordinates.min(axis=0)
    xmax, ymax = projected_coordinates.max(axis=0)
    x_padding = (xmax - xmin) * 0.035
    y_padding = (ymax - ymin) * 0.035
    map_lookup = {
        code: rows.set_index("_district_key").to_dict("index")
        for code, rows in lq_by_district.groupby("KBLI_2digit")
    }
    coverage_path = visual_dir / "cakupan_lq_kecamatan_tiga_sektor_2021_2025.csv"
    district_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    fig.subplots_adjust(left=0.015, right=0.89, bottom=0.30, top=0.82, wspace=0.025)
    validation = {}
    for ax, sector in zip(axes, selected_sectors.itertuples(index=False)):
        code = sector.KBLI_2digit
        values = map_lookup.get(code, {})
        for feature in prepared_features:
            district_data = values.get(feature["district_key"], {})
            value = district_data.get("rata_rata_lq", np.nan)
            fill_color = (
                lq_cmap(lq_norm(float(value)))
                if pd.notna(value) and np.isfinite(value)
                else missing_color
            )
            for polygon_rings in feature["projected_polygons"]:
                ax.add_patch(
                    Polygon(
                        polygon_rings[0],
                        closed=True,
                        facecolor=fill_color,
                        edgecolor="#ffffff",
                        linewidth=0.65,
                        joinstyle="round",
                        zorder=2,
                    )
                )
                for hole in polygon_rings[1:]:
                    ax.add_patch(
                        Polygon(
                            hole,
                            closed=True,
                            facecolor="white",
                            edgecolor="#ffffff",
                            linewidth=0.45,
                            zorder=3,
                        )
                    )
        ax.set_xlim(xmin - x_padding, xmax + x_padding)
        ax.set_ylim(ymin - y_padding, ymax + y_padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        title = f"KBLI {code} | {CATEGORY_LABELS[sector.kategori_gabungan]}"
        representative_label = str(sector.Judul_KBLI_5_digit_terbanyak)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        sector_coverage = district_coverage.loc[district_coverage["KBLI_2digit"].eq(code)]
        available_lq = sector_coverage["rata_rata_lq_2021_2025"].dropna()
        year_distribution = (
            sector_coverage["jumlah_tahun_lq_tersedia"]
            .value_counts()
            .reindex(range(6), fill_value=0)
            .sort_index()
        )
        available_year_counts = sector_coverage.loc[
            sector_coverage["rata_rata_lq_2021_2025"].notna(),
            "jumlah_tahun_lq_tersedia",
        ]
        complete_year_lq = sector_coverage.loc[
            sector_coverage["jumlah_tahun_lq_tersedia"].eq(len(YEARS)),
            "rata_rata_lq_2021_2025",
        ].dropna()
        validation[code] = {
            "district_count": len(sector_coverage),
            "available_lq_count": int(sector_coverage["rata_rata_lq_2021_2025"].notna().sum()),
            "valid_year_range": (
                int(available_year_counts.min()) if not available_year_counts.empty else 0,
                int(available_year_counts.max()) if not available_year_counts.empty else 0,
            ),
            "year_distribution": {
                int(year_count): int(district_count)
                for year_count, district_count in year_distribution.items()
            },
            "lq_min": float(available_lq.min()) if not available_lq.empty else np.nan,
            "lq_max": float(available_lq.max()) if not available_lq.empty else np.nan,
            "above_one": int(available_lq.gt(1.0).sum()),
            "equal_one": int(available_lq.eq(1.0).sum()),
            "below_one": int(available_lq.lt(1.0).sum()),
            "full_year_count": int(len(complete_year_lq)),
            "full_year_lq_min": (
                float(complete_year_lq.min()) if not complete_year_lq.empty else np.nan
            ),
            "full_year_lq_max": (
                float(complete_year_lq.max()) if not complete_year_lq.empty else np.nan
            ),
            "full_year_above_one": int(complete_year_lq.gt(1.0).sum()),
            "full_year_equal_one": int(complete_year_lq.eq(1.0).sum()),
            "full_year_below_one": int(complete_year_lq.lt(1.0).sum()),
        }
        coverage_text = ", ".join(
            f"{year_count} th: {district_count}"
            for year_count, district_count in validation[code]["year_distribution"].items()
            if district_count
        )
        ax.text(
            0.5,
            -0.035,
            f"Contoh kegiatan usaha: {representative_label}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#333333",
        )
        ax.text(
            0.5,
            -0.105,
            f"LQ tersedia: {validation[code]['available_lq_count']}/"
            f"{validation[code]['district_count']} kec. | Tahun valid (kecamatan): {coverage_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            color="#333333",
        )

    fig.suptitle(
        "Persebaran Rata-rata Location Quotient (LQ) Kegiatan Usaha pada Tiga Sektor "
        "KBLI di Kota Surabaya Tahun 2021–2025",
        fontsize=14,
        fontweight="bold",
        y=0.96,
        x=0.46,
    )
    colorbar_ax = fig.add_axes([0.915, 0.30, 0.018, 0.52])
    colorbar_ticks = sorted(set(np.linspace(vmin, vmax, 5).tolist() + [1.0]))
    colorbar = fig.colorbar(
        ScalarMappable(norm=lq_norm, cmap=lq_cmap),
        cax=colorbar_ax,
        ticks=colorbar_ticks,
        extend="max" if actual_max > vmax else "neither",
    )
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.set_label("Rata-rata LQ (2021–2025)", fontsize=9)
    colorbar.ax.axhline(1.0, color="#222222", linewidth=1.2)
    colorbar.ax.text(
        1.8,
        lq_norm(1.0),
        "LQ = 1",
        transform=colorbar.ax.get_yaxis_transform(),
        va="center",
        fontsize=8.5,
    )
    fig.text(
        0.46,
        0.095,
        "LQ < 1: konsentrasi relatif lebih rendah  |  "
        "LQ = 1: setara tingkat kota  |  "
        "LQ > 1: konsentrasi relatif lebih tinggi  |  "
        "Abu-abu: nilai tidak tersedia",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.46,
        0.045,
        f"Rata-rata dihitung dari tahun yang tersedia. Batas warna atas {vmax:.2f}; "
        f"{int(valid_lq.gt(vmax).sum())} nilai di atas batas ditampilkan dengan warna maksimum.",
        ha="center",
        fontsize=8.5,
    )

    png_path = visual_dir / "peta_lq_tiga_sektor_rata_rata_2021_2025_final.png"
    svg_path = visual_dir / "peta_lq_tiga_sektor_rata_rata_2021_2025_final.svg"
    pdf_path = visual_dir / "peta_lq_tiga_sektor_rata_rata_2021_2025_final.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    from PIL import Image

    with Image.open(png_path) as image:
        image.verify()
    with Image.open(png_path) as image:
        png_size = image.size
        png_dpi = image.info.get("dpi")
    for path, signature in ((svg_path, b"<?xml"), (pdf_path, b"%PDF-")):
        with path.open("rb") as file:
            if not file.read(len(signature)).startswith(signature):
                raise IOError(f"Format keluaran tidak valid: {path}")
    if png_size[0] < 2400 or png_size[1] < 900 or not png_dpi or min(png_dpi) < 300:
        raise IOError(f"Resolusi PNG tidak memenuhi syarat minimal 300 dpi: {png_size}, {png_dpi}")
    matched_district_count = len(
        set(selected_lq["_district_key"]).intersection(geo_names_by_key)
    )
    info = {
        "selected_sectors": selected_sectors,
        "alternatives": alternatives,
        "district_coverage": district_coverage,
        "matched_district_count": matched_district_count,
        "district_count": len(features),
        "png_size": png_size,
        "png_dpi": png_dpi,
        "scale": {"vmin": vmin, "center": 1.0, "vmax": vmax},
        "actual_lq_range": {"min": actual_min, "max": actual_max},
        "clipped_value_count": int(valid_lq.gt(vmax).sum()),
        "validation": validation,
        "shared_scale": True,
        "coverage_csv": coverage_path,
    }
    print("Ringkasan validasi peta LQ:")
    for code, result in validation.items():
        distribution = ", ".join(
            f"{years} tahun: {count} kecamatan"
            for years, count in result["year_distribution"].items()
            if count
        )
        print(
            f"KBLI {code}: kecamatan={result['district_count']}; "
            f"LQ tersedia={result['available_lq_count']}; "
            f"tahun valid={result['valid_year_range'][0]}-"
            f"{result['valid_year_range'][1]}/{len(YEARS)}; "
            f"distribusi=({distribution}); "
            f"LQ min-maks={result['lq_min']:.5f}-{result['lq_max']:.5f}; "
            f">1={result['above_one']}, =1={result['equal_one']}, "
            f"<1={result['below_one']}; "
            f"pembanding 5/5={result['full_year_count']} kecamatan "
            f"(LQ {result['full_year_lq_min']:.5f}-{result['full_year_lq_max']:.5f}; "
            f">1={result['full_year_above_one']}, "
            f"=1={result['full_year_equal_one']}, "
            f"<1={result['full_year_below_one']})"
        )
    print(
        "Skala warna bersama untuk ketiga panel: "
        f"{info['shared_scale']} (min={vmin:.5f}, tengah=1.00, "
        f"batas atas visual={vmax:.5f}; rentang data aktual="
        f"{actual_min:.5f}-{actual_max:.5f}; nilai di atas batas="
        f"{info['clipped_value_count']})."
    )
    return svg_path, png_path, pdf_path, info
