"""
Canonical ROSIE biomarker order (channel index -> name).
Matches rosie/visualization.ipynb and rosie/README.md.
"""

ROSIE_BIOMARKERS = [
    "DAPI", "CD45", "CD68", "CD14", "PD1", "FoxP3", "CD8", "HLA-DR", "PanCK", "CD3e",
    "CD4", "aSMA", "CD31", "Vimentin", "CD45RO", "Ki67", "CD20", "CD11c", "Podoplanin", "PDL1",
    "GranzymeB", "CD38", "CD141", "CD21", "CD163", "BCL2", "LAG3", "EpCAM", "CD44", "ICOS",
    "GATA3", "Gal3", "CD39", "CD34", "TIGIT", "ECad", "CD40", "VISTA", "HLA-A", "MPO",
    "PCNA", "ATM", "TP63", "IFNg", "Keratin8/18", "IDO1", "CD79a", "HLA-E", "CollagenIV", "CD66",
]

ROSIE_NUM_CHANNELS = len(ROSIE_BIOMARKERS)


def marker_names_to_indices(
    names: list[str] | None,
    indices: list[int] | None = None,
) -> list[int]:
    """
    Resolve marker subset to indices.
    - If `indices` is provided (rosie_marker_subset_indices), use it directly.
    - If `names` is provided (rosie_marker_subset), map names -> indices.
    - If both None, return list(range(50)) (full set).
    """
    if indices is not None:
        return list(indices)
    if names is None:
        return list(range(ROSIE_NUM_CHANNELS))
    name_to_idx = {n: i for i, n in enumerate(ROSIE_BIOMARKERS)}
    out = []
    for n in names:
        if n in name_to_idx:
            out.append(name_to_idx[n])
        else:
            raise ValueError(f"Unknown ROSIE marker name: {n}. Valid: {ROSIE_BIOMARKERS[:10]}...")
    return out
