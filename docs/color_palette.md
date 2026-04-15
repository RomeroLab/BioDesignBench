# BioDesignBench Color Palette

Standard color palette for all project figures and visualizations.
Based on the Okabe-Ito colorblind-friendly palette with purposeful cross-category overlap.

## Quick Reference (Python dict)

```python
COLORS = {
    # Models
    "gpt":       "#56B4E9",  # Sky Blue
    "claude":    "#E69F00",  # Orange
    "gemini":    "#009E73",  # Bluish Green
    "deepseek":  "#CC79A7",  # Reddish Purple
    # Baselines
    "oracle":    "#D55E00",  # Vermilion
    "hardcoded": "#0072B2",  # Blue
    "expert":    "#666666",  # Gray
    # Rubric
    "quality":     "#009E73",  # Bluish Green (= Gemini)
    "feasibility": "#E69F00",  # Orange (= Claude)
    # Heatmap
    "high_count": "#D55E00",  # Vermilion (= Oracle)
    "mid_count":  "#E69F0066",  # Orange Tint
}
```

## Full Palette

### A. Models (Category C)

| Element  | HEX       | Icon / Pattern                       |
|----------|-----------|--------------------------------------|
| GPT      | `#56B4E9` | Bar (User: Solid / Benchmark: Hashed)|
| Claude   | `#E69F00` | Bar (User: Solid / Benchmark: Hashed)|
| Gemini   | `#009E73` | Bar (User: Solid / Benchmark: Hashed)|
| DeepSeek | `#CC79A7` | Bar (User: Solid / Benchmark: Hashed)|

### B. Baselines (Category C)

| Element       | HEX       | Icon / Pattern     |
|---------------|-----------|--------------------|
| Human Oracle  | `#D55E00` | Horizontal dotted  |
| Hard-coded    | `#0072B2` | Horizontal solid   |
| Human Expert  | `#666666` | Horizontal dashed  |

### C. Agent Loop (Category A)

| Step       | HEX       | Icon            | Overlap        |
|------------|-----------|-----------------|----------------|
| PLAN       | `#56B4E9` | Loop arrow      | GPT color      |
| CALL TOOLS | `#E69F00` | Loop arrow      | Claude color   |
| EVALUATE   | `#CC79A7` | Loop arrow      | DeepSeek color |
| ITERATE    | `#009E73` | Loop arrow      | Gemini color   |

### D. Tool Categories (Category A)

| Tool Type  | Color             | Icon              | Overlap              |
|------------|-------------------|-------------------|----------------------|
| Structure  | Sky Blue Tint     | Toolbox / border  | GPT / Plan           |
| Sequence   | Bluish Green Tint | Toolbox / border  | Gemini / Iterate     |
| Backbone   | Orange Tint       | Toolbox / border  | Claude / Call Tools  |
| Scoring    | Reddish Purple Tint| Toolbox / border | DeepSeek / Evaluate  |

### E. Rubric Dimensions (Category A)

| Dimension    | HEX       | Icon            | Overlap                   |
|--------------|-----------|-----------------|---------------------------|
| Quality      | `#009E73` | Legend square    | Gemini / Iterate / Success|
| Feasibility  | `#E69F00` | Legend square    | Claude / Call / Constraint|

### F. Taxonomy Heatmap (Category B)

| Level      | Color              | Pattern       | Overlap            |
|------------|--------------------|---------------|--------------------|
| High Count | `#D55E00`          | Hatched fill  | Oracle / Target    |
| Mid Count  | Orange Tint        | Hatched fill  |                    |

## Matplotlib Usage

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Model colors (ordered: GPT, Claude, Gemini, DeepSeek)
MODEL_COLORS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
MODEL_NAMES  = ["GPT", "Claude", "Gemini", "DeepSeek"]

# Baseline line styles
BASELINE_STYLES = {
    "oracle":    {"color": "#D55E00", "linestyle": ":", "linewidth": 2},
    "hardcoded": {"color": "#0072B2", "linestyle": "-", "linewidth": 2},
    "expert":    {"color": "#666666", "linestyle": "--", "linewidth": 2},
}

# Example bar chart
fig, ax = plt.subplots()
for i, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
    ax.bar(i, [85, 72, 78, 65][i], color=color, label=name)
for name, style in BASELINE_STYLES.items():
    ax.axhline(y=70, **style, label=name)
ax.legend()
```

## Design Rationale

- **Colorblind-friendly**: Based on Okabe-Ito palette, safe for deuteranopia/protanopia.
- **Cross-category overlap**: Model colors reused in Loop/Tool/Rubric categories to create
  visual consistency (e.g., GPT Sky Blue = PLAN step = Structure tools).
- **User vs Benchmark mode**: Differentiated by bar fill pattern (solid vs hashed),
  not color, preserving colorblind accessibility.
