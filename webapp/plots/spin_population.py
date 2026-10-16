import base64
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor


def _prepare_mol_and_drawer(smiles, spin_population, uncertainty, rgb, theme="light"):
    """Shared preprocessing: returns (mol, drawer, hover_texts, atom_xs, atom_ys)"""
    mol = Chem.MolFromSmiles(smiles)
    nh_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N' and a.GetTotalNumHs() > 0]
    if nh_indices:
        mol = Chem.AddHs(mol, onlyOnAtoms=nh_indices)
    rdDepictor.Compute2DCoords(mol)
    num_heavy_atoms = len(spin_population)

    if uncertainty is not None:
        atom_colors = {i: uncertainty_to_alpha(uncertainty[i], rgb=rgb) for i in range(num_heavy_atoms)}
    else:
        atom_colors = {
            i: spin_to_color(spin_population[i], vmin=np.min(spin_population), vmax=np.max(spin_population), rgb=rgb)
            for i in range(num_heavy_atoms)
        }

    atom_radii = {i: float(1.5 * spin_population[i]) for i in range(num_heavy_atoms)}

    drawer, _ = load_acs_style(theme)
    draw_opts = {
        "highlightAtoms": list(range(num_heavy_atoms)),
        "highlightAtomColors": atom_colors,
        "highlightAtomRadii": atom_radii,
    }

    if uncertainty is not None:
        total_unc = np.sum(uncertainty)
        confidence = 'High' if total_unc < 1.0 else 'Medium' if total_unc < 1.5 else 'Low'
        draw_opts["legend"] = f"{confidence} confidence prediction\nSum uncertainty: {np.round(total_unc, 4)}"

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, **draw_opts)
    drawer.FinishDrawing()

    # Atom screen coordinates + hover text
    atom_xs, atom_ys, hover_texts = [], [], []
    for i in range(num_heavy_atoms):
        pt = drawer.GetDrawCoords(i)
        atom_xs.append(pt.x)
        atom_ys.append(500 - pt.y)
        text = f"<b>Atom {i} ({mol.GetAtomWithIdx(i).GetSymbol()})</b><br>Spin density: {spin_population[i]:.4f}"
        if uncertainty is not None:
            text += f"<br>Uncertainty: {uncertainty[i]:.4f}"
        hover_texts.append(text)

    return mol, drawer, atom_xs, atom_ys, hover_texts


def _make_colorbar_info(spin_population, uncertainty, rgb):
    """Returns (r, g, b, cmin, cmax, label, alpha_start, alpha_end) of the colour bar."""
    r, g, b = np.array(rgb) / 255.0
    cmin, cmax = float(np.min(spin_population)), float(np.max(spin_population))
    label = 'Uncertainty' if uncertainty is not None else 'Spin density'
    alpha_start, alpha_end = (0.8, 0.0) if uncertainty is not None else (0.0, 0.8)
    return r, g, b, cmin, cmax, label, alpha_start, alpha_end


def visualize_density_interactive(smiles, spin_population, uncertainty=None, rgb=[0, 64, 128], theme="light"):
    mol, drawer, atom_xs, atom_ys, hover_texts = _prepare_mol_and_drawer(smiles, spin_population, uncertainty, rgb, theme)
    r, g, b, cmin, cmax, label, alpha_start, alpha_end = _make_colorbar_info(spin_population, uncertainty, rgb)

    img_b64 = base64.b64encode(drawer.GetDrawingText()).decode()

    fig = go.Figure()

    fig.add_layout_image(
        source=f"data:image/png;base64,{img_b64}",
        x=0, y=500, xref="x", yref="y",
        sizex=500, sizey=500, sizing="stretch", layer="below"
    )

    fig.add_trace(go.Scatter(
        x=atom_xs, y=atom_ys, mode="markers",
        marker=dict(size=20, opacity=0),
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts,
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=[
                [0.0, f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha_start})"],
                [1.0, f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha_end})"],
            ],
            cmin=cmin, cmax=cmax, color=[cmin],
            showscale=True,
            colorbar=dict(
                title=dict(text=label, font=dict(color="#f8fafc" if theme=="dark" else "#0f172a")),
                thickness=15,
                len=0.75,
                tickfont=dict(color="#94a3b8" if theme=="dark" else "#64748b")
            ),
        ),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 500], visible=False),
        yaxis=dict(range=[0, 500], visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        width=480, height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#1e293b" if theme=="dark" else "#ffffff", 
            font_size=13, 
            font_color="#f8fafc" if theme=="dark" else "#0f172a", 
            bordercolor="#334155" if theme=="dark" else "#e2e8f0"
        ),
    )

    return fig

def spin_to_color(value, vmin, vmax, rgb=[0, 64, 128]):
    norm = max(0, min(0.8, (value - vmin) / (vmax - vmin)))
    return tuple(np.append(np.array(rgb) / 255.0, norm).tolist())


def uncertainty_to_alpha(value, alpha_max=0.8, uncertainty_factor=2, rgb=[0, 64, 128]):
    return tuple(np.append(np.array(rgb) / 255.0, alpha_max - uncertainty_factor * value).tolist())


def load_acs_style(theme="light"):
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    if theme == "dark":
        opts.updateAtomPalette({k: (0.95, 0.95, 0.95) for k in range(0, 120)})
    else:
        opts.updateAtomPalette({k: (0.05, 0.05, 0.05) for k in range(0, 120)})
    opts.bondLineWidth = 3.0
    opts.minFontSize = 32
    opts.maxFontSize = 42
    opts.padding = 0.05
    opts.addStereoAnnotation = False
    opts.explicitMethyl = False
    opts.addAtomIndices = False
    return drawer, opts