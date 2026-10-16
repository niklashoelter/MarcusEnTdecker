from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pathlib import Path
import io
from fastapi.responses import Response
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use('Agg')

from ml.prediction import predict
from webapp.smiles_utils import is_valid_smiles, canonicalize_smiles, is_single_fragment_smiles
from webapp.plots.spin_population import visualize_density_interactive
from webapp.plots.plot_modes import plot_mode_distribution_single
from webapp.photocatalysts import photocatalysts, structure_image
from ml.energy import marcus_ent, two_gate
from webapp.plots import marcus_plot
import matplotlib.pyplot as plt

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This safely pre-loads the model before accepting web traffic
    print("Pre-warming PyTorch model into global cache...")
    try:
        from ml.prediction import predict as warmup_predict
        # Trigger cache loading via a single quick methane molecule
        warmup_predict(["C"])
        print("Model cached successfully!")
    except Exception as e:
        print(f"Could not pre-warm model: {e}")
    yield
    # Code here runs on app shutdown if necessary

app = FastAPI(title="EnTdecker", lifespan=lifespan)

# Mount static files (like ketcher)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class PredictRequest(BaseModel):
    smiles: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
def api_predict(request: PredictRequest):
    limit=25
    if len(request.smiles) > limit:
        raise HTTPException(status_code=400, detail=f"A maximum of {limit} SMILES per request is allowed.")

    valid = []
    invalid = []
    for s in request.smiles:
        is_val = False
        try:
            is_val = is_valid_smiles(s)
        except Exception:
            pass
        if not is_val:
            invalid.append(s)
            continue
        if not is_single_fragment_smiles(s):
            invalid.append(s)
            continue
        canon = canonicalize_smiles(s)
        valid.append(canon if canon else s.strip())

    if not valid:
        raise HTTPException(status_code=400, detail="No valid SMILES provided")

    # Run predictions
    try:
        df = predict(valid)
        results = df.to_dict(orient="records")
        return {"results": results, "invalid": invalid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PlotModeRequest(BaseModel):
    mu: List[float]
    sigma: List[float]
    pi: List[float]
    predicted_value: float
    uncertainty: float
    target_name: str
    unit: str
    theme: str = "light"

@app.post("/api/plot/mode")
def api_plot_mode(request: PlotModeRequest):
    try:
        fig = plot_mode_distribution_single(
            mu=request.mu,
            sigma=request.sigma,
            pi=request.pi,
            predicted_value=request.predicted_value,
            uncertainty=request.uncertainty,
            target_name=request.target_name,
            unit=request.unit,
            theme=request.theme
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PlotSpinRequest(BaseModel):
    smiles: str
    spin_population: List[float]
    uncertainty: Optional[List[float]] = None
    show_uncertainty: bool = False
    theme: str = "light"

@app.post("/api/plot/spin")
def api_plot_spin(request: PlotSpinRequest):
    try:
        fig = visualize_density_interactive(
            request.smiles, 
            request.spin_population, 
            request.uncertainty if request.show_uncertainty else None,
            theme=request.theme
        )
        return Response(content=fig.to_json(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StructureRequest(BaseModel):
    smiles: str
    name: str = ""
    theme: str = "light"

@app.post("/api/structure")
def api_structure(request: StructureRequest):
    if not request.smiles.strip() and (not request.name or structure_image(request.name) is None):
        raise HTTPException(status_code=404, detail="No structure available")
    try:
        # A photocatalyst is drawn from its packaged image where it has one; `name` is its
        # id, looked up in the table and never used as a path.
        if request.name:
            image = structure_image(request.name)
            if image is not None:
                return Response(content=image, media_type="image/png")
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        # Consistent with load_acs_style but uses SVG for web view
        drawer = rdMolDraw2D.MolDraw2DSVG(200, 150)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        if request.theme == "dark":
            opts.updateAtomPalette({k: (0.95, 0.95, 0.95) for k in range(0, 120)})
        else:
            opts.updateAtomPalette({k: (0.05, 0.05, 0.05) for k in range(0, 120)})
        opts.bondLineWidth = 2.0
        opts.minFontSize = 20
        opts.maxFontSize = 30
        opts.padding = 0.05

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Clean up the xml namespace sometimes included multiple times
        svg = svg.replace('xmlns:svg="http://www.w3.org/2000/svg"', '')
        return Response(content=svg, media_type="image/svg+xml")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photocatalysts")
def api_photocatalysts():
    """The photocatalyst library: DFT E_T, lambda_S0_T1 and lambda_T1_S0 in kcal/mol, the
    DFT excited-state potentials in V vs SCE and the measured tau0 in ns (None where absent)."""
    return {"photocatalysts": photocatalysts()}

class MarcusRequest(BaseModel):
    """A pair, in `unit`. lambda_S0_T1 is the S0 surface at the T1 geometry, lambda_T1_S0
    the T1 surface at the S0 geometry; ml.energy.marcus_ent pairs them into
    lambda_forward = lambda_T1_S0(PC) + lambda_S0_T1(Q) and
    lambda_reverse = lambda_S0_T1(PC) + lambda_T1_S0(Q)."""
    pc_adiabatic_E_T: float
    pc_lambda_S0_T1: float
    pc_lambda_T1_S0: float
    quencher_E_T: float
    quencher_lambda_S0_T1: float
    quencher_lambda_T1_S0: float
    unit: str = "kcal/mol"
    theme: str = "light"
    pc_tau0_ns: Optional[float] = None

# kcal/mol per unit of the energies a request carries
KCAL_PER_UNIT = {"kcal/mol": 1.0, "kJ/mol": 1 / 4.184, "eV": 23.060548868,
                 "Hartree": 627.5094740631}

@app.post("/api/marcus/compute")
def api_marcus_compute(request: MarcusRequest):
    dG, dG_act_fwd, dG_act_bwd = marcus_ent.calculate_driving_force_and_barrier(
        pc_adiabatic_E_T=request.pc_adiabatic_E_T,
        pc_lambda_S_T=request.pc_lambda_S0_T1,
        pc_lambda_T_S=request.pc_lambda_T1_S0,
        quencher_E_T=request.quencher_E_T,
        quencher_lambda_S_T=request.quencher_lambda_S0_T1,
        quencher_lambda_T_S=request.quencher_lambda_T1_S0,
        symmetric_approx=False,
    )
    # The two-gate verdict needs the photocatalyst lifetime; without it, the barrier only.
    t_pc = verdict = None
    if request.pc_tau0_ns is not None:
        if request.unit not in KCAL_PER_UNIT:
            raise HTTPException(status_code=400, detail=f"Unknown unit {request.unit}")
        try:
            t_pc_kcal = two_gate.lifetime_barrier(request.pc_tau0_ns)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        t_pc = t_pc_kcal / KCAL_PER_UNIT[request.unit]
        verdict = two_gate.quenches(dG, dG_act_fwd, t_pc)
    return {
        "dG": float(dG),
        "dG_activation": float(dG_act_fwd),
        "dG_activation_backEnT": float(dG_act_bwd),
        "t_PC": None if t_pc is None or t_pc == float("-inf") else float(t_pc),
        "two_gate": verdict,
    }

@app.post("/api/marcus/plot")
def api_marcus_plot(request: MarcusRequest):
    try:
        res = marcus_plot.plot_reaction_coordinate(
            pc_adiabatic_E_T=request.pc_adiabatic_E_T,
            pc_lambda_S0_T1=request.pc_lambda_S0_T1,
            pc_lambda_T1_S0=request.pc_lambda_T1_S0,
            quencher_E_T=request.quencher_E_T,
            quencher_lambda_S0_T1=request.quencher_lambda_S0_T1,
            quencher_lambda_T1_S0=request.quencher_lambda_T1_S0,
            unit=request.unit,
            theme=request.theme,
        )
        fig = res.get("fig", None)
        if fig is None:
            raise ValueError("Could not generate figure")
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        
        # We also want to return symmetric_approx info. Using headers to pass metadata.
        headers = {
            "X-Symmetric-Approx": "true" if res.get("symmetric_approx") else "false",
            "X-Symmetric-Reason": str(res.get("symmetric_reason", ""))
        }
        return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text("utf8")

if __name__ == "__main__":
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=5000, reload=True)
