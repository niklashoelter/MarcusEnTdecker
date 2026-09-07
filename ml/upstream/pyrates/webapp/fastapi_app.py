from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
from pathlib import Path
import io
import json
import base64
from fastapi.responses import Response
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use('Agg')

# Note: assuming pyrates is in PYTHONPATH
from pyrates.webapp.science_utils.prediction import predict
from pyrates.webapp.science_utils.smiles_utils import is_valid_smiles, canonicalize_smiles, is_single_fragment_smiles
from pyrates.utilities.plotting.visualize_spin_population import visualize_density_interactive, visualize_density
from pyrates.utilities.plotting.plot_modes import plot_mode_distribution_single
from pyrates.webapp.science_utils.photocatalysts import list_photocatalyst_names, get_photocatalyst_row
from pyrates.webapp.science_utils import marcus_theory
import matplotlib.pyplot as plt
import numpy as np

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This safely pre-loads the model before accepting web traffic
    print("Pre-warming PyTorch model into global cache...")
    try:
        from pyrates.webapp.science_utils.prediction import predict as warmup_predict
        # Trigger cache loading via a single quick methane molecule
        warmup_predict(["C"])
        print("Model cached successfully!")
    except Exception as e:
        print(f"Could not pre-warm model: {e}")
    yield
    # Code here runs on app shutdown if necessary

app = FastAPI(title="PyRates REST API", lifespan=lifespan)

# Mount static files (like ketcher)
import os
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
    uncertainty: List[float] = None
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
    try:
        # Check for pre-drawn structure image if name is provided
        if request.name:
            import importlib.resources as ir
            
            # Since names contain underscores (e.g. Ir_ppy3), preserve alphanumeric, space, -, and _
            safe_name = "".join(c for c in request.name if c.isalnum() or c in (" ", "-", "_")).strip()
            if safe_name:
                # Look in 'pc_structures' inside the packaged data directory
                try:
                    img_path = ir.files("pyrates.webapp.data").joinpath("pc_structures", f"{safe_name}.png")
                    if img_path.is_file():
                        return Response(content=img_path.read_bytes(), media_type="image/png")
                except Exception:
                    pass

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
    names = list_photocatalyst_names()
    pcs = []
    for n in names:
        row = get_photocatalyst_row(n)
        s = row.get("photocatalyst_smiles", "")
        
        # handle missing values for E_red / E_ox safely
        e_red = row.get("E_red")
        e_ox = row.get("E_ox")
        e_red_val = float(e_red) if not pd.isna(e_red) else None
        e_ox_val = float(e_ox) if not pd.isna(e_ox) else None

        pcs.append({
            "name": n,
            "E_T_adiabatic": float(row["E_T_adiabatic"]),
            "lamda_S_T": float(row["lamda_S_T"]),
            "lamda_T_S": float(row["lamda_T_S"]),
            "E_red": e_red_val,
            "E_ox": e_ox_val,
            "smiles": str(s) if str(s).lower() != "nan" and s else ""
        })
    return {"photocatalysts": pcs}

class MarcusRequest(BaseModel):
    pc_adiabatic_E_T: float
    pc_lambda_S_T: float
    pc_lambda_T_S: float
    quencher_E_T: float
    quencher_lambda_S_T: float
    quencher_lambda_T_S: float
    unit: str = "kcal/mol"
    theme: str = "light"

@app.post("/api/marcus/compute")
def api_marcus_compute(request: MarcusRequest):
    dG, dG_act_fwd, dG_act_bwd = marcus_theory.calculate_driving_force_and_barrier(
        pc_adiabatic_E_T=request.pc_adiabatic_E_T,
        pc_lambda_S_T=request.pc_lambda_S_T,
        pc_lambda_T_S=request.pc_lambda_T_S,
        quencher_E_T=request.quencher_E_T,
        quencher_lambda_S_T=request.quencher_lambda_S_T,
        quencher_lambda_T_S=request.quencher_lambda_T_S,
        symmetric_approx=False,
    )
    return {
        "dG": float(dG),
        "dG_activation": float(dG_act_fwd),
        "dG_activation_backEnT": float(dG_act_bwd),
    }

@app.post("/api/marcus/plot")
def api_marcus_plot(request: MarcusRequest):
    try:
        res = marcus_theory.plot_reaction_coordinate(
            pc_adiabatic_E_T=request.pc_adiabatic_E_T,
            pc_lambda_S_T=request.pc_lambda_S_T,
            pc_lambda_T_S=request.pc_lambda_T_S,
            quencher_E_T=request.quencher_E_T,
            quencher_lambda_S_T=request.quencher_lambda_S_T,
            quencher_lambda_T_S=request.quencher_lambda_T_S,
            unit=request.unit,
            render=True,
            show=False,
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
    uvicorn.run("pyrates.webapp.fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
