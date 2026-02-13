import joblib
import os
model = joblib.load("tabpfn_model_male.pkl")
TABPFN_DIR = r"D:\tabpfn_models"
os.makedirs(TABPFN_DIR, exist_ok=True)
try:
    steps = getattr(model, "named_steps", {})
    for name, step in steps.items():
        if hasattr(step, "set_params") and ("model_path" in step.get_params(deep=False)):
            model.set_params(**{f"{name}__model_path": TABPFN_DIR})
except Exception as e:
    print("set_params failed:", e)
    