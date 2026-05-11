from toksearch import MdsSignal, Pipeline
import xarray as xr


pipe = Pipeline([165920])

kappa_signal = MdsSignal(r"\kappa", "efit01")

dims = ("r", "z", "times")
psirz_signal = MdsSignal(r'\psirz', 'efit01', dims=dims, data_order=['times', 'z', 'r'])

pipe.fetch_dataset("ds", {"kappa": kappa_signal, "psirz": psirz_signal})

results = pipe.compute_serial()

print(results[0])