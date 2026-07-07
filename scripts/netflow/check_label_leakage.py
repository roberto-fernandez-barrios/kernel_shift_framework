import json

import numpy as np

for s in ["unsw_dos", "unsw_recon", "toniot_scanning"]:
    X = np.load(f"data/processed/netflow/{s}/X.npy", mmap_mode="r")
    y = np.load(f"data/processed/netflow/{s}/y.npy").astype(np.float64)
    names = json.load(open(f"data/processed/netflow/{s}/meta_export.json"))["feature_names"]
    Xf = np.asarray(X, dtype=np.float64)
    yc = y - y.mean()
    Xc = Xf - Xf.mean(axis=0)
    denom = Xc.std(axis=0) * y.std()
    corr = np.zeros(Xf.shape[1])
    nz = denom > 0
    corr[nz] = (Xc[:, nz] * yc[:, None]).mean(axis=0) / denom[nz]
    order = np.argsort(-np.abs(corr))[:5]
    print(f"{s}: top |corr(feature, y)|:")
    for i in order:
        print(f"   {names[i]:28s} {corr[i]:+.3f}")
