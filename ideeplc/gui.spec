import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import TOC
from pathlib import Path


os.environ["MODIN_ENGINE"] = "python"

app_name = "iDeepLC"
script_path = "gui.py"
icon_path = None

packages = ["PIL", "requests", "torch", "ideeplc", "lxml", "pyteomics", "tqdm"]
hiddenimports = set()
datas, binaries = [], []

for pkg in packages:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports.update(h)
    except ImportError:
        continue

# Explicitly include the pretrained model


a = Analysis(
    [script_path],
    pathex=[os.getcwd()],
    binaries=binaries,
    datas=[
        ("models/pretrained_model.pth", "ideeplc/models"),
        ("models/*.pth", "ideeplc/models"),
        ("structure_feature/aa_stan.csv", "ideeplc/structure_feature"),
        ("structure_feature/ptm_stan.csv", "ideeplc/structure_feature"),
    ],
    hiddenimports=["scipy.special.cython_special"],
    hooksconfig={},
    noarchive=False,
    cipher=None,
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    excludes=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # GUI mode
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name=app_name,
)
