import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import TOC
from pathlib import Path
from ideeplc import __version__ as version


os.environ["MODIN_ENGINE"] = "python"

app_name = "iDeepLC"
app_name = f"{app_name}_{version}"
script_path = "gui.py"
icon_path = "ideeplc/logo/ideeplc.ico"

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
        ("ideeplc/models/pretrained_model.pth", "ideeplc/models"),
        ("ideeplc/models/*.pth", "ideeplc/models"),
        ("ideeplc/structure_feature/aa_stan.csv", "ideeplc/structure_feature"),
        ("ideeplc/structure_feature/ptm_stan.csv", "ideeplc/structure_feature"),
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
    console=True,  # GUI mode
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    name=app_name,
)
