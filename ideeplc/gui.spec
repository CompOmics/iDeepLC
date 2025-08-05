import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# Ensure ideeplc is discoverable
project_root = Path(os.path.abspath(sys.argv[0])).parent.parent
sys.path.insert(0, str(project_root))

from ideeplc import __version__ as version

os.environ["MODIN_ENGINE"] = "python"

app_name = f"iDeepLC_{version}"
script_path = "gui.py"
icon_path = str(project_root / "ideeplc" / "logo" / "ideeplc.ico")

packages = ["PIL", "requests", "torch", "ideeplc", "lxml", "pyteomics", "tqdm"]
hiddenimports = set()
datas, binaries = [], []

# Collect data and binaries from all necessary packages
for pkg in packages:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports.update(h)
    except ImportError:
        continue

# Explicitly include your required data files
extra_datas = [
    (str(project_root / "ideeplc" / "models" / "pretrained_model.pth"), "ideeplc/models"),
    (str(project_root / "ideeplc" / "structure_feature" / "aa_stan.csv"), "ideeplc/structure_feature"),
    (str(project_root / "ideeplc" / "structure_feature" / "ptm_stan.csv"), "ideeplc/structure_feature"),
]

datas.extend(extra_datas)

# Analysis step
a = Analysis(
    [script_path],
    pathex=[os.getcwd()],
    binaries=binaries,
    datas=datas,
    hiddenimports=list(hiddenimports) + ["scipy.special.cython_special"],
    hooksconfig={},
    noarchive=False,
    cipher=None,
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    excludes=[],
)

pyz = PYZ(a.pure, a.zipped_data)

# Executable
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
    console=True,  # set to False if you want to hide the console
    icon=icon_path,
)

# Bundle everything
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    name=app_name,
)
