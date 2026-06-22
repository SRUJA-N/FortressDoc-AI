$ErrorActionPreference = "Stop"

# Run this script from the FortressDoc AI project folder:
#   .\run_app.ps1
#
# It uses the Python launcher directly, so it works even when pip.exe and
# streamlit.exe are installed outside PATH.

Set-Location -LiteralPath $PSScriptRoot

py -m pip install -r requirements.txt
py -m streamlit run app.py
