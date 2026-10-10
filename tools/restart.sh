#!/bin/bash
# redémarre le serveur de développement (port 8602) avec le code à jour
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8602 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }"
export PYTHONIOENCODING=utf-8 LOKY_MAX_CPU_COUNT=4
cd "$(dirname "$0")/.."
(python -m streamlit run app.py --server.headless true --server.port 8602 --browser.gatherUsageStats false > cache/shots/server_new.log 2>&1 &)
python - <<'PY'
import time, urllib.request
for i in range(40):
    try: urllib.request.urlopen("http://localhost:8602/_stcore/health",timeout=2).read(); print("serveur prêt"); break
    except Exception: time.sleep(1)
PY
