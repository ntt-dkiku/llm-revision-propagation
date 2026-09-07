#!/usr/bin/env bash
# CI guard: verify no package in uv.lock or package-lock.json was released
# within the cooldown window. Read-only HTTP queries against PyPI / npm
# registries; the script never installs or modifies anything.
#
# Exit codes:
#   0 = all packages older than cutoff
#   1 = at least one package newer than cutoff, or required config missing
#   2 = network failure (warn but do not fail unless --strict)

set -euo pipefail

MAX_AGE_DAYS="${SUPPLY_CHAIN_COOLDOWN_DAYS:-7}"
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --max-age-days=*) MAX_AGE_DAYS="${arg#*=}" ;;
    --strict) STRICT=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

CUTOFF_ISO="$(date -u -d "-${MAX_AGE_DAYS} days" +%Y-%m-%dT%H:%M:%SZ)"
echo "[cooldown-check] cutoff = ${CUTOFF_ISO} (${MAX_AGE_DAYS} days)"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- 1) pyproject.toml has exclude-newer ---
if ! grep -qE '^\s*exclude-newer\s*=' pyproject.toml; then
  echo "[cooldown-check] FAIL: [tool.uv].exclude-newer not set in pyproject.toml" >&2
  exit 1
fi

# --- 2) uv.lock present ---
if [[ ! -f uv.lock ]]; then
  echo "[cooldown-check] FAIL: uv.lock missing — commit it" >&2
  exit 1
fi

# --- 3) Verify every uv.lock package release date is <= cutoff ---
uv run --no-project --with packaging python <<PYEOF
import sys, json, urllib.request, tomllib
from datetime import datetime, timezone

cutoff = datetime.fromisoformat("${CUTOFF_ISO}".replace("Z", "+00:00"))
with open("uv.lock", "rb") as f:
    data = tomllib.loads(f.read().decode())

bad, neterr = [], 0
for pkg in data.get("package", []):
    name, version = pkg.get("name"), pkg.get("version")
    if not name or not version:
        continue
    src = pkg.get("source") or {}
    # Skip packages without a PyPI registry source (editable installs, virtual workspaces, git deps).
    if src.get("registry") != "https://pypi.org/simple":
        continue
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            meta = json.load(resp)
        urls = meta.get("urls") or []
        if not urls:
            continue
        upload = urls[0].get("upload_time_iso_8601")
        if not upload:
            continue
        when = datetime.fromisoformat(upload.replace("Z", "+00:00"))
        if when > cutoff:
            bad.append((name, version, upload))
    except Exception as e:
        neterr += 1
        print(f"[cooldown-check] WARN: pypi lookup failed for {name}=={version}: {e}", file=sys.stderr)

if bad:
    print("[cooldown-check] FAIL: packages newer than cutoff:")
    for n, v, t in bad:
        print(f"  - {n}=={v} released {t}")
    sys.exit(1)
strict = ${STRICT}
if neterr and strict:
    print(f"[cooldown-check] FAIL (strict): {neterr} pypi lookups failed", file=sys.stderr)
    sys.exit(1)
print(f"[cooldown-check] uv: {len(data.get('package', []))} packages OK ({neterr} network warnings)")
PYEOF

# --- 4) package-lock.json (if present) ---
if [[ -f package-lock.json ]]; then
  if ! command -v node >/dev/null 2>&1; then
    echo "[cooldown-check] SKIP: node not on PATH — npm age check unavailable on this host"
  else
    CUTOFF_ISO="${CUTOFF_ISO}" STRICT="${STRICT}" node <<'NODEEOF'
const fs = require("fs");
const https = require("https");
const cutoff = new Date(process.env.CUTOFF_ISO);
const strict = process.env.STRICT === "1";
const lock = JSON.parse(fs.readFileSync("package-lock.json", "utf8"));
const items = [];
for (const [k, v] of Object.entries(lock.packages || {})) {
  if (!k || !v || !v.version) continue;
  const name = k.replace(/^node_modules\//, "");
  if (!name) continue;
  items.push([name, v.version]);
}
function fetchTime(name) {
  return new Promise((res, rej) => {
    https.get("https://registry.npmjs.org/" + encodeURIComponent(name), r => {
      let d = "";
      r.on("data", c => (d += c));
      r.on("end", () => { try { res(JSON.parse(d)); } catch (e) { rej(e); } });
    }).on("error", rej);
  });
}
(async () => {
  const bad = [];
  let neterr = 0;
  for (const [name, version] of items) {
    try {
      const meta = await fetchTime(name);
      const t = meta.time && meta.time[version];
      if (t && new Date(t) > cutoff) bad.push([name, version, t]);
    } catch (e) {
      neterr++;
      process.stderr.write(`[cooldown-check] WARN: npm lookup failed ${name}@${version}: ${e}\n`);
    }
  }
  if (bad.length) {
    console.log("[cooldown-check] FAIL: npm packages newer than cutoff:");
    for (const b of bad) console.log("  -", b.join(" "));
    process.exit(1);
  }
  if (neterr && strict) {
    console.error(`[cooldown-check] FAIL (strict): ${neterr} npm lookups failed`);
    process.exit(1);
  }
  console.log(`[cooldown-check] npm: ${items.length} packages OK (${neterr} network warnings)`);
})();
NODEEOF
  fi
else
  echo "[cooldown-check] note: no package-lock.json yet (bootstrap)"
fi

echo "[cooldown-check] OK"
