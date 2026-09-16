# -*- coding: utf-8 -*-
"""Pull female age at first marriage (World Bank SP.DYN.SMAM.FE, sourced from UN)
   as an INDEPENDENT demographic anchor for the A(t) approx A0 reduction condition.
   Save to _smam_female_worldbank.csv and report coverage honestly:
   how many countries, points per country, year range, overlap with the S panel,
   and how many countries have ENOUGH points to estimate a rise-rate curvature.
   Read-only on the paper; only writes a new data csv + prints diagnostics. (2026-06-30)"""
import sys, json, csv, urllib.request, urllib.error, time
sys.stdout.reconfigure(encoding='utf-8')

IND = "SP.DYN.SMAM.FE"  # Age at first marriage, female
URL = (f"https://api.worldbank.org/v2/country/all/indicator/{IND}"
       f"?format=json&per_page=20000")
OUT = r"data\_smam_female_worldbank.csv"
SPANEL = r"data\_panel_S_MAC.csv"

# ---- the 191-country ISO3 set the paper actually models (from the S panel) ----
spanel_codes = set()
with open(SPANEL, encoding='utf-8-sig') as f:  # utf-8-sig strips a leading BOM
    r = csv.DictReader(f)
    for row in r:
        spanel_codes.add(row['code'])
print(f"S panel (paper-modelled) countries: {len(spanel_codes)}")

# ---- pull from World Bank API (with a couple of retries) ----
def fetch(url, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'research/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:  # noqa
            last = e
            print(f"  retry {k+1}/{tries} after error: {e}")
            time.sleep(2 * (k + 1))
    raise last

print(f"\nFetching World Bank {IND} (female age at first marriage) ...")
payload = fetch(URL)
if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
    print("API returned no data block. Raw header:")
    print(json.dumps(payload[0] if isinstance(payload, list) else payload, indent=2)[:800])
    sys.exit(1)

meta, rows = payload[0], payload[1]
print(f"API meta: total={meta.get('total')}, pages={meta.get('pages')}, "
      f"per_page={meta.get('per_page')}")

# ---- keep only real countries (ISO3 in S panel), non-null values ----
records = []  # (code, year, value)
for rec in rows:
    code = (rec.get('countryiso3code') or '').strip()
    val = rec.get('value')
    date = rec.get('date')
    if not code or val is None or date is None:
        continue
    if code not in spanel_codes:
        continue  # drops aggregates (WLD/ARB/EUU...) and non-modelled territories
    try:
        records.append((code, int(date), float(val)))
    except (TypeError, ValueError):
        continue

records.sort(key=lambda t: (t[0], t[1]))

# ---- write csv ----
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['code', 'year', 'smam_fe'])
    w.writerows(records)
print(f"\nWrote {len(records)} obs -> {OUT}")

# ---- coverage diagnostics (the honest part) ----
from collections import defaultdict
by_c = defaultdict(list)
for code, yr, v in records:
    by_c[code].append(yr)

n_countries = len(by_c)
all_years = [yr for _, yr, _ in records]
pts = sorted(len(v) for v in by_c.values())
def med(xs):
    n = len(xs)
    if n == 0: return float('nan')
    s = sorted(xs)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

overlap = set(by_c) & spanel_codes
ge5 = sum(1 for v in by_c.values() if len(v) >= 5)
ge8 = sum(1 for v in by_c.values() if len(v) >= 8)
ge12 = sum(1 for v in by_c.values() if len(v) >= 12)

print("\n===== COVERAGE (honest) =====")
print(f"countries with >=1 SMAM obs (in S panel): {n_countries} / {len(spanel_codes)}")
print(f"total observations: {len(records)}")
print(f"year range: {min(all_years)}..{max(all_years)}")
print(f"points/country: min={pts[0]}, median={med(pts)}, max={pts[-1]}")
print(f"countries with >=5 time points : {ge5}")
print(f"countries with >=8 time points : {ge8}")
print(f"countries with >=12 time points: {ge12}")
print("\n(>=8 points is roughly the floor for estimating whether the RISE is "
      "decelerating, i.e. whether the oscillation amplitude has begun to damp.)")

# show the best-covered countries
ranked = sorted(by_c.items(), key=lambda kv: -len(kv[1]))[:15]
print("\nbest-covered countries (code: n_points, yr_min..yr_max):")
for code, yrs in ranked:
    print(f"  {code}: {len(yrs)}  ({min(yrs)}..{max(yrs)})")
