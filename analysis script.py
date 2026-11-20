"""
Refill interruption and return to care among key populations on antiretroviral therapy in Nigeria.
Cohort of verified clients in care on 1 April 2024, followed to 10 April 2026.

Usage:
  python "analysis script.py" <pharmacy csv> <patient csv> <client verification csv> <RADET Nov 2024> <RADET Apr 2026> <output folder>
RADET files may be .xlsx or .csv. No names or hospital numbers are read.
"""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PH, PT, VER, R24, R26, OUT = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "analysis log.txt"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()

T0 = pd.Timestamp("2024-04-01")      # start of follow-up, after the January to March 2024 verification round
CLOSE = pd.Timestamp("2026-04-10")   # date of data extraction
PAUSE = pd.Timestamp("2025-01-24")   # United States foreign aid stop-work order
GRACE = 28                           # days after supply ran out before a client is counted as interrupted
KPS = ["FSW", "MSM", "PWID", "TRANSGENDER"]
days = lambda n: pd.to_timedelta(n, unit="D")

# ---------------------------------------------------------------- 1. load
def read_any(path, cols):
    if path.lower().endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(path); sh = "radet" if "radet" in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(xl, sheet_name=sh, usecols=lambda c: c in cols)
    return pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)

ph = pd.read_csv(PH, low_memory=False)
ph = ph.rename(columns={"Facility Name": "fac", "Patient Id": "pid", "Date Visit(yyyy-mm-dd)": "date", "Regimen Line": "line",
                        "Refill Period": "refill", "Next Appointment (yyyy-mm-dd)": "next"})
ph = ph[ph.line.astype(str).str.contains("Line")]              # antiretroviral regimen rows only
ph["date"] = pd.to_datetime(ph.date, errors="coerce"); ph["next"] = pd.to_datetime(ph["next"], errors="coerce")
ph = ph[ph.date.notna() & (ph.date <= CLOSE) & (ph.date >= "2000-01-01")]
ph["rf"] = ph.refill.where(ph.refill.between(7, 366))
log("Antiretroviral pharmacy rows:", len(ph), "clients:", ph.pid.nunique())
v = (ph.sort_values(["pid", "date"]).groupby(["pid", "date"], sort=False)
       .agg(fac=("fac", "last"), rf=("rf", "max"), nxt=("next", "max")).reset_index())
log("Pickup visits after removing same-day duplicate rows:", len(v), "clients:", v.pid.nunique())
log("Visits with no usable refill length (set to 30 days):", int(v.rf.isna().sum()))
v["rf"] = v.rf.fillna(30)
v["due"] = v.date + days(v.rf)                                  # day the dispensed supply ran out
gapn = (v.nxt - v.date).dt.days
v["due_appt"] = np.where(gapn.between(7, 366), v.nxt, v.due); v["due_appt"] = pd.to_datetime(v.due_appt)
log("Next appointment within 7 days of supply end: %.1f%%" % (100 * ((gapn - v.rf).abs() <= 7).mean()))
v = v.sort_values(["pid", "date"]); v["nextdate"] = v.groupby("pid").date.shift(-1)

pt = pd.read_csv(PT, low_memory=False, usecols=["State", "Patient Id", "Date Of Birth (yyyy-mm-dd)", "Sex", "Marital Status",
        "Education", "Occupation", "ART Start Date (yyyy-mm-dd)", "Last Viral Load", "Date of Last Viral Load (yyyy-mm-dd)"])
pt.columns = ["state", "pid", "dob", "sex", "marital", "edu", "occ", "artstart", "vl", "vldate"]
ve = pd.read_csv(VER, low_memory=False, usecols=["Patient Id", "Date of Observation", "Outcome"]); ve.columns = ["pid", "obs", "ver"]
a = read_any(R24, ["Patient ID", "Target group"]).rename(columns={"Patient ID": "pid", "Target group": "kp24"}).drop_duplicates("pid")
b = read_any(R26, ["Patient ID", "Target group", "Current ART Status", "Date of Current ART Status"])
b = b.rename(columns={"Patient ID": "pid", "Target group": "kp26", "Current ART Status": "status", "Date of Current ART Status": "statusdate"}).drop_duplicates("pid")
c = pt.merge(ve, on="pid", how="left").merge(a, on="pid", how="left").merge(b, on="pid", how="left")
for k in ["dob", "artstart", "obs", "statusdate", "vldate"]: c[k] = pd.to_datetime(c[k], errors="coerce")
c["kp"] = c.kp24.fillna(c.kp26).astype(str).str.upper()
c["invdate"] = c.obs.where(c.ver == "invalid")          # date a record was judged invalid
log("Verification entries:", len(ve), " dated January to March 2024:", int(ve.obs.pipe(pd.to_datetime, errors="coerce").between("2024-01-01", "2024-03-31").sum()))
log("Pharmacy clients not linked to the patient table:", int((~pd.Series(v.pid.unique()).isin(pt.pid)).sum()))

# ---------------------------------------------------------------- 2. cohort
first = v.groupby("pid").date.min().rename("firstpick").reset_index()
def build(t0, rule="main", verbose=False):
    last = v[v.date < t0].groupby("pid").tail(1)[["pid", "fac", "date", "rf", "due", "due_appt"]].rename(columns={"date": "lastpick"})
    hist = v[v.date < t0].copy(); hist["late"] = (hist.nextdate - hist.due).dt.days
    hist = hist[hist.nextdate.notna() & (hist.nextdate < t0)]
    h = hist.groupby("pid").agg(n_int=("late", "size"), n_late28=("late", lambda s: int((s > 28).sum())), n_late90=("late", lambda s: int((s > 90).sum()))).reset_index()
    cc = c.merge(first, on="pid").merge(last, on="pid", how="left").merge(h, on="pid", how="left")
    flow = [("Clients with at least one antiretroviral pickup", len(cc))]
    x = cc[cc.lastpick.notna()]; flow.append(("Had a pickup before the start of follow-up", len(x)))
    x = x[(x.due + days(GRACE)) >= t0]; flow.append(("In care at the start of follow-up (supply plus 28 days reached that date)", len(x)))
    incare = x.copy()
    if rule == "main":      # exclude only records already judged invalid when follow-up began; a later invalid outcome can follow from leaving care
        x = x[~((x.ver == "invalid") & (x.obs < t0))]; flow.append(("Record not judged invalid at client verification before the start of follow-up", len(x)))
    elif rule == "valid_q1":
        x = x[(x.ver == "valid") & (x.obs < T0) & (x.obs >= "2024-01-01")]; flow.append(("Record judged valid at client verification, January to March 2024", len(x)))
    else:
        x = x[x.ver == "valid"]; flow.append(("Record judged valid at client verification on any date", len(x)))
    x = x[x.kp.isin(KPS)]; flow.append(("Recorded as female sex worker, man who has sex with men, person who injects drugs or transgender person", len(x)))
    x["age"] = (t0 - x.dob).dt.days / 365.25
    if verbose: log("Age check: under 18:", int((x.age < 18).sum()), " over 90:", int((x.age > 90).sum()), " no date of birth:", int(x.age.isna().sum()))
    x = x[x.age.between(18, 90)]; flow.append(("Aged 18 years or older at the start of follow-up", len(x)))
    gone = x.status.isin(["Died", "Transferred Out", "Stopped Treatment"]) & (x.statusdate < t0)
    x = x[~gone]; flow.append(("No death, transfer or treatment stop recorded before the start of follow-up", len(x)))
    if verbose:
        for k, n in flow: log("FLOW  %-110s %7d" % (k, n))
    return x, incare, flow
x, allincare, flow = build(T0, verbose=True)

def derive(x, t0=T0):
    x = x.copy()
    for k in ["n_int", "n_late28", "n_late90"]: x[k] = x[k].fillna(0).astype(int)
    x["yrs_art"] = (t0 - x[["artstart", "firstpick"]].min(axis=1)).dt.days / 365.25
    x["agegrp"] = pd.cut(x.age, [17, 24.999, 34.999, 44.999, 120], labels=["18-24", "25-34", "35-44", "45+"])
    x["artgrp"] = pd.cut(x.yrs_art, [-1, 0.999, 2.999, 4.999, 100], labels=["<1", "1-2", "3-4", "5+"])
    x["refgrp"] = pd.cut(x.rf, [0, 59, 149, 400], labels=["1 month", "2-4 months", "5-6 months"])
    x["prior"] = pd.cut(x.n_late28, [-1, 0, 1, 2, 1000], labels=["0", "1", "2", "3+"])
    x["prior90"] = (x.n_late90 > 0).astype(int)
    x["edu3"] = x.edu.replace({"Senior Secondary": "Secondary", "Junior Secondary": "Secondary", "Post Secondary": "Post-secondary",
                               "Primary School": "Primary or none", "Quranic": "Primary or none", "None": "Primary or none"}).fillna("Not recorded")
    x.loc[~x.edu3.isin(["Secondary", "Post-secondary", "Primary or none", "Not recorded"]), "edu3"] = "Primary or none"
    x["mar2"] = np.where(x.marital == "Single", "Single", np.where(x.marital.isna(), "Not recorded", "Ever married"))
    x["occ3"] = x.occ.replace({"Freelance": "Employed", "Retired": "Unemployed"}).fillna("Not recorded")
    x["kpsex"] = np.where(x.kp == "PWID", np.where(x.sex == "Female", "PWID, female", "PWID, male"), x.kp)
    return x

def outcomes(x, v, grace=GRACE, duecol="due", close=CLOSE, t0=T0, censor_invalid=True):
    """First interruption after T0 and return to care after it."""
    w = v[v.pid.isin(x.pid)].copy()
    w["d"] = w[duecol]
    w = w[(w.d + days(grace)) >= t0]                       # intervals whose risk day falls on or after T0
    w["riskday"] = w.d + days(grace)
    w["missed"] = w.nextdate.isna() | (w.nextdate > w.riskday)
    ev = w[w.missed].groupby("pid").head(1)[["pid", "riskday", "nextdate", "date", "rf"]].rename(
        columns={"riskday": "evdate", "nextdate": "retdate", "date": "evpick", "rf": "evrf"})
    y = x.merge(ev, on="pid", how="left")
    sd = y.statusdate.where(y.status.isin(["Died", "Transferred Out", "Stopped Treatment"]) & (y.statusdate >= t0))
    iv = y.invdate.where(y.invdate >= t0) if censor_invalid else pd.Series(pd.NaT, index=y.index)
    # a record judged invalid during follow-up is censored on that date: most were closed as duplicates before or as their supply ran out
    exitd = pd.concat([sd, iv], axis=1).min(axis=1)
    end = pd.concat([exitd, pd.Series(close, index=y.index)], axis=1).min(axis=1)
    y["exit"] = np.where(exitd.notna() & (exitd <= close), np.where(iv.notna() & (iv <= sd.fillna(pd.Timestamp.max)), "Record judged invalid", y.status), "none")
    y["enddate"] = end
    y["event"] = (y.evdate.notna() & (y.evdate <= end)).astype(int)
    # a client whose death, transfer or stop is dated within 28 days after supply end is an exit, not an interruption
    y["t"] = (np.where(y.event == 1, y.evdate, end).astype("datetime64[ns]") - t0.to_datetime64()).astype("timedelta64[D]").astype(float)
    y["t"] = y.t.clip(lower=0.5)
    r = y[y.event == 1].copy()
    r["ret"] = (r.retdate.notna() & (r.retdate <= close)).astype(int)
    riv = r.invdate.where(r.invdate >= r.evdate) if censor_invalid else pd.Series(pd.NaT, index=r.index)
    rend = pd.concat([r.statusdate.where(r.status.isin(["Died", "Transferred Out", "Stopped Treatment"]) & (r.statusdate >= r.evdate)), riv,
                      pd.Series(close, index=r.index)], axis=1).min(axis=1)
    r["rexit"] = np.where(rend < close, np.where(riv.notna() & (riv <= rend), "Record judged invalid", r.status), "none")
    r["ret"] = np.where(r.ret.eq(1) & (r.retdate <= rend), 1, 0)
    r["rt"] = (np.where(r.ret == 1, r.retdate, rend).astype("datetime64[ns]") - r.evdate.values).astype("timedelta64[D]").astype(float)
    r["rt"] = r.rt.clip(lower=0.5)
    return y, r

x = derive(x); y, r = outcomes(x, v)
y["vergrp"] = np.where((y.ver == "valid") & (y.obs < T0), "Valid, Jan-Mar 2024", np.where(y.ver == "valid", "Valid, later date", np.where(y.ver == "invalid", "Invalid, later date", "No record")))
N = len(y); log("\nCOHORT", N, "clients;", y.fac.nunique(), "facilities;", y.state.nunique(), "states")
log("Person-years:", round(y.t.sum() / 365.25), " median follow-up (months):", round(y.t.median() / 30.44, 1))
log("First interruptions:", int(y.event.sum()), " rate per 100 person-years: %.1f" % (100 * y.event.sum() / (y.t.sum() / 365.25)))
log("Exits before interruption:", y[y.event == 0].exit.value_counts().to_dict())

# ---------------------------------------------------------------- 3. tables
xw = pd.ExcelWriter(os.path.join(OUT, "analysis results.xlsx"))
pd.DataFrame(flow, columns=["Step", "Clients"]).to_excel(xw, "Flow", index=False)
VARS = [("kp", "Key population"), ("sex", "Sex"), ("agegrp", "Age, years"), ("state", "State"), ("edu3", "Education"), ("mar2", "Marital status"),
        ("occ3", "Occupation"), ("artgrp", "Years on treatment"), ("refgrp", "Supply at last pickup"), ("prior", "Earlier late returns (>28 days)"),
        ("prior90", "Any earlier return >90 days late"), ("vergrp", "Client verification record")]
def km_at(df, tcol, ecol, times):
    k = KaplanMeierFitter().fit(df[tcol], df[ecol]); out = []
    for tt in times:
        s = float(k.predict(tt)); ci = k.confidence_interval_.loc[:tt].iloc[-1]
        out.append((100 * (1 - s), 100 * (1 - ci.iloc[1]), 100 * (1 - ci.iloc[0])))
    return out
rows = []
for col, lab in VARS:
    for lev, g in y.groupby(col, observed=True):
        k12, k24 = km_at(g, "t", "event", [365, 730])
        rows.append([lab, str(lev), len(g), round(100 * len(g) / N, 1), int(g.event.sum()), round(100 * g.event.sum() / (g.t.sum() / 365.25), 1),
                     round(k12[0], 1), round(k12[1], 1), round(k12[2], 1), round(k24[0], 1), round(k24[1], 1), round(k24[2], 1)])
k12, k24 = km_at(y, "t", "event", [365, 730])
rows.insert(0, ["All", "All", N, 100.0, int(y.event.sum()), round(100 * y.event.sum() / (y.t.sum() / 365.25), 1), round(k12[0], 1), round(k12[1], 1), round(k12[2], 1),
                round(k24[0], 1), round(k24[1], 1), round(k24[2], 1)])
T1 = pd.DataFrame(rows, columns=["Characteristic", "Level", "n", "%", "Interruptions", "Rate per 100 py", "12m %", "12m lo", "12m hi", "24m %", "24m lo", "24m hi"])
T1.to_excel(xw, "Table 1 incidence", index=False); log("\n" + T1.to_string(index=False))
log("\nAge median (IQR):", y.age.describe()[["25%", "50%", "75%"]].round(1).to_dict(), " Years on treatment:", y.yrs_art.describe()[["25%", "50%", "75%"]].round(1).to_dict())
log("Earlier intervals per client median:", y.n_int.median(), " any earlier late return: %.1f%%" % (100 * (y.n_late28 > 0).mean()))

# ---------------------------------------------------------------- 4. Cox models
def cox(df, tcol, ecol, terms, refs, strata="fac", label=""):
    X = pd.DataFrame(index=df.index)
    for t in terms:
        d = pd.get_dummies(df[t].astype(str), prefix=t, prefix_sep="=").astype(float)
        d = d.drop(columns=[f"{t}={refs[t]}"]); X = pd.concat([X, d], axis=1)
    X[tcol] = df[tcol]; X[ecol] = df[ecol]; X["fac"] = df["fac"].values; X["pid"] = df["pid"].values
    m = CoxPHFitter().fit(X.drop(columns=["pid"]), tcol, ecol, strata=[strata], robust=True)
    s = m.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(5); s.columns = ["HR", "lo", "hi", "p"]
    s.insert(0, "term", s.index); s.insert(0, "model", label)
    return m, s.reset_index(drop=True), X
REFS = {"kpsex": "FSW", "agegrp": "25-34", "edu3": "Secondary", "mar2": "Single", "occ3": "Employed", "artgrp": "3-4", "refgrp": "5-6 months", "prior": "0"}
TERMS = list(REFS)
out = []
for t in TERMS:
    _, s, _ = cox(y, "t", "event", [t], REFS, label="crude"); out.append(s)
mA, sA, XA = cox(y, "t", "event", TERMS, REFS, label="adjusted"); out.append(sA)
T2 = pd.concat(out); T2.to_excel(xw, "Table 2 Cox interruption", index=False); log("\nCOX, first interruption (stratified by clinic): one characteristic at a time, then adjusted\n" + T2.round(3).to_string(index=False))
log("Concordance, full model: %.3f" % mA.concordance_index_)
m0, _, _ = cox(y, "t", "event", ["prior"], REFS, label="prior only"); log("Concordance, earlier late returns only: %.3f" % m0.concordance_index_)
m1, _, _ = cox(y, "t", "event", [t for t in TERMS if t != "prior"], REFS, label="no prior"); log("Concordance, model without earlier late returns: %.3f" % m1.concordance_index_)
try:
    pht = proportional_hazard_test(mA, XA.drop(columns=["pid"]), time_transform="rank").summary[["test_statistic", "p"]].round(3)
    pht.to_excel(xw, "PH test"); log("\nProportional hazards test\n" + pht.to_string())
except Exception as e: log("PH test failed", e)

# return to care
log("\nRETURN TO CARE after first interruption: episodes", len(r), " returned", int(r.ret.sum()))
rr = []
r = r.merge(y[["pid", "vergrp"]], on="pid")
log("Returned, crude: %.1f%%; outcome of those not returned: %s" % (100 * r.ret.mean(), r[r.ret == 0].rexit.value_counts().to_dict()))
log("Days from supply end to return, median:", (r[r.ret == 1].rt + GRACE).median())
for lab, g in [("All", r)] + [(k, g) for k, g in r.groupby("kp")] + [("Earlier late returns " + str(k), g) for k, g in r.groupby("prior", observed=True)]:
    e = km_at(g, "rt", "ret", [28, 90, 180, 365]); k = KaplanMeierFitter().fit(g.rt, g.ret)
    rr.append([lab, len(g), int(g.ret.sum())] + [round(z, 1) for q in e for z in q] + [k.median_survival_time_])
T3 = pd.DataFrame(rr, columns=["Group", "Episodes", "Returned", "28d %", "lo", "hi", "90d %", "lo", "hi", "180d %", "lo", "hi", "365d %", "lo", "hi", "Median days"])
T3.to_excel(xw, "Table 3 return", index=False); log(T3.to_string(index=False))
mR, sR, _ = cox(r, "rt", "ret", TERMS, REFS, label="return adjusted"); sR.to_excel(xw, "Cox return", index=False)
log("\nCOX, return after interruption\n" + sR.round(3).to_string(index=False))
log("First interruptions by verification group:", y.groupby("vergrp").event.agg(["size", "sum"]).to_dict())
log("Status in April 2026 of those not returned:", r[r.ret == 0].status.value_counts(dropna=False).to_dict())

# ---------------------------------------------------------------- 5. facility variation and calendar time
fs = y.groupby(["state", "fac"]).apply(lambda g: pd.Series({"n": len(g), "events": g.event.sum(), "rate": 100 * g.event.sum() / (g.t.sum() / 365.25),
        "km12": km_at(g, "t", "event", [365])[0][0]})).reset_index()
fs = fs.sort_values(["state", "fac"]); fs["code"] = fs.groupby("state").cumcount().map(lambda i: chr(65 + i)); fs["label"] = fs.state + " " + fs.code
fs.to_excel(xw, "Facility key and rates", index=False); log("\nFACILITY 12-month risk range: %.1f to %.1f; median %.1f" % (fs.km12.min(), fs.km12.max(), fs.km12.median()))
log(fs.drop(columns=["fac"]).round(1).to_string(index=False))
w = v[v.pid.isin(y.pid)].copy(); w["riskday"] = w.due + days(GRACE)
w = w[(w.riskday >= T0) & (w.riskday <= CLOSE)]
w["exitd"] = w.pid.map(y.set_index("pid").enddate)
w = w[w.exitd >= w.riskday]
w["missed"] = (w.nextdate.isna() | (w.nextdate > w.riskday)).astype(int)
w["month"] = w.due.dt.to_period("M")
mo = w.groupby("month").missed.agg(["size", "sum"]); mo["pct"] = 100 * mo["sum"] / mo["size"]
mo = mo[(mo.index >= "2024-04") & (mo.index <= "2026-02")]
wb = v[v.pid.isin(y.pid)].copy(); wb["riskday"] = wb.due + days(GRACE); wb = wb[(wb.riskday < T0) & (wb.due >= "2023-01-01")]
wb["missed"] = (wb.nextdate > wb.riskday).astype(int); wb["month"] = wb.due.dt.to_period("M")
mb = wb.groupby("month").missed.agg(["size", "sum"]); mb["pct"] = 100 * mb["sum"] / mb["size"]; mb = mb[mb.index <= "2024-02"]
mb.to_excel(xw, "Monthly before baseline"); log("\nBEFORE BASELINE, same clients: intervals by month supply ran out (all returned, by definition of the cohort)\n" + mb.round(1).to_string())
w["fac"] = w.pid.map(y.set_index("pid").fac); w["per"] = np.where(w.due < "2024-07-01", "A", "B")
cp = w[w.due < "2025-01-01"].groupby(["fac", "per"]).missed.mean().unstack() * 100
log("Clinics where the percentage missed was higher in April to June 2024 than in July to December 2024: %d of %d" % ((cp.A > cp.B).sum(), len(cp)))
mo.to_excel(xw, "Monthly intervals"); log("\nDISPENSING INTERVALS by month supply ran out\n" + mo.round(1).to_string())
pre = w[(w.due >= "2024-04-01") & (w.due < PAUSE)]; post = w[(w.due >= PAUSE) & (w.due <= "2026-02-28")]
log("Intervals ending before pause: %d, missed %.2f%%; on or after: %d, missed %.2f%%" % (len(pre), 100 * pre.missed.mean(), len(post), 100 * post.missed.mean()))
log("Interruptions per client in follow-up:", w.groupby("pid").missed.sum().reindex(y.pid).fillna(0).value_counts().sort_index().to_dict())

# ---------------------------------------------------------------- 6. sensitivity analyses
sens = []
def summ(lab, yy, rr_):
    k = km_at(yy, "t", "event", [365, 730]); _, s, _ = cox(yy, "t", "event", TERMS, REFS, label=lab)
    g = s.set_index("term"); e = km_at(rr_, "rt", "ret", [90])[0] if len(rr_) else (np.nan,) * 3
    sens.append([lab, len(yy), int(yy.event.sum()), round(k[0][0], 1), round(k[1][0], 1)] +
                [f"{g.loc[t,'HR']:.2f} ({g.loc[t,'lo']:.2f}-{g.loc[t,'hi']:.2f})" for t in ["prior=1", "prior=2", "prior=3+", "kpsex=MSM", "kpsex=PWID, male", "kpsex=PWID, female", "kpsex=TRANSGENDER"]] + [round(e[0], 1)])
summ("Main analysis", y, r)
y2, r2 = outcomes(x, v, grace=90); summ("Interruption defined as more than 90 days late", y2, r2)
y3, r3 = outcomes(x, v, duecol="due_appt"); summ("Due date taken from the next appointment", y3, r3)
y4, r4 = outcomes(x, v, close=PAUSE); summ("Follow-up ended 24 January 2025", y4, r4)
xa, _, _ = build(T0, rule="valid_q1"); xa = derive(xa); y5, r5 = outcomes(xa, v); summ("Only records judged valid in January to March 2024", y5, r5)
log("Valid in Q1 2024 only: in care at end by status: ", y5.status.value_counts(dropna=False).to_dict())
xc, _, _ = build(T0, rule="valid_any"); xc = derive(xc); y8, r8 = outcomes(xc, v); summ("Only records judged valid on any date", y8, r8)
T0B = pd.Timestamp("2024-07-01"); xb, _, _ = build(T0B); xb = derive(xb, T0B); y7, r7 = outcomes(xb, v, t0=T0B); summ("Follow-up started 1 July 2024", y7, r7)
log("July 2024 start: person-years %d, rate per 100 py %.1f" % (y7.t.sum() / 365.25, 100 * y7.event.sum() / (y7.t.sum() / 365.25)))
y6 = y[y.n_int >= 2]; summ("At least two earlier dispensing intervals", y6, r[r.pid.isin(y6.pid)])
y9, r9 = outcomes(x, v, censor_invalid=False); summ("Records judged invalid during follow-up not censored", y9, r9)
log("Not censored: returned crude %.1f%%; not returned by line-list status: %s" % (100 * r9.ret.mean(), r9[r9.ret == 0].status.value_counts(dropna=False).to_dict()))
log("Not censored: first interruptions among records judged invalid during follow-up: %d of %d" % (y9[y9.invdate >= T0].event.sum(), (y9.invdate >= T0).sum()))
y10 = y[y.due >= T0]; summ("Supply had not run out before April 1, 2024", y10, r[r.pid.isin(y10.pid)]); log("Clients whose supply ran out before April 1, 2024: %d, first interruptions among them: %d" % ((y.due < T0).sum(), y[y.due < T0].event.sum()))
y11 = y[(y.edu3 != "Not recorded") & (y.mar2 != "Not recorded") & (y.occ3 != "Not recorded")]; summ("Complete information on education, marital status and occupation", y11, r[r.pid.isin(y11.pid)])
wx = v[v.pid.isin(y.pid) & v.nextdate.notna() & (v.due >= PAUSE)].copy(); wx["fac"] = wx.pid.map(y.set_index("pid").fac)
heap = (wx.nextdate == wx.due).groupby(wx.fac).mean().sort_values(ascending=False); top = list(heap.index[:4])
log("Share of pickups on the exact day supply ran out, from January 24, 2025, by clinic (highest four removed in sensitivity analysis):", heap.round(3).values.tolist())
y12 = y[~y.fac.isin(top)]; summ("Four clinics with the most pickups on the exact due day removed", y12, r[r.pid.isin(y12.pid)])
S = pd.DataFrame(sens, columns=["Analysis", "Clients", "Interruptions", "12m %", "24m %", "aHR 1 earlier", "aHR 2 earlier", "aHR 3+ earlier", "aHR MSM", "aHR PWID male", "aHR PWID female", "aHR transgender", "Returned by 90d %"])
S.loc[S.Analysis == "Follow-up ended 24 January 2025", "12m %"] = round(km_at(y4, "t", "event", [298])[0][0], 1)
S.to_excel(xw, "Sensitivity", index=False); log("\nSENSITIVITY\n" + S.to_string(index=False))
# clients in care but not in the cohort
nc = allincare[~allincare.pid.isin(y.pid)]; log("\nIn care on 1 April 2024 but not in cohort:", len(nc), nc.ver.fillna("not verified").value_counts().to_dict(), nc.kp.where(nc.kp.isin(KPS), "other or not recorded").value_counts().to_dict())

nk = allincare[~((allincare.ver == "invalid") & (allincare.obs < T0)) & ~allincare.kp.isin(KPS)].copy()
nk["grp"] = np.where(nk.kp.isin(["NAN", "NONE", ""]) | nk.kp24.isna() & nk.kp26.isna(), "No group on either line list", "Other group recorded")
for gname, g in nk.groupby("grp"):
    yy, _ = outcomes(g, v); kk = km_at(yy, "t", "event", [365])[0]
    log("Excluded for key population group, %s: %d records; first interruptions %d; 12-month risk %.1f%%" % (gname, len(yy), yy.event.sum(), kk[0]))
import lifelines, matplotlib as _m, platform
log("Software: Python", platform.python_version(), "pandas", pd.__version__, "lifelines", lifelines.__version__, "numpy", np.__version__)
# ---------------------------------------------------------------- 6b. further checks
y["evq"] = pd.PeriodIndex(y.evdate.where(y.event == 1), freq="Q").astype(str)
log("\nFirst interruptions by calendar quarter:", y[y.event == 1].evq.value_counts().sort_index().to_dict())
log("First interruptions in first 91 days: %d (%.1f%%)" % ((y.event.eq(1) & (y.t <= 91)).sum(), 100 * (y.event.eq(1) & (y.t <= 91)).sum() / y.event.sum()))
# landmark models: first 91 days and later
ye = y.copy(); ye["event"] = np.where((ye.event == 1) & (ye.t <= 91), 1, 0); ye["t"] = ye.t.clip(upper=91)
yl = y[(y.t > 91)].copy(); yl["t"] = yl.t - 91
_, sE, _ = cox(ye, "t", "event", TERMS, REFS, label="first 91 days"); _, sL, _ = cox(yl, "t", "event", TERMS, REFS, label="after 91 days")
LM = pd.concat([sE, sL]); LM.to_excel(xw, "Landmark Cox", index=False)
log("\nLANDMARK first 91 days: events %d\n%s" % (ye.event.sum(), sE.to_string(index=False))); log("\nLANDMARK after 91 days: n %d events %d\n%s" % (len(yl), yl.event.sum(), sL.to_string(index=False)))
# discrimination without stratification
def cidx(terms):
    X = pd.concat([pd.get_dummies(y[t].astype(str), prefix=t, drop_first=True).astype(float) for t in terms], axis=1); X["t"] = y.t.values; X["event"] = y.event.values
    return CoxPHFitter(penalizer=0.001).fit(X, "t", "event").concordance_index_
log("\nConcordance, unstratified: facility only %.3f; client characteristics only %.3f; both %.3f" % (cidx(["fac"]), cidx(TERMS), cidx(TERMS + ["fac"])))
# between-facility variation in 12-month risk (method of moments on facility log-odds)
y["ev12"] = ((y.event == 1) & (y.t <= 365)).astype(int)
f12 = y.groupby("fac").ev12.agg(["sum", "size"]); f12 = f12[f12["size"] >= 50]
lo = np.log((f12["sum"] + 0.5) / (f12["size"] - f12["sum"] + 0.5)); se2 = 1 / (f12["sum"] + 0.5) + 1 / (f12["size"] - f12["sum"] + 0.5)
wt = 1 / se2; mu = (wt * lo).sum() / wt.sum(); Q = (wt * (lo - mu) ** 2).sum(); tau2 = max(0, (Q - (len(lo) - 1)) / (wt.sum() - (wt ** 2).sum() / wt.sum()))
log("Between-facility variance of log-odds (12-month interruption): %.3f; median odds ratio %.2f; ICC %.3f" % (tau2, np.exp(np.sqrt(2 * tau2) * 0.6745), tau2 / (tau2 + np.pi ** 2 / 3)))
f2 = f12[f12["size"] >= 1000]; lo2 = np.log((f2["sum"] + 0.5) / (f2["size"] - f2["sum"] + 0.5)); s2 = 1 / (f2["sum"] + 0.5) + 1 / (f2["size"] - f2["sum"] + 0.5); w2_ = 1 / s2
mu2 = (w2_ * lo2).sum() / w2_.sum(); Q2 = (w2_ * (lo2 - mu2) ** 2).sum(); t2_ = max(0, (Q2 - (len(lo2) - 1)) / (w2_.sum() - (w2_ ** 2).sum() / w2_.sum()))
log("Without the clinic with 330 clients: median odds ratio %.2f (%d clinics)" % (np.exp(np.sqrt(2 * t2_) * 0.6745), len(f2)))
log("12-month interruption, crude:", int(y.ev12.sum()), "of", N, "= %.1f%%" % (100 * y.ev12.mean()))
# status and viral load at the end of follow-up
log("\nStatus on 10 April 2026:", y.status.value_counts(dropna=False).to_dict())
log(pd.crosstab(y.vergrp, y.status.fillna("none")).to_string())
lastp = v.groupby("pid").date.max(); y["lastp"] = y.pid.map(lastp)
y["incare_end"] = ((y.pid.map(v.groupby("pid").due.last()) + days(GRACE)) >= CLOSE) & ~y.status.isin(["Died", "Transferred Out", "Stopped Treatment"]) & ~(y.invdate >= T0)
_iv = y[y.invdate >= T0]; _ld = _iv.pid.map(v.groupby("pid").due.last())
log("Records judged invalid during follow-up: judged before the last supply had run out for 28 days: %.1f%%; before it ran out: %.1f%%" % (100 * (_iv.invdate <= _ld + days(GRACE)).mean(), 100 * (_iv.invdate <= _ld).mean()))
log("Records judged invalid during follow-up:", int((y.invdate >= T0).sum()), " deaths recorded:", int((y.status == "Died").sum()), " per 100 client-years over the two years: %.2f" % (100 * (y.status == "Died").sum() / (N * (CLOSE - T0).days / 365.25)))
log("In care on 10 April 2026 by pharmacy record: %d (%.1f%%)" % (y.incare_end.sum(), 100 * y.incare_end.mean()))
log(y.groupby("vergrp").incare_end.agg(["size", "sum", "mean"]).round(3).to_string())
log("Return by verification group:\n" + r.groupby("vergrp").ret.agg(["size", "sum", "mean"]).round(3).to_string())
vln = pd.to_numeric(y.vl, errors="coerce"); rec = y.vldate >= "2025-04-10"
log("Viral load result dated in the final 12 months: %d (%.1f%%); below 1000 copies/mL: %d (%.1f%%)" % (rec.sum(), 100 * rec.mean(), (vln[rec] < 1000).sum(), 100 * (vln[rec] < 1000).mean()))
ev = y.event == 1
log("  viral load result in final 12 months: with interruption %.1f%%, without %.1f%%" % (100 * rec[ev].mean(), 100 * rec[~ev].mean()))
log("  among clients with an interruption: result %d, below 1000: %.1f%%; without: result %d, below 1000: %.1f%%" % ((rec & ev).sum(), 100 * (vln[rec & ev] < 1000).mean(), (rec & ~ev).sum(), 100 * (vln[rec & ~ev] < 1000).mean()))
# timing of pickups relative to the day supply ran out, cohort clients, by period
w2 = v[v.pid.isin(y.pid) & v.nextdate.notna() & (v.due >= T0)].copy(); w2["late"] = (w2.nextdate - w2.due).dt.days
w2["per"] = np.where(w2.due < "2024-07-01", "Apr-Jun 2024", np.where(w2.due < PAUSE, "Jul 2024-23 Jan 2025", "24 Jan 2025 onward"))
tm = w2.groupby("per").late.agg(n="size", median="median", q10=lambda s: s.quantile(.1), q90=lambda s: s.quantile(.9), exact=lambda s: 100 * (s == 0).mean(), within7=lambda s: 100 * (s.abs() <= 7).mean())
w2["fac"] = w2.pid.map(y.set_index("pid").fac); w["per3"] = np.where(w.due < "2024-07-01", "Apr-Jun 2024", np.where(w.due < PAUSE, "Jul 2024-23 Jan 2025", "24 Jan 2025 onward"))
cl = w2.groupby(["fac", "per"]).late.agg(exact=lambda s: 100 * (s == 0).mean(), within7=lambda s: 100 * (s.abs() <= 7).mean()).join(w.groupby(["fac", "per3"]).missed.agg(intervals="size", missed=lambda s: 100 * s.mean()).rename_axis(["fac", "per"])).reset_index()
cl.to_excel(xw, "Clinic by period", index=False)
tm.to_excel(xw, "Pickup timing"); log("\nPICKUP TIMING (days after supply ran out)\n" + tm.round(1).to_string())
ms = pd.crosstab(y.kp, y.refgrp, normalize="index").mul(100).round(1); log("\nSupply length by group (%)\n" + ms.to_string())
xw.close()


# ---------------------------------------------------------------- 7. figure
plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
KC = {"FSW": "#c51b7d", "MSM": "#1f78b4", "PWID": "#33a02c", "TRANSGENDER": "#e66101"}
KL = {"FSW": "Female sex workers", "MSM": "Men who have sex with men", "PWID": "People who inject drugs", "TRANSGENDER": "Transgender people"}
LS = {"FSW": "-", "MSM": "--", "PWID": "-.", "TRANSGENDER": ":"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
for kpn in KPS:
    g = y[y.kp == kpn]; k = KaplanMeierFitter().fit(g.t / 30.44, g.event)
    ax[0].step(k.survival_function_.index, 100 * (1 - k.survival_function_.iloc[:, 0]), where="post", color=KC[kpn], ls=LS[kpn], lw=2, label=f"{KL[kpn]} (n={len(g):,})")
ax[0].set(xlabel="Months since April 1, 2024", ylabel="Clients with a first interruption (%)", xlim=(0, 24), ylim=(0, 22)); ax[0].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].plot(mb.index.to_timestamp(), mb.pct, color="#7f7f7f", lw=2, ls="--", marker="o", ms=3, label="Before follow-up (same clients)")
ax[1].plot(mo.index.to_timestamp(), mo.pct, color="#2c7fb8", lw=2, marker="o", ms=3.5, label="During follow-up")
ax[1].axvline(T0, color="#252525", ls=":", lw=1.2); ax[1].text(T0, ax[1].get_ylim()[1] * 0.99, " Follow-up\n began", fontsize=8, va="top")
ax[1].axvline(PAUSE, color="#d7301f", ls="--", lw=1.2); ax[1].text(PAUSE, ax[1].get_ylim()[1] * 0.99, " Stop-work\n order", color="#d7301f", fontsize=8, va="top")
ax[1].set(xlabel="Month the dispensed supply ran out", ylabel="Intervals with no pickup within 28 days (%)"); ax[1].set_ylim(bottom=0); ax[1].tick_params(axis="x", rotation=45); ax[1].legend(frameon=False, fontsize=8, loc="center right")
for kpn in KPS:
    g = r[r.kp == kpn]; k = KaplanMeierFitter().fit(g.rt, g.ret)
    ax[2].step(k.survival_function_.index, 100 * (1 - k.survival_function_.iloc[:, 0]), where="post", color=KC[kpn], ls=LS[kpn], lw=2, label=f"{KL[kpn]} (n={len(g):,})")
ax[2].set(xlabel="Days since interruption", ylabel="Clients back on treatment (%)", xlim=(0, 180), ylim=(0, 102)); ax[2].legend(frameon=False, fontsize=8, loc="lower right")
for a_, t in zip(ax, "abc"):
    a_.spines[["top", "right"]].set_visible(False); a_.text(-0.12, 1.04, t, transform=a_.transAxes, fontsize=14, fontweight="bold")
plt.tight_layout(); fig.savefig(os.path.join(OUT, "Fig1.png")); fig.savefig(os.path.join(OUT, "Fig1.tif"), pil_kwargs={"compression": "tiff_lzw"})
log("\nDone.")
