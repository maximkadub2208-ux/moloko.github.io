"""
Лабораторная работа 11. Система управления ресурсами молочного цеха.
Продукт: пастеризованное молоко. ДАННЫЕ СИНТЕТИЧЕСКИЕ (генератор ниже, SEED=42).
Все результаты получены на симуляционных данных и не доказывают экономию на реальном предприятии.
"""
import copy, json, math, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
STEP_MIN = 5; DT_H = STEP_MIN / 60; SPD = 288          # шагов в сутках
DAYS = 42; N = DAYS * SPD                              # 12 096 наблюдений
YIELD = 0.985                                          # выход продукта из сырья
WASH, GAP = 6, 6                                       # мойка (30 мин), пауза между партиями (30 мин)
L = 12                                                 # окно нейросети = 1 час
MODE_ID = {"production": 0, "waiting": 1, "washing": 2}
C_W = 0.12                                             # учебный тариф на воду, тг/л
def c_e(step):                                         # учебный тариф на энергию, тг/кВт·ч
    h = (np.asarray(step) % SPD) / 12
    return np.where((h >= 7) & (h < 23), 25.0, 15.0)

# ---------------------------------------------------------------- 1. Генератор данных
def physics(mode, flow, amb, inlet):
    """Зависимости: энергия = постоянная часть + нагрев + охлаждение/насосы; вода = охлаждение/мойка."""
    vol = flow * DT_H
    E = np.where(mode == 0, 0.8 + vol * (0.0113 + 0.0025 + 0.0004 * (amb - 18) + 0.00015 * (inlet - 4)),
                 np.where(mode == 2, 3.0, 0.6))
    W = np.where(mode == 0, vol * 0.05, np.where(mode == 2, 150.0, 2.0))
    out = np.where(mode == 0, vol * YIELD, 0.0)
    return E, W, out

def telemetry(mode, flow_set, rng=None):
    n = len(mode); step = np.arange(n)
    amb = 18 + 6 * np.sin(2 * np.pi * ((step % SPD) / 12 - 9) / 24)
    if rng is not None:
        amb = amb + rng.normal(0, 0.5, n)
        dr = np.zeros(n)
        for i in range(1, n): dr[i] = 0.9 * dr[i - 1] + rng.normal(0, 0.008)   # дрейф расхода (AR(1))
        flow = flow_set * (1 + dr); inlet = 4 + 0.05 * (amb - 18) + rng.normal(0, 0.3, n)
    else:
        flow = flow_set.astype(float); inlet = 4 + 0.05 * (amb - 18)
    prod, wash = mode == 0, mode == 2
    flow = np.where(prod, flow, 0.0)
    pt = np.where(prod, 72.0, np.where(wash, 80.0, 35.0))
    pr = np.where(prod, 2.0 + 0.0004 * flow, np.where(wash, 1.5, 0.3))
    ld = np.where(prod, flow / 5500 * 100, np.where(wash, 10.0, 3.0))
    E, W, out = physics(mode, flow, amb, inlet)
    if rng is not None:
        pt = pt + rng.normal(0, 0.25, n); pr = pr + rng.normal(0, 0.05, n); ld = ld + rng.normal(0, 0.5, n)
        E = E * (1 + rng.normal(0, 0.03, n)); W = W * (1 + rng.normal(0, 0.03, n))
    names = np.array(["production", "waiting", "washing"])
    bid = np.cumsum(prod & ~np.r_[False, prod[:-1]])
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq=f"{STEP_MIN}min"),
        "batch_id": np.where(prod | wash, bid, 0), "product_type": "pasteurized_milk",
        "milk_flow": flow, "inlet_temperature": inlet, "process_temperature": pt, "pressure": pr,
        "equipment_load": ld, "ambient_temperature": amb, "operating_mode": names[mode],
        "planned_mode": names[mode], "planned_flow_lh": flow_set.astype(float),   # план известен заранее
        "electricity_kwh": E, "water_l": W, "output_l": out, "downtime_min": np.where(mode == 1, STEP_MIN, 0)})

def make_schedule(days, rng):
    mode = np.ones(days * SPD, int); flow = np.zeros(days * SPD); batches = []
    for d in range(days):
        for s0 in (72, 144, 216):                      # партии около 06:00, 12:00, 18:00
            s = d * SPD + s0 + int(rng.integers(0, 12)); f = float(rng.integers(30, 51) * 100); dur = 48
            mode[s:s + dur] = 0; flow[s:s + dur] = f; mode[s + dur:s + dur + WASH] = 2
            batches.append((s, s + dur))
    return mode, flow, batches

rng = np.random.default_rng(SEED)
mode_arr, flow_arr, batches = make_schedule(DAYS, rng)
df = telemetry(mode_arr, flow_arr, rng)

# разбиение 70/15/15 по времени
T = np.arange(L - 1, N - 1); n = len(T); i1, i2 = int(0.7 * n), int(0.85 * n)
tr, va, te = T[:i1], T[i1:i2], T[i2:]

# сценарий отклонения: +25% энергии на 8 интервалов, 6 эпизодов, только в тестовом периоде
lab = np.zeros(N, bool)
cand = [b for b in batches if b[0] > te[0] + 1]
for k in rng.choice(len(cand), 6, replace=False):
    s, e = cand[k]; a = int(rng.integers(s + 6, e - 10)); lab[a:a + 8] = True
df.loc[lab, "electricity_kwh"] *= 1.25
df["anomaly_label"] = lab

# пропуски в датчиках (0.3%) — только для проверки очистки
sensors = ["milk_flow", "inlet_temperature", "process_temperature", "pressure", "ambient_temperature"]
for c in sensors: df.loc[rng.random(N) < 0.003, c] = np.nan
df.to_csv(os.path.join(OUT, "synthetic_dairy.csv"), index=False)

# ---------------------------------------------------------------- 2. Подготовка данных
print("Дубликаты времени:", df.timestamp.duplicated().sum(), "| монотонность:", df.timestamp.is_monotonic_increasing,
      "| пропуски:", int(df[sensors].isna().sum().sum()))
med = df.loc[:tr[-1], sensors].median()                # медианы только по обучающей части
clean = df.copy(); clean[sensors] = clean[sensors].ffill().fillna(med)

def make_features(d):
    f = pd.DataFrame(index=d.index)
    for c in ["milk_flow", "inlet_temperature", "process_temperature", "pressure", "equipment_load",
              "ambient_temperature", "electricity_kwh", "water_l"]: f[c] = d[c]
    mid = d.operating_mode.map(MODE_ID)
    for k, nm in enumerate(["prod", "wait", "wash"]): f["m_" + nm] = (mid == k).astype(float)
    f["E_lag1"] = d.electricity_kwh.shift(1); f["E_lag2"] = d.electricity_kwh.shift(2)
    f["E_roll6"] = d.electricity_kwh.rolling(6).mean(); f["W_lag1"] = d.water_l.shift(1)
    h = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    f["h_sin"] = np.sin(2 * np.pi * h / 24); f["h_cos"] = np.cos(2 * np.pi * h / 24)
    pm = d.planned_mode.shift(-1).map(MODE_ID)         # ПЛАН следующего интервала (известен заранее)
    for k, nm in enumerate(["prod", "wait", "wash"]): f["pm_" + nm] = (pm == k).astype(float)
    f["plan_flow_next"] = d.planned_flow_lh.shift(-1).fillna(0)
    return f

F = make_features(clean)
fcols = list(F.columns)
zcols = ["milk_flow", "process_temperature", "pressure", "equipment_load", "ambient_temperature",
         "electricity_kwh", "water_l", "m_prod", "m_wait", "m_wash", "pm_prod", "pm_wait", "pm_wash", "plan_flow_next"]
E_all, W_all = clean.electricity_kwh.values, clean.water_l.values
yE = {k: E_all[idx + 1] for k, idx in (("tr", tr), ("va", va), ("te", te))}
yW = {k: W_all[idx + 1] for k, idx in (("tr", tr), ("va", va), ("te", te))}
X = {k: F.iloc[idx].values for k, idx in (("tr", tr), ("va", va), ("te", te))}

def metrics(y, p): return mean_absolute_error(y, p), mean_squared_error(y, p) ** 0.5, r2_score(y, p)

# ---------------------------------------------------------------- 3. Модели машинного обучения
base = E_all[te]                                       # базовый прогноз: E(t+1) = E(t)
sc = StandardScaler().fit(X["tr"]); lin = LinearRegression().fit(sc.transform(X["tr"]), yE["tr"])
m_E = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=SEED).fit(X["tr"], yE["tr"])
m_W = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=SEED).fit(X["tr"], yW["tr"])
p_lin = lin.predict(sc.transform(X["te"])); p_gb = m_E.predict(X["te"])
mk_rf = lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
mk_et = lambda: ExtraTreesRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
m_rf, m_et = mk_rf().fit(X["tr"], yE["tr"]), mk_et().fit(X["tr"], yE["tr"])
m_rfW, m_etW = mk_rf().fit(X["tr"], yW["tr"]), mk_et().fit(X["tr"], yW["tr"])
lin_W = LinearRegression().fit(sc.transform(X["tr"]), yW["tr"])

# ---------------------------------------------------------------- 4. Нейросеть по окну L=12
Z = F[zcols].fillna(0).values
zs = StandardScaler().fit(Z[:tr[-1] + 1]); Zs = zs.transform(Z)
win = lambda idx: Zs[idx[:, None] + np.arange(-L + 1, 1)]
W3 = {k: win(idx) for k, idx in (("tr", tr), ("va", va), ("te", te))}

def fit_nn(Xtr, ytr, Xva, yva):
    mu, sd = ytr.mean(), ytr.std()
    try:
        import torch, torch.nn as nn
        torch.manual_seed(SEED)
        class Net(nn.Module):
            def __init__(s, f):
                super().__init__(); s.l = nn.LSTM(f, 32, batch_first=True); s.o = nn.Linear(32, 1)
            def forward(s, x): h, _ = s.l(x); return s.o(h[:, -1]).squeeze(-1)
        net = Net(Xtr.shape[2]); opt = torch.optim.Adam(net.parameters(), 1e-3)
        T_ = lambda a: torch.tensor(a, dtype=torch.float32)
        xt, yt, xv, yv = T_(Xtr), T_((ytr - mu) / sd), T_(Xva), T_((yva - mu) / sd)
        best, bad, state = 1e9, 0, None
        for ep in range(100):
            net.train(); perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 128):
                b = perm[i:i + 128]; opt.zero_grad()
                nn.functional.mse_loss(net(xt[b]), yt[b]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): v = nn.functional.mse_loss(net(xv), yv).item()
            if v < best - 1e-6: best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8: break                      # ранняя остановка
        net.load_state_dict(state); net.eval()
        def predict(Xn):
            with torch.no_grad(): return net(T_(Xn)).numpy() * sd + mu
        return predict, "LSTM (PyTorch)"
    except ImportError:                                 # запасной вариант без PyTorch
        flat = lambda a: a.reshape(len(a), -1)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), batch_size=128, learning_rate_init=1e-3, random_state=SEED)
        best, bad, bm = 1e9, 0, None
        for ep in range(200):
            mlp.partial_fit(flat(Xtr), (ytr - mu) / sd)
            v = mean_absolute_error((yva - mu) / sd, mlp.predict(flat(Xva)))
            if v < best - 1e-5: best, bad, bm = v, 0, copy.deepcopy(mlp)
            else:
                bad += 1
                if bad >= 10: break                     # ранняя остановка
        return (lambda Xn: bm.predict(flat(Xn)) * sd + mu), "MLP по окну L=12 (замена LSTM: нет PyTorch)"

nn_pred, nn_name = fit_nn(W3["tr"], yE["tr"], W3["va"], yE["va"])
p_nn = nn_pred(W3["te"])

PT = {"Базовый": base, "Линейная": p_lin, "Случайный лес": m_rf.predict(X["te"]), "Бустинг": p_gb,
      "Доп. деревья": m_et.predict(X["te"]), "Нейросеть (окно)": p_nn}
rows = [(k, *metrics(yE["te"], v)) for k, v in PT.items()]
PWt = {"Базовый": W_all[te], "Линейная": lin_W.predict(sc.transform(X["te"])), "Случайный лес": m_rfW.predict(X["te"]),
       "Бустинг": m_W.predict(X["te"]), "Доп. деревья": m_etW.predict(X["te"])}
wm = [dict(name=k, **dict(zip(("mae", "rmse", "r2"), map(float, metrics(yW["te"], v))))) for k, v in PWt.items()]
pd.DataFrame(wm).round(4).to_csv(os.path.join(OUT, "results_models_water.csv"), index=False)
res_models = pd.DataFrame(rows, columns=["Модель", "MAE, кВт·ч", "RMSE, кВт·ч", "R2"]).round(4)
res_models.to_csv(os.path.join(OUT, "results_models.csv"), index=False)
print("\n", res_models.to_string(index=False))
print("Вода (бустинг), тест: MAE=%.3f RMSE=%.3f R2=%.3f" % metrics(yW["te"], m_W.predict(X["te"])))
print("Тест аномалий исключён из обучения/валидации: метрики на тесте включают участки с отклонениями.")

# ---------------------------------------------------------------- 5. Обнаружение отклонений
def alarm_of(r, thr_, K=3):
    out_, run_ = np.zeros(len(r), bool), 0
    for i, e in enumerate(r > thr_): run_ = run_ + 1 if e else 0; out_[i] = run_ >= K
    return out_
PV = {"Базовый": E_all[va], "Линейная": lin.predict(sc.transform(X["va"])), "Случайный лес": m_rf.predict(X["va"]),
      "Бустинг": m_E.predict(X["va"]), "Доп. деревья": m_et.predict(X["va"]), "Нейросеть (окно)": nn_pred(W3["va"])}
thr_m = {k: float(np.percentile(np.abs(yE["va"] - v), 95)) for k, v in PV.items()}   # порог по валидации для каждой модели
R_m = {k: np.abs(yE["te"] - v) for k, v in PT.items()}; AL = {k: alarm_of(R_m[k], thr_m[k]) for k in PT}
best = min([k for k in PV if k != "Базовый"], key=lambda k: mean_absolute_error(yE["va"], PV[k]))   # лучшая по валидации
K = 3; thr = thr_m[best]; p_te = PT[best]; r_te = R_m[best]; al = AL[best]
def runs(b):
    out, s = [], None
    for i, v in enumerate(np.r_[b, False]):
        if v and s is None: s = i
        if not v and s is not None: out.append((s, i - 1)); s = None
    return out
lab_te = lab[te + 1]; ev = runs(lab_te); alerts = runs(al)
det = [(s, e, np.flatnonzero(al[s:e + K + 1])) for s, e in ev]
recall = np.mean([len(d[2]) > 0 for d in det]); delays = [d[2][0] * STEP_MIN for d in det if len(d[2])]
prec = np.mean([any(a <= e + K and b >= s for s, e in ev) for a, b in alerts]) if alerts else float("nan")
print(f"\nПорог (95-й процентиль, валидация): {thr:.3f} кВт·ч | эпизодов отклонений: {len(ev)} | предупреждений: {len(alerts)}"
      f"\nprecision={prec:.2f} recall={recall:.2f} задержка, мин: {delays}")
pd.DataFrame([dict(threshold=thr, episodes=len(ev), alerts=len(alerts), precision=prec, recall=recall,
                   mean_delay_min=np.mean(delays) if delays else None)]).to_csv(os.path.join(OUT, "results_anomaly.csv"), index=False)

# график «факт — прогноз»
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
sl = slice(0, 288 * 2); tt = df.timestamp.values[te + 1][sl]
ax[0].plot(tt, yE["te"][sl], label="факт", lw=1); ax[0].plot(tt, p_gb[sl], label="бустинг", lw=1)
ax[0].plot(tt, p_nn[sl], label=nn_name.split(" (")[0], lw=1, alpha=.7); ax[0].set_ylabel("кВт·ч / 5 мин"); ax[0].legend(); ax[0].set_title("Факт и прогноз электроэнергии (тест, синтетические данные)")
ax[1].plot(tt, r_te[sl], lw=1, label="|ошибка|"); ax[1].axhline(thr, color="r", ls="--", label="порог"); ax[1].scatter(tt[al[sl]], r_te[sl][al[sl]], c="r", s=10, label="предупреждение")
ax[1].legend(); ax[1].set_ylabel("кВт·ч")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_fact_forecast.png"), dpi=130); plt.close()

# ---------------------------------------------------------------- 6. Рекомендации по ресурсам
def build_line(bs):
    m = np.ones(SPD, int); f = np.zeros(SPD); ok = True
    for s, d, fl in bs:
        if s < 0 or s + d + WASH > SPD: ok = False; continue
        if (m[s:s + d + WASH] != 1).any(): ok = False
        m[s:s + d] = 0; f[s:s + d] = fl; m[s + d:s + d + WASH] = 2
    return m, f, ok

cache = {}
def forecast_line(bs):
    """Прогноз E и W на сутки: расписание повторяется 2 дня, берём второй (есть история для лагов)."""
    key = tuple(bs)
    if key not in cache:
        m, f, _ = build_line(bs); d2 = telemetry(np.tile(m, 2), np.tile(f, 2))
        Xs = make_features(d2).iloc[np.arange(SPD - 1, 2 * SPD - 1)].values
        cache[key] = (np.clip(m_E.predict(Xs), 0, None), np.clip(m_W.predict(Xs), 0, None),
                      d2.electricity_kwh.values[SPD:], d2.water_l.values[SPD:])
    return cache[key]

def evaluate(lines, Qp, M, H, blocked):
    ms, fs, ok = [], [], True
    for ln in lines:
        m, f, o = build_line(ln); ok &= o; ms.append(m); fs.append(f)
    out = sum(np.where(m == 0, f / 12 * YIELD, 0).sum() for m, f in zip(ms, fs))
    busy = [float((m != 1).sum() / 12) if ln else 0.0 for m, ln in zip(ms, lines)]
    fl_ok = all(3000 - 1e-6 <= b[2] <= 5000 + 1e-6 for ln in lines for b in ln)
    blk = blocked is not None and bool((ms[blocked[0]][blocked[1]:blocked[2]] != 1).any())
    cons = dict(Q=out >= Qp - 1e-6, M=out / YIELD <= M + 1e-6, H=all(b <= H + 1e-9 for b in busy), valid=ok and fl_ok and not blk)
    r = dict(lines=lines, Q=out, M=out / YIELD, busy=busy, feasible=all(cons.values()), cons=cons,
             E=0.0, W=0.0, E_sim=0.0, W_sim=0.0, cost=0.0)
    tar = c_e(np.arange(SPD))
    for ln in lines:
        if ln:
            e, w, es, ws = forecast_line(ln)
            r["E"] += e.sum(); r["W"] += w.sum(); r["E_sim"] += es.sum(); r["W_sim"] += ws.sum()
            r["cost"] += (tar * e).sum() + C_W * w.sum()
    r["load"] = [[float(np.mean(np.where(m == 0, f / 5500 * 100, np.where(m == 2, 10, 3) if ln else 0)[h * 12:(h + 1) * 12])) if ln else 0.0
                  for h in range(24)] for m, f, ln in zip(ms, fs, lines)]
    return r

def original(Q):
    d = math.ceil(Q / 3 / (4000 / 12 * YIELD)); return [[(72, d, 4000.0), (144, d, 4000.0), (216, d, 4000.0)], []]

def candidates(Q):
    seen, out = set(), []
    for Lc in (1, 2):
        for nb in range(Lc, 5):
            for Fg in range(3000, 5001, 500):
                d = math.ceil(Q / nb / (Fg / 12 * YIELD)); Fx = Q / nb * 12 / (YIELD * d)
                if Fx < 3000: continue
                for s0 in range(0, 13, 2):
                    seq = lambda c: [(s0 * 12 + j * (d + WASH + GAP), d, Fx) for j in range(c)]
                    opts = [[seq(nb), []], [[], seq(nb)]] if Lc == 1 else [[seq((nb + 1) // 2), seq(nb // 2)]]
                    for o in opts:
                        k = str(o)
                        if k not in seen: seen.add(k); out.append(o)
    return out

def shifted(Q):
    b0 = original(Q)[0]
    return [[[(s + sh, d, f) for s, d, f in b0], []] for sh in range(-72, 73, 6)]

SC = [("Базовый", dict(Q=40000, M=45000, H=14, blk=None)),
      ("Повышенный спрос", dict(Q=72000, M=75000, H=14, blk=None)),
      ("Ограничение сырья", dict(Q=40000, M=30000, H=14, blk=None)),
      ("Простой линии", dict(Q=40000, M=45000, H=14, blk=(0, 120, 192))),
      ("Изменение расписания", dict(Q=40000, M=45000, H=14, blk=None, shift=True))]
sc_rows, sc_json = [], {}
for name, p in SC:
    Qp, reduced = p["Q"], False
    if Qp > p["M"] * YIELD: Qp, reduced = 29500, True   # сырья не хватает: допустимый объём выпуска
    o = evaluate(original(Qp), Qp, p["M"], p["H"], p["blk"])
    pool = shifted(Qp) if p.get("shift") else candidates(Qp)
    ev_ = [evaluate(c, Qp, p["M"], p["H"], p["blk"]) for c in pool]
    ev_ = [e for e in ev_ if e["feasible"]]
    b = min(ev_, key=lambda e: e["cost"]) if ev_ else None
    row = dict(scenario=name, plan_Q=p["Q"], Q_used=Qp, plan_reduced=reduced,
               orig_feasible=o["feasible"], orig_Q=o["Q"], orig_E=o["E"], orig_W=o["W"], orig_cost=o["cost"],
               rec_feasible=b is not None)
    if b:
        row.update(rec_Q=b["Q"], rec_E=b["E"], rec_W=b["W"], rec_cost=b["cost"], rec_E_sim=b["E_sim"], rec_W_sim=b["W_sim"],
                   rec_busy_h=b["busy"], rec_lines=str(b["lines"]), saving_pct=100 * (o["cost"] - b["cost"]) / o["cost"])
    sc_rows.append(row)
    pack = lambda e: None if e is None else dict(Q=e["Q"], E=e["E"], W=e["W"], cost=e["cost"], feasible=e["feasible"],
                                                 E_per_l=e["E"] / e["Q"], W_per_l=e["W"] / e["Q"], busy=e["busy"], load=e["load"])
    sc_json[name] = dict(original=pack(o), recommended=pack(b), plan_Q=p["Q"], Q_used=Qp, plan_reduced=reduced)
    print(f"\n[{name}] план {p['Q']} л -> {Qp} л | исходное: допустимо={o['feasible']} J={o['cost']:.0f} тг"
          + (f" | рекомендуемое: J={b['cost']:.0f} тг, E={b['E']:.0f} кВт·ч, W={b['W']:.0f} л, занятость={np.round(b['busy'],1)}, "
             f"линии={[len(x) for x in b['lines']]} партий" if b else " | допустимого варианта нет"))
pd.DataFrame(sc_rows).to_csv(os.path.join(OUT, "results_scenarios.csv"), index=False)

# таблица управления ресурсами (базовый сценарий)
bs = sc_json["Базовый"]
keys = ("Q", "E", "W", "E_per_l", "W_per_l", "cost")
col = lambda e: [round(e[k], 3) for k in keys] + ["Да" if e["feasible"] else "Нет"]
res_res = pd.DataFrame({"Показатель": ["Выпуск продукции, л", "Потребление энергии, кВт·ч", "Потребление воды, л",
                                       "Энергия на литр, кВт·ч/л", "Вода на литр, л/л", "Затраты на энергию и воду, тг", "Выполнение ограничений"],
                        "Исходный режим": col(bs["original"]), "Рекомендуемый режим": col(bs["recommended"])})
res_res.to_csv(os.path.join(OUT, "results_resources.csv"), index=False); print("\n", res_res.to_string(index=False))

# ---------------------------------------------------------------- 7. JSON для виртуального цеха
pw = m_W.predict(X["te"]); j0 = max(0, ev[0][0] - 40); j1 = min(len(te) - 1, ev[0][0] + 70)
frames = []
for j in range(j0, j1):
    t = te[j] + 1; r = clean.iloc[t]
    frames.append(dict(ts=str(r.timestamp), mode=r.operating_mode, flow=round(float(r.milk_flow), 1), inlet=round(float(r.inlet_temperature), 2),
                       temp=round(float(r.process_temperature), 2), pressure=round(float(r.pressure), 3), load=round(float(r.equipment_load), 1),
                       E=round(float(df.electricity_kwh[t]), 3), W=round(float(df.water_l[t]), 2), out=round(float(df.output_l[t]), 1),
                       E_pred_now=round(float(p_te[j]), 3), E_next=round(float(p_te[j + 1]), 3), W_next=round(float(pw[j + 1]), 2),
                       resid=round(float(r_te[j]), 3), alarm=bool(al[j]),
                       preds={k: [round(float(PT[k][j]), 3), round(float(PT[k][j + 1]), 3), bool(AL[k][j]), round(float(R_m[k][j]), 3)] for k in PT}))
models = [dict(name=k, **dict(zip(("mae", "rmse", "r2"), map(float, metrics(yE["te"], v)))), thr=thr_m[k], alerts=len(runs(AL[k]))) for k, v in PT.items()]
pi = permutation_importance(m_E, X["te"], yE["te"], n_repeats=3, random_state=SEED, scoring="neg_mean_absolute_error")
imp = [dict(name=fcols[i], value=float(pi.importances_mean[i])) for i in np.argsort(-pi.importances_mean)[:8]]
dsinfo = dict(n=N, days=DAYS, step_min=STEP_MIN, source="Синтетические данные", split=[len(tr), len(va), len(te)], episodes=len(ev),
              seed=SEED, nn=nn_name, features=len(fcols), window=L)
payload = dict(generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), source="СИНТЕТИЧЕСКИЕ ДАННЫЕ (симуляция, SEED=42)",
               threshold=round(float(thr), 3), model=best, models=models, water_models=wm, importance=imp, dataset=dsinfo,
               thresholds={k: round(v, 3) for k, v in thr_m.items()}, frames=frames, scenarios=sc_json,
               tariffs=dict(energy_day=25, energy_night=15, water_per_l=C_W))
js = json.dumps(payload, ensure_ascii=False, default=float)
open(os.path.join(OUT, "workshop_data.json"), "w", encoding="utf-8").write(js)
tpl = os.path.join(OUT, "workshop_template.html")
if os.path.exists(tpl):
    open(os.path.join(OUT, "workshop.html"), "w", encoding="utf-8").write(open(tpl, encoding="utf-8").read().replace("/*__DATA__*/null", js))
print("\nВерсии: python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__)
