"""
main.py
-------------------------------------------------
FULL PIPELINE (MACRO + MICRO + VISUAL)

What this script does:
1. Load Prophet, TCN_v1, TCN_v2, and the saved scaler from your friend's
   Spot_Gold_Forecasting_Model.ipynb to generate a MACRO BIAS:
      - "LONG", "SHORT", or "NONE"
      - plus confidence ("HIGH", "MEDIUM", "LOW")

2. Load your SAGAN Discriminator (D_SAGAN_L256_E50.pt),
   run it on the latest 256 ticks from recent_ticks.csv to get MICROSTRUCTURE:
      - "BUY", "SELL", "NEUTRAL"
      - risk_state: "NORMAL" or "HIGH_RISK_OOD"

3. Fuse both:
      - If HIGH_RISK_OOD  -> NO TRADE
      - If LONG + BUY     -> LONG ENTRY
      - If SHORT + SELL   -> SHORT ENTRY
      - Else              -> WAIT

4. Suggest levels:
      - Suggested Entry (mid at signal)
      - Stop Loss (risk in USD)
      - Take Profit (reward in USD)

5. Plot a scalp overlay:
      - mid price over last 256 ticks
      - action color
      - entry / SL / TP lines


YOU MUST EDIT THESE BEFORE RUNNING:
- MODEL_DIR                (path to your model folder / drive)
- FEATURE_ORDER            (channel order used to train D_SAGAN_L256_E50.pt)
- Discriminator1D class    (paste EXACT class from your ISY5002_v2_1.ipynb)
- MU_D, SIGMA_D            (mean/std of discriminator scores on REAL training windows)
- CUSTOM_OBJECTS_TCN       (if your TCN model uses a custom TCN layer)

Files expected in MODEL_DIR:
    prophet_model.joblib
    tcn_v1.keras
    tcn_v2.keras
    tcn_scaler.joblib
    recent_daily_prices.csv    (daily OHLC history)
    recent_ticks.csv           (latest rolling 256 ticks)
    D_SAGAN_L256_E50.pt

Run:
    python integrated_gold_pipeline.py
"""

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

from datetime import datetime
import json
import sys

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

try:
    import tensorflow as tf
except Exception as e:
    tf = None

try:
    from prophet import Prophet
except Exception as e:
    Prophet = None

try:
    from tcn import TCN
except Exception as e:
    TCN = None



# ============================================================
# ================= USER CONFIG ==============================
# ============================================================

# QuantAlpha root (one level up from src/)
ROOT = Path(__file__).resolve().parents[1]  # .../QuantAlpha

# Common subdirs 
SRC_DIR   = ROOT / "src"
DATA_DIR  = SRC_DIR / "data"     
PROPHET_MODELS_DIR = SRC_DIR / "models" / "prophet"  
TCN_MODELS_DIR = SRC_DIR / "models" / "tcn"   
SAGAN_MODELS_DIR = SRC_DIR / "models" / "sagan"  

DAILY_PATH        = DATA_DIR / "recent_daily_prices_Jan_July.csv"
TICK_SOURCE_PATH  = DATA_DIR / "recent_ticks_29.08.2025.csv"
DISCRIMINATOR_PATH= SAGAN_MODELS_DIR / "D_SAGAN_L256_E50.pt"

PROPHET_PATH      = PROPHET_MODELS_DIR / "prophet_model.joblib"
TCN_V1_PATH       = TCN_MODELS_DIR / "tcn_v1_baseline.keras"
TCN_V2_PATH       = TCN_MODELS_DIR / "tcn_v2_improved.keras"
SCALER_PATH       = TCN_MODELS_DIR / "tcn_v2_improved.joblib"

CUSTOM_OBJECTS_TCN = {"TCN": TCN}

# Same channel order used to train discriminator.
FEATURE_ORDER = [
    "mid_rel","microprice_rel",
    "spread_z","imbalance_z",
    "d_mid_z","d_spread_z","d_imb_z",
    "rolling_vol_z",
    "bid_volume_log1p_z","ask_volume_log1p_z"
]
# Compute once offline: run D on many real 256-tick windows, take mean & std.
MU_D    = 0.75  # <-- ADJUST MEAN
SIGMA_D = 0.10  # <-- ADJUST STD

# Threshold for microstructural dominance.
IMBALANCE_TH = 0.5   # >0.5 means strong bid imbalance, <-0.5 strong ask imbalance

# Risk/Reward in USD per ounce for scalp suggestion.
RISK_USD_PER_OZ   = 0.5
REWARD_USD_PER_OZ = 1.0

def ensure_run_dir(prefix="gold_pipeline"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "logs" / f"{prefix}_{ts}"
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    return run_dir

class Tee:
    def __init__(self, *targets):
        self.targets = targets
    def write(self, s):
        for t in self.targets:
            t.write(s)
            t.flush()
    def flush(self):
        for t in self.targets:
            t.flush()

# ============================================================
# ========== DISCRIMINATOR ARCHITECTURE ======================
# ============================================================

class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.k = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.v = nn.Conv1d(channels, channels,       1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, T = x.shape
        q = self.q(x).view(B, C // 8, T)   # [B, C/8, T]
        k = self.k(x).view(B, C // 8, T)   # [B, C/8, T]
        v = self.v(x).view(B, C, T)        # [B, C,   T]

        # scaled dot-product attention over time
        attn = torch.bmm(q.transpose(1, 2), k) / math.sqrt(C // 8 + 1e-6)  # [B, T, T]
        attn = F.softmax(attn, dim=-1)

        o = torch.bmm(v, attn.transpose(1, 2))  # [B, C, T]
        return self.gamma * o + x


class Discriminator1D(nn.Module):
    def __init__(self, in_ch=10, base=64):
        super().__init__()
        # All convs are spectral_norm wrapped, causal-ish receptive field growth
        self.conv1 = spectral_norm(nn.Conv1d(in_ch, base, 3, padding=1))               # keep L
        self.conv2 = spectral_norm(nn.Conv1d(base, base, 3, padding=2, dilation=2))    # keep L
        self.attn  = SelfAttention1D(base)
        self.conv3 = spectral_norm(nn.Conv1d(base, base, 3, padding=4, dilation=4))    # keep L
        self.head  = spectral_norm(nn.Conv1d(base, 1, 1))

    def forward(self, x):
        # x: [B, in_ch, L]  (L should be 256 )
        h = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        h = F.leaky_relu(self.conv2(h), 0.2, inplace=True)
        h = self.attn(h)
        h = F.leaky_relu(self.conv3(h), 0.2, inplace=True)
        score_map = self.head(h)           # [B, 1, L]
        score = score_map.mean(dim=-1)     # [B, 1]
        return score                       # keep shape (B,1)


def load_discriminator(discriminator_path: str):
    D = Discriminator1D(in_ch=len(FEATURE_ORDER), base=64)

    state = torch.load(discriminator_path, map_location="cpu")
    D.load_state_dict(state)
    D.eval()
    print("[INFO] SAGAN Discriminator loaded.")
    return D


# ============================================================
# ========== FEATURE ENGINEERING / WINDOW PREP ===============
# ============================================================

def engineer_tick_features(df_ticks_window: pd.DataFrame,
                           roll_vol_window: int = 64,
                           eps: float = 1e-6) -> pd.DataFrame:
    """
    Build the 10 SAGAN discriminator features from your live tick data.
    Your tick file columns are:
        'Gmt time', 'Ask', 'Bid', 'AskVolume', 'BidVolume'
    We will rename them internally and generate the exact 10 features:
        mid_rel, microprice_rel,
        spread_z, imbalance_z,
        d_mid_z, d_spread_z, d_imb_z,
        rolling_vol_z,
        bid_volume_log1p_z, ask_volume_log1p_z
    """

    # Make a working copy
    out = df_ticks_window.copy().reset_index(drop=True)

    # Standardize column names 
    ask_col = "Ask"
    bid_col = "Bid"
    askvol_col = "AskVolume"
    bidvol_col = "BidVolume"

    # 1. Core microstructure values
    out["mid"] = (out[ask_col] + out[bid_col]) / 2.0
    out["spread"] = out[ask_col] - out[bid_col]

    out["imbalance"] = (out[bidvol_col] - out[askvol_col]) / (
        out[bidvol_col] + out[askvol_col] + eps
    )

    # "microprice" a.k.a weighted mid depending on which side is more aggressive
    out["microprice"] = (
        out[ask_col] * out[bidvol_col] + out[bid_col] * out[askvol_col]
    ) / (out[bidvol_col] + out[askvol_col] + eps)

    # 2. Tick-to-tick deltas
    out["d_mid"] = out["mid"].diff().fillna(0.0)
    out["d_spread"] = out["spread"].diff().fillna(0.0)
    out["d_imb"] = out["imbalance"].diff().fillna(0.0)

    # 3. Rolling vol of d_mid
    out["rolling_vol"] = (
        out["d_mid"]
        .rolling(roll_vol_window)
        .std(ddof=1)
        .fillna(0.0)
    )

    # 4. Log volumes
    out["bid_volume_log1p"] = np.log1p(out[bidvol_col])
    out["ask_volume_log1p"] = np.log1p(out[askvol_col])

    # 5. "Session-like" normalization over this 256-tick slice
    def zscore(col):
        mu = out[col].mean()
        sd = out[col].std()
        if (sd is None) or (not np.isfinite(sd)) or (sd < 1e-9):
            sd = 1e-9
        return (out[col] - mu) / sd

    # anchor everything to local mid median
    mid_median = out["mid"].median()
    if not np.isfinite(mid_median):
        mid_median = out["mid"].iloc[-1]

    # relative features
    out["mid_rel"] = out["mid"] / (mid_median + eps) - 1.0
    out["microprice_rel"] = out["microprice"] / (mid_median + eps) - 1.0

    # z-scored levels
    out["spread_z"] = zscore("spread")
    out["imbalance_z"] = zscore("imbalance")

    # z-scored deltas
    out["d_mid_z"] = zscore("d_mid")
    out["d_spread_z"] = zscore("d_spread")
    out["d_imb_z"] = zscore("d_imb")

    # z-scored rolling vol
    out["rolling_vol_z"] = zscore("rolling_vol")

    # z-scored log volumes
    out["bid_volume_log1p_z"] = zscore("bid_volume_log1p")
    out["ask_volume_log1p_z"] = zscore("ask_volume_log1p")

    return out

def prepare_window_for_discriminator(df_ticks_full: pd.DataFrame,
                                     L: int = 256):
    """
    - Take last L ticks
    - Compute engineered features
    - Stack in (1, F, L) using FEATURE_ORDER
    """
    if len(df_ticks_full) < L:
        raise ValueError(f"Need at least {L} ticks, only have {len(df_ticks_full)}")

    df_last = df_ticks_full.tail(L).copy().reset_index(drop=True)
    feats_full = engineer_tick_features(df_last)

    for f in FEATURE_ORDER:
        if f not in feats_full.columns:
            raise KeyError(f"Feature '{f}' not found. FEATURE_ORDER must match training channels exactly.")

    arr = feats_full[FEATURE_ORDER].to_numpy().T.astype(np.float32)  # (F,L)
    x_tensor = torch.from_numpy(arr).unsqueeze(0)                     # (1,F,L)

    # used later for plotting and entry price suggestion
    return x_tensor, feats_full, df_last


# ============================================================
# ========== MICROSTRUCTURE SCORING ==========================
# ============================================================

def classify_risk_state(disc_score: float,
                        mu_d: float,
                        sigma_d: float,
                        z_thresh: float = 2.0) -> str:
    """
    If the live discriminator score is way below the typical "real" score,
    we call it HIGH_RISK_OOD (unsafe regime).
    """
    cutoff = mu_d - z_thresh * sigma_d
    return "HIGH_RISK_OOD" if disc_score < cutoff else "NORMAL"


def derive_micro_signal(feats_full: pd.DataFrame,
                        tail_len: int = 32,
                        imb_th: float = IMBALANCE_TH):
    """
    Check last N ticks inside the 256-window:
    - imbalance_z mean tells who is hitting harder (bids or asks)
    - d_mid_z sum tells direction of recent price change
    """
    tail = feats_full.tail(tail_len).copy()

    imb_mean = tail["imbalance_z"].mean()
    dmid_sum = tail["d_mid_z"].sum()

    if imb_mean > imb_th and dmid_sum > 0:
        return "BUY", "strong bid imbalance, mid drifting up"
    if imb_mean < -imb_th and dmid_sum < 0:
        return "SELL", "ask pressure, mid drifting down"
    return "NEUTRAL", "no clear short-term dominance"


def run_micro_block(D: torch.nn.Module,
                    df_ticks_live: pd.DataFrame,
                    decision_idx: int | None = None):
    """
    Returns dict with:
        risk_state, micro_signal, pressure_notes,
        disc_score, ref_mid, ticks_window, idx_decision
    decision_idx:
        - if provided: where in the 256-tick window we pretend to 'make the call'
        - if None: default to last tick (live mode)
    """
    x_tensor, feats_full, df_last = prepare_window_for_discriminator(df_ticks_live, L=256)

    with torch.no_grad():
        disc_raw = D(x_tensor.float())
        disc_score = float(disc_raw.view(-1).item())

    risk_state = classify_risk_state(disc_score, MU_D, SIGMA_D)
    msig, note = derive_micro_signal(feats_full)

    # ref_mid: we’ll take the price at the decision index specifically, not always last
    if decision_idx is None:
        # live mode = last tick
        decision_idx = len(df_last) - 1
    # clamp to valid
    decision_idx = max(0, min(decision_idx, len(df_last) - 1))

    if "Bid" in df_last.columns and "Ask" in df_last.columns:
        ref_mid = (df_last["Bid"].iloc[decision_idx] + df_last["Ask"].iloc[decision_idx]) / 2.0
    elif "bid" in df_last.columns and "ask" in df_last.columns:
        ref_mid = (df_last["bid"].iloc[decision_idx] + df_last["ask"].iloc[decision_idx]) / 2.0
    else:
        raise KeyError(
            f"Could not find Bid/Ask columns in df_last. Columns present: {list(df_last.columns)}"
        )

    return {
        "risk_state":      risk_state,
        "micro_signal":    msig,
        "pressure_notes":  note,
        "disc_score":      disc_score,
        "ref_mid":         ref_mid,
        "ticks_window":    df_last,
        "idx_decision":    decision_idx,
    }



# ============================================================
# ========== MACRO MODULE (PROPHET + TCNs) ===================
# ============================================================

def load_macro_models():
    """
    Load Prophet, both TCN models, and the scaler.
    """
    models = {}

    if not os.path.exists(PROPHET_PATH):
        print("[WARN] Prophet model not found:", PROPHET_PATH)
        models["prophet"] = None
    else:
        models["prophet"] = joblib.load(PROPHET_PATH)

    if tf is None:
        print("[WARN] TensorFlow not available. TCN models will be None.")
        models["tcn_v1"] = None
        models["tcn_v2"] = None
    else:
        if os.path.exists(TCN_V1_PATH):
            models["tcn_v1"] = tf.keras.models.load_model(
                TCN_V1_PATH,
                custom_objects=CUSTOM_OBJECTS_TCN,
                compile=False
            )
        else:
            print("[WARN] TCN v1 model not found:", TCN_V1_PATH)
            models["tcn_v1"] = None

        if os.path.exists(TCN_V2_PATH):
            models["tcn_v2"] = tf.keras.models.load_model(
                TCN_V2_PATH,
                custom_objects=CUSTOM_OBJECTS_TCN,
                compile=False
            )
        else:
            print("[WARN] TCN v2 model not found:", TCN_V2_PATH)
            models["tcn_v2"] = None

    if not os.path.exists(SCALER_PATH):
        print("[WARN] Scaler not found:", SCALER_PATH)
        models["scaler"] = None
    else:
        models["scaler"] = joblib.load(SCALER_PATH)

    return models


def make_tcn_input(df_prices: pd.DataFrame,
                   scaler,
                   window_size: int = 20):
    """
    Build (1, window_size, num_features) for the TCN.

    scaler can be:
    - a real sklearn scaler with .transform()   
    - None or something else                  
    """

    if len(df_prices) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows of daily data for TCN "
            f"(got {len(df_prices)})."
        )

    tail = df_prices.tail(window_size).copy()

    # IMPORTANT: update this list to match scaler.feature_names_in_ if different
    feature_cols_in_training = [
        "Open",
        "High",
        "Low",
        "Close",
        "Change_%",  
    ]

    missing_cols = [c for c in feature_cols_in_training if c not in tail.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for TCN input: {missing_cols}")

    last_close = float(tail["Close"].iloc[-1])

    X_seq = tail[feature_cols_in_training].to_numpy(dtype=np.float32)  # shape (20, num_features)

    # Try to scale if it's an sklearn scaler
    X_proc = None
    if scaler is not None and hasattr(scaler, "transform"):
        # standard / minmax scaler path
        X_proc = scaler.transform(X_seq)  # (20, num_features)
    else:
        # fallback raw features (model was trained on already-normalized inputs or you don't have scaler)
        print("[WARN] No valid scaler.transform() found. Using raw features for TCN input.")
        X_proc = X_seq

    X_proc = np.expand_dims(X_proc, axis=0)  # (1,20,num_features)
    return X_proc, last_close




def predict_tcn_close_usd(model,
                          X_scaled_or_raw,
                          scaler,
                          close_col_name="Close"):
    """
    Run TCN to get next-close prediction.
    If we have a proper sklearn scaler with data_min_/data_max_,
    we'll inverse scale.
    Otherwise we just return the raw model output.
    """

    if model is None:
        return np.nan

    y_scaled = model.predict(X_scaled_or_raw, verbose=0)  # (1,1) or similar
    y_scaled_val = float(y_scaled.flatten()[0])

    # try inverse scaling if scaler looks like sklearn
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        try:
            close_idx = list(scaler.feature_names_in_).index(close_col_name)
        except Exception:
            close_idx = None

        if close_idx is not None:
            data_min = scaler.data_min_[close_idx]
            data_max = scaler.data_max_[close_idx]
            return y_scaled_val * (data_max - data_min) + data_min

    # fallback: return raw network output
    return y_scaled_val


def predict_prophet_next_close(prophet_model, df_prices: pd.DataFrame):
    """
    Prophet wants columns ds (datetime) and y (close).
    We'll build that from df_prices.
    """
    if prophet_model is None:
        return np.nan

    # Build Prophet-style history frame
    if "Date" in df_prices.columns:
        ds_col = pd.to_datetime(df_prices["Date"])
    else:
        ds_col = pd.to_datetime(df_prices.index)

    df_prophet_hist = pd.DataFrame({
        "ds": ds_col,
        "y":  df_prices["Close"].values
    })

    # Prophet generate 1-step ahead
    future = prophet_model.make_future_dataframe(periods=1, freq="D")
    forecast = prophet_model.predict(future)

    # last row of forecast is the next prediction
    return float(forecast["yhat"].iloc[-1])


def derive_macro_bias(last_close,
                      prophet_pred,
                      tcn1_pred,
                      tcn2_pred):
    """
    Turn raw price predictions into:
        macro_bias  = LONG / SHORT / NONE
        macro_conf  = HIGH / MEDIUM / LOW
        vote_detail = breakdown
    """
    def vote(pred):
        if np.isnan(pred):
            return None
        return "BUY" if pred > last_close else "SELL"

    v_prophet = vote(prophet_pred)
    v_tcn1    = vote(tcn1_pred)
    v_tcn2    = vote(tcn2_pred)

    votes = [v for v in [v_prophet, v_tcn1, v_tcn2] if v is not None]

    buy_votes  = votes.count("BUY")
    sell_votes = votes.count("SELL")

    if buy_votes >= 2:
        macro_bias = "LONG"
        macro_conf = "HIGH" if buy_votes == 3 else "MEDIUM"
    elif sell_votes >= 2:
        macro_bias = "SHORT"
        macro_conf = "HIGH" if sell_votes == 3 else "MEDIUM"
    else:
        macro_bias = "NONE"
        macro_conf = "LOW"

    detail = {
        "Prophet_vote": v_prophet,
        "TCN_v1_vote":  v_tcn1,
        "TCN_v2_vote":  v_tcn2,
        "buy_votes":    buy_votes,
        "sell_votes":   sell_votes,
        "last_close":   last_close,
        "prophet_pred": prophet_pred,
        "tcn1_pred":    tcn1_pred,
        "tcn2_pred":    tcn2_pred,
    }

    return macro_bias, macro_conf, detail


def run_macro_block():
    """
    Load models and daily data, then compute macro bias.
    """
    if not os.path.exists(DAILY_PATH):
        raise FileNotFoundError(f"Missing daily prices CSV at {DAILY_PATH}")

    df_daily = pd.read_csv(DAILY_PATH)

    # ensure datetime index or Date column exists
    if "Date" in df_daily.columns:
        df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce", utc=False)
        df_daily = df_daily.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        df_daily = df_daily.sort_values("Date").reset_index(drop=True)
    else:
        # assume already sorted by time
        pass

    models = load_macro_models()
    scaler = models.get("scaler")

    X_scaled, last_close = make_tcn_input(df_daily, scaler, window_size=20)
    tcn1_pred = predict_tcn_close_usd(models.get("tcn_v1"), X_scaled, scaler)
    tcn2_pred = predict_tcn_close_usd(models.get("tcn_v2"), X_scaled, scaler)
    prophet_pred = predict_prophet_next_close(models.get("prophet"), df_daily)

    macro_bias, macro_conf, detail = derive_macro_bias(
        last_close,
        prophet_pred,
        tcn1_pred,
        tcn2_pred
    )

    return {
        "macro_bias":       macro_bias,   # LONG / SHORT / NONE
        "macro_confidence": macro_conf,   # HIGH / MEDIUM / LOW
        "detail":           detail,       # breakdown of votes/preds
    }


# ============================================================
# ========== FUSION (MACRO + MICRO) ==========================
# ============================================================

def fuse_macro_micro(macro_bias, macro_conf, micro_dict):
    """
    Final action:
      - NO TRADE         (if regime is abnormal)
      - LONG ENTRY
      - SHORT ENTRY
      - WAIT
    """
    if micro_dict["risk_state"] == "HIGH_RISK_OOD":
        return {
            "action": "NO TRADE",
            "confidence": "NONE",
            "reason": "Market regime abnormal vs training baseline"
        }

    if macro_bias == "NONE":
        return {
            "action": "WAIT",
            "confidence": "LOW",
            "reason": "No reliable macro bias"
        }

    if macro_bias == "LONG" and micro_dict["micro_signal"] == "BUY":
        return {
            "action": "LONG ENTRY",
            "confidence": macro_conf,
            "reason": "Macro bullish + local buy pressure + normal regime"
        }

    if macro_bias == "SHORT" and micro_dict["micro_signal"] == "SELL":
        return {
            "action": "SHORT ENTRY",
            "confidence": macro_conf,
            "reason": "Macro bearish + local sell pressure + normal regime"
        }

    return {
        "action": "WAIT",
        "confidence": "LOW",
        "reason": "Macro bias and tape disagree"
    }


def suggest_levels(action, ref_mid,
                   risk_amt=RISK_USD_PER_OZ,
                   reward_amt=REWARD_USD_PER_OZ):
    """
    Convert final action into concrete trade levels.
    """
    if action == "LONG ENTRY":
        entry = ref_mid
        sl    = ref_mid - risk_amt
        tp    = ref_mid + reward_amt
        return entry, sl, tp

    if action == "SHORT ENTRY":
        entry = ref_mid
        sl    = ref_mid + risk_amt
        tp    = ref_mid - reward_amt
        return entry, sl, tp

    return None, None, None

# ============================================================
# ========== VISUALIZATION ===================================
# ============================================================

def plot_overlay(df_ticks_window,
                 action,
                 suggested_entry=None,
                 stop_loss=None,
                 take_profit=None,
                 idx_decision=None,
                 save_path=None,
                 show=True,
                 close=False):
    """
    Plot scalp plan:
      - mid price of last 256 ticks
      - colored marker at decision tick
      - optional entry/SL/TP lines
      - highlight what happened AFTER the call
    Supports Bid/Ask format and 'Gmt time' timestamp.
    """

    dfp = df_ticks_window.copy().reset_index(drop=True)

    # choose columns for bid/ask/time
    if "Bid" in dfp.columns and "Ask" in dfp.columns:
        bid_col = "Bid"
        ask_col = "Ask"
        time_col = "Gmt time" if "Gmt time" in dfp.columns else None
    elif "bid" in dfp.columns and "ask" in dfp.columns:
        bid_col = "bid"
        ask_col = "ask"
        time_col = "timestamp" if "timestamp" in dfp.columns else None
    else:
        raise KeyError(
            f"plot_overlay: Can't find bid/ask columns. Got {list(dfp.columns)}"
        )

    # compute mid
    dfp["mid"] = (dfp[bid_col] + dfp[ask_col]) / 2.0

    # x-axis
    if time_col and time_col in dfp.columns:
        x_axis = dfp[time_col]
    else:
        x_axis = np.arange(len(dfp))

    # default decision index: last point
    if idx_decision is None:
        idx_decision = len(dfp) - 1

    # make sure idx_decision is in range
    idx_decision = max(0, min(idx_decision, len(dfp) - 1))

    # split before/after decision
    before_x = x_axis[:idx_decision+1]
    before_y = dfp["mid"].iloc[:idx_decision+1]

    after_x  = x_axis[idx_decision:]
    after_y  = dfp["mid"].iloc[idx_decision:]

    # plot
    plt.figure(figsize=(10,5))

    # plot full mid lightly
    plt.plot(x_axis, dfp["mid"], alpha=0.3, label="mid (full 256)")

    # highlight 'after decision' segment a bit thicker
    plt.plot(after_x, after_y, linewidth=2.0, label="after call")

    # decision point marker colored by action
    last_mid = dfp["mid"].iloc[idx_decision]
    if action == "LONG ENTRY":
        color = "green"
    elif action == "SHORT ENTRY":
        color = "red"
    elif action == "WAIT":
        color = "gray"
    else:
        color = "black"

    plt.scatter(
        [after_x.iloc[0] if hasattr(after_x, "iloc") else after_x[0]],
        [last_mid],
        s=80,
        color=color,
        zorder=5,
        label=f"call: {action}"
    )

    # horizontal trade levels if we actually entered
    if suggested_entry is not None:
        plt.axhline(suggested_entry, linestyle="--", label=f"Entry {suggested_entry:.2f}")
    if stop_loss is not None:
        plt.axhline(stop_loss, linestyle="--", label=f"SL {stop_loss:.2f}")
    if take_profit is not None:
        plt.axhline(take_profit, linestyle="--", label=f"TP {take_profit:.2f}")

    plt.title("Decision vs Aftermath (last 256 ticks)")
    plt.xlabel("time" if time_col else "tick index")
    plt.ylabel("mid price")
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Figure saved to: {save_path}")

    if show:
        plt.show()

    if close and not show:
        plt.close()

# ============================================================
# ========== MAIN ============================================
# ============================================================

def load_recent_ticks(csv_path: str) -> pd.DataFrame:
    """
    Load the most recent tick data for microstructure analysis.
    Expects columns like:
        'Gmt time', 'Ask', 'Bid', 'AskVolume', 'BidVolume'

    Returns a DataFrame sorted by time (ascending).
    We'll let later logic slice the last 256 ticks.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing tick data CSV at {csv_path}")

    df_ticks = pd.read_csv(csv_path)

    # try to parse the time column 
    if "Gmt time" in df_ticks.columns:
        # Format looks like '04.08.2025 00:00:00.098'
        try:
            df_ticks["Gmt time"] = pd.to_datetime(
                df_ticks["Gmt time"],
                format="%d.%m.%Y %H:%M:%S.%f",
                errors="coerce",
                utc=False
            )
        except Exception:
            # silently ignore parse failure
            pass

        # sort by parsed time if we could parse, else just leave ordering
        if pd.api.types.is_datetime64_any_dtype(df_ticks["Gmt time"]):
            df_ticks = df_ticks.sort_values("Gmt time").reset_index(drop=True)

    return df_ticks

def filter_to_ny_session(df_ticks: pd.DataFrame,
                         session_start_utc="13:30:00",
                         session_end_utc="22:00:00") -> pd.DataFrame:
    """
    Keep only rows whose timestamp falls in NY session hours.
    Assumes df_ticks['Gmt time'] is already parsed to pandas datetime
    (UTC-like) in load_recent_ticks() and sorted ascending.

    session_start_utc, session_end_utc are strings HH:MM:SS.
    """
    if "Gmt time" not in df_ticks.columns:
        # If we can't filter by time, just return original.
        return df_ticks

    # ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_ticks["Gmt time"]):
        # try again in case it wasn't parsed
        df_ticks = df_ticks.copy()
        df_ticks["Gmt time"] = pd.to_datetime(
            df_ticks["Gmt time"],
            format="%d.%m.%Y %H:%M:%S.%f",
            errors="coerce",
            utc=False
        )

    # drop rows we couldn't parse
    df_ticks = df_ticks.dropna(subset=["Gmt time"]).reset_index(drop=True)

    # Extract just the time-of-day as string HH:MM:SS for filter
    times_str = df_ticks["Gmt time"].dt.strftime("%H:%M:%S")

    mask = (times_str >= session_start_utc) & (times_str <= session_end_utc)
    df_ny = df_ticks.loc[mask].copy().reset_index(drop=True)

    # If for some reason we got <256 ticks (too few), fallback to full df
    if len(df_ny) < 256:
        return df_ticks
    else:
        return df_ny


def main():
    # Create a timestamped run directory and tee console to log file
    run_dir = ensure_run_dir("gold_pipeline")
    log_path = run_dir / "summary.log"
    figs_dir = run_dir / "figs"

    # Tee stdout and stderr so existing prints go to both console and file
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "a", buffering=1) as lf:
        sys.stdout = Tee(_orig_stdout, lf)
        sys.stderr = Tee(_orig_stderr, lf)
        try:
            # 1. MACRO BLOCK (Prophet + TCNs on daily data)
            macro_out = run_macro_block()

            # 2. MICRO BLOCK (SAGAN Discriminator)
            D = load_discriminator(DISCRIMINATOR_PATH)
            df_ticks_live = load_recent_ticks(TICK_SOURCE_PATH)
            df_ticks_live_ny = filter_to_ny_session(df_ticks_live)
            DECISION_IDX_FOR_REVIEW = 200  # tweak this number

            micro_out = run_micro_block(
                D,
                df_ticks_live_ny,
                decision_idx=DECISION_IDX_FOR_REVIEW
            )

            # 3. FUSE macro + micro
            fused = fuse_macro_micro(
                macro_bias = macro_out["macro_bias"],
                macro_conf = macro_out["macro_confidence"],
                micro_dict = {
                    "risk_state":    micro_out["risk_state"],
                    "micro_signal":  micro_out["micro_signal"],
                },
            )

            # 4. TRADE LEVELS 
            if fused["action"] == "LONG ENTRY":
                suggested_entry = micro_out["ref_mid"]
                stop_loss       = micro_out["ref_mid"] - 0.8
                take_profit     = micro_out["ref_mid"] + 1.0
            elif fused["action"] == "SHORT ENTRY":
                suggested_entry = micro_out["ref_mid"]
                stop_loss       = micro_out["ref_mid"] + 0.8
                take_profit     = micro_out["ref_mid"] - 1.0
            else:
                suggested_entry = None
                stop_loss       = None
                take_profit     = None

            # 5. PRINT SUMMARY 
            print("=== MACRO MODULE ===")
            print(f"macro_bias: {macro_out['macro_bias']}")
            print(f"macro_confidence: {macro_out['macro_confidence']}")
            print(f"detail: {macro_out['detail']}")

            print("\n=== MICRO MODULE ===")
            print(f"risk_state: {micro_out['risk_state']}")
            print(f"micro_signal: {micro_out['micro_signal']}")
            print(f"pressure_notes: {micro_out['pressure_notes']}")
            print(f"disc_score: {micro_out['disc_score']}")

            print("\n=== FINAL DECISION ===")
            print(f"action: {fused['action']}")
            print(f"confidence: {fused['confidence']}")
            print(f"reason: {fused['reason']}")
            print(f"\nSuggested Entry: {suggested_entry}")
            print(f"Stop Loss: {stop_loss}")
            print(f"Take Profit: {take_profit}")

            # 6. PLOT OVERLAY — save figure to run folder for review
            fig_name = f"overlay_{fused['action'].replace(' ', '_').lower()}.png"
            save_path = figs_dir / fig_name
            plot_overlay(
                df_ticks_window = micro_out["ticks_window"],
                action          = fused["action"],
                suggested_entry = suggested_entry,
                stop_loss       = stop_loss,
                take_profit     = take_profit,
                idx_decision    = micro_out.get("idx_decision", None),
                save_path       = save_path,
                show            = True,
                close           = False,
            )

            # 7. JSON summary
            summary = {
                "macro": macro_out,
                "micro": {
                    "risk_state":     micro_out["risk_state"],
                    "micro_signal":   micro_out["micro_signal"],
                    "pressure_notes": micro_out["pressure_notes"],
                    "disc_score":     float(micro_out["disc_score"]),
                    "ref_mid":        float(micro_out["ref_mid"]),
                },
                "decision": fused,
                "levels": {
                    "entry":      float(suggested_entry) if suggested_entry is not None else None,
                    "stop_loss":  float(stop_loss) if stop_loss is not None else None,
                    "take_profit":float(take_profit) if take_profit is not None else None,
                },
                "artifacts": {
                    "summary_log": str(log_path),
                    "figure":      str(save_path),
                },
            }
            with open(run_dir / "summary.json", "w") as jf:
                json.dump(summary, jf, indent=2)
            print(f"[INFO] JSON summary saved to: {run_dir / 'summary.json'}")

        finally:
            # restore original streams even if something fails
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr

if __name__ == "__main__":
    main()
