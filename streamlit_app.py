"""
Step 4 — Aeromonas Growth Predictor (v6)
Run: streamlit run step4_streamlit_app.py

Changes from v5:
- App title/subtitle restored (visible header)
- Section order: Growth Rate Curve FIRST, then threshold
- Threshold visualization: Option A timeline dots (no overlap)
- Title/chart spacing fixed throughout
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

st.set_page_config(
    page_title="Aeromonas Growth Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-baseweb="tab-list"] button {
    font-size: 20px !important; font-weight: 700 !important;
    padding: 12px 24px !important;
}
h2 { font-size: 26px !important; font-weight: 700 !important; }
h3 { font-size: 20px !important; font-weight: 600 !important; }
.stCaption { color: #a0aec0 !important; font-size: 14px !important; }
div[data-testid="metric-container"] label {
    font-size: 15px !important; color: #a0aec0 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 34px !important; font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0e1117"
PLOT_BG  = "#1a1f2e"
GRID_CLR = "#2d3748"
TEXT_CLR = "#e2e8f0"
MUTED    = "#718096"
COL_GPR  = "#63b3ed"
COL_TSB  = "#63b3ed"
COL_SJ   = "#68d391"
COL_RAT  = "#a0aec0"
COL_TRAIN= "#f6e05e"
COL_POS  = "#fc8181"
COL_NEG  = "#63b3ed"
COL_WARN = "#f6ad55"
COL_SAFE = "#68d391"
COL_CRIT = "#fc8181"

FS_TICK=14; FS_TITLE=16; FS_CHART=17; FS_ANNOT=13; FS_LEG=13; FS_BAR=13; FS_KEY=15

_FONT = dict(color=TEXT_CLR, family="sans-serif", size=FS_TICK)
PLOTLY_LAYOUT = dict(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)

def _xax(title="", **kw):
    return dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR,
                tickfont=dict(color=TEXT_CLR, size=FS_TICK),
                title=dict(text=title, font=dict(color=TEXT_CLR, size=FS_TITLE)), **kw)

def _yax(title="", **kw):
    bt = kw.pop("tickfont", dict(color=TEXT_CLR, size=FS_TICK))
    return dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, tickfont=bt,
                title=dict(text=title, font=dict(color=TEXT_CLR, size=FS_TITLE)), **kw)

def _legend(**kw):
    b = dict(bgcolor="rgba(26,31,46,0.85)", bordercolor=GRID_CLR,
             borderwidth=1, font=dict(color=TEXT_CLR, size=FS_LEG))
    b.update(kw); return b

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH = Path("outputs/aeromonas_kinetics_master.csv")
FEATURES  = ["temperature_C","medium_enc","NaCl_pct","PCS_conc_pct","pcs_enc"]
FEAT_SHORT= ["Temp (°C)","Medium","NaCl (%)","PCS Conc (%)","PCS Type"]

TRAIN_TEMP_MIN, TRAIN_TEMP_MAX = 4.0, 15.0
TRAIN_NACL_MAX=3.0; TRAIN_PCS_MAX=0.13
TEMP_MIN,TEMP_MAX=1.0,20.0
NACL_MIN,NACL_MAX=0.0,4.5
PCS_MIN,PCS_MAX=0.0,0.15

DEFAULT_INITIAL_LOG=2.0; DEFAULT_LIMIT_LOG=5.0; LAG_MEDIAN_H=5.0


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading models (~10 s)…")
def load_all():
    df = pd.read_csv(DATA_PATH)
    df = df[df["mu_max"].notna()].copy()
    le_m=LabelEncoder(); le_p=LabelEncoder(); le_s=LabelEncoder()
    df["medium_enc"]  = le_m.fit_transform(df["medium"])
    df["pcs_enc"]     = le_p.fit_transform(df["PCS_type"].fillna("None"))
    df["species_enc"] = le_s.fit_transform(df["species"])
    X=df[FEATURES].values; y=df["mu_max"].values

    rf=RandomForestRegressor(n_estimators=500,max_depth=4,min_samples_leaf=3,
                             max_features="sqrt",random_state=42,n_jobs=-1)
    rf.fit(X,y)

    gpr_k=(C(1.0,(1e-3,1e2))*Matern(length_scale=1.0,length_scale_bounds=(0.1,10.0),nu=1.5)
           +WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-5,1.0)))
    sc=StandardScaler()
    gpr=GaussianProcessRegressor(kernel=gpr_k,n_restarts_optimizer=5,
                                 normalize_y=True,random_state=42)
    gpr.fit(sc.fit_transform(X),y)

    exp=shap.TreeExplainer(rf)
    sv_train=exp.shap_values(X)
    base_val=(float(exp.expected_value[0]) if isinstance(exp.expected_value,np.ndarray)
              else float(exp.expected_value))

    df_rat=df[(df["medium"]=="TSB")&(df["experiment"]=="Temperature")
              &df["mu_max"].notna()&df["sqrt_mu_max"].notna()]
    try:
        def _r(T,b,Tmin): return b*np.maximum(T-Tmin,0)
        popt,_=curve_fit(_r,df_rat["temperature_C"].values,
                         df_rat["sqrt_mu_max"].values,p0=[0.03,-5],maxfev=5000)
        rat_params=popt
    except: rat_params=None

    bench={"Ratkowsky":0.612,"RandomForest":0.801,"XGBoost":0.789,"GPR":0.892}
    shap_imp=pd.DataFrame({"Feature":FEAT_SHORT,
                           "Mean_Abs_SHAP":np.abs(sv_train).mean(axis=0)
                          }).sort_values("Mean_Abs_SHAP",ascending=False).reset_index(drop=True)
    return dict(rf=rf,gpr=gpr,sc=sc,exp=exp,sv_train=sv_train,base_val=base_val,
                le_m=le_m,le_p=le_p,le_s=le_s,X=X,y=y,df=df,
                rat_params=rat_params,bench=bench,shap_imp=shap_imp,
                med_enc_tsb=int(le_m.transform(["TSB"])[0]),
                med_enc_sj=int(le_m.transform(["SJ"])[0]),
                pcs_enc_none=int(le_p.transform(["None"])[0]))


# ── Helpers ────────────────────────────────────────────────────────────────────
def enc_x(temp,med_enc,nacl,pcs_c,pcs_ev):
    return np.array([[temp,med_enc,nacl,pcs_c,pcs_ev]],dtype=float)

def predict_pt(x,m):
    rf_p=float(m["rf"].predict(x)[0])
    mu,sd=m["gpr"].predict(m["sc"].transform(x),return_std=True)
    return rf_p,float(mu[0]),float(sd[0]),max(0.0,float(mu[0])-1.96*float(sd[0])),float(mu[0])+1.96*float(sd[0])

def sweep_temp(m,med_enc,nacl,pcs_c,pcs_ev,n=100):
    ts=np.linspace(TEMP_MIN,TEMP_MAX,n)
    Xw=np.column_stack([ts,np.full(n,med_enc),np.full(n,nacl),
                        np.full(n,pcs_c),np.full(n,pcs_ev)])
    mu,sd=m["gpr"].predict(m["sc"].transform(Xw),return_std=True)
    return ts,mu,sd

def time_to_threshold(mu,ci_lo,ci_hi,log_initial,log_limit,lag_h):
    delta_ln=(log_limit-log_initial)*np.log(10)
    def _t(r): return np.inf if r<=0 else lag_h+delta_ln/r
    return _t(mu),_t(max(ci_hi,1e-9)),_t(max(ci_lo,1e-9))

def fmt_time(h):
    if np.isinf(h): return "∞"
    return f"{h:.0f} h  ({h/24:.1f} d)"


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_growth_curve(m,temp,medium,nacl,pcs_c,pcs_ev,show_sj,compact=False):
    med_enc=m["med_enc_tsb"] if medium=="TSB" else m["med_enc_sj"]
    ts,mu,sd=sweep_temp(m,med_enc,nacl,pcs_c,pcs_ev)
    ci_lo=np.maximum(0,mu-1.96*sd); ci_hi=mu+1.96*sd

    fig=go.Figure()

    for x0,x1,pos in [(TEMP_MIN,TRAIN_TEMP_MIN,"top left"),
                      (TRAIN_TEMP_MAX,TEMP_MAX,"top right")]:
        fig.add_vrect(x0=x0,x1=x1,fillcolor=COL_TRAIN,opacity=0.07,line_width=0,
                      annotation_text="extrapolation",annotation_position=pos,
                      annotation_font=dict(color=COL_TRAIN,size=12))
    for tv in [TRAIN_TEMP_MIN,TRAIN_TEMP_MAX]:
        fig.add_vline(x=tv,line_dash="dot",line_color=MUTED,line_width=1.2,opacity=0.6)

    if show_sj and medium=="SJ":
        ts2,mu2,sd2=sweep_temp(m,m["med_enc_tsb"],0.0,0.0,m["pcs_enc_none"])
        fig.add_trace(go.Scatter(
            x=np.concatenate([ts2,ts2[::-1]]),
            y=np.concatenate([np.maximum(0,mu2+1.96*sd2),np.maximum(0,mu2-1.96*sd2)[::-1]]),
            fill="toself",fillcolor="rgba(99,179,237,0.10)",
            line=dict(width=0),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ts2,y=mu2,mode="lines",
            line=dict(color=COL_TSB,width=1.8,dash="dash"),name="TSB (comparison)",
            hovertemplate="TSB  T=%{x:.1f}°C  μmax=%{y:.4f} h⁻¹<extra></extra>"))

    if m["rat_params"] is not None and medium=="TSB":
        def _r(T,b,Tmin): return (b*np.maximum(T-Tmin,0))**2
        fig.add_trace(go.Scatter(x=ts,y=_r(ts,*m["rat_params"]),mode="lines",
            line=dict(color=COL_RAT,width=1.5,dash="dashdot"),name="Ratkowsky",
            hovertemplate="Ratkowsky  T=%{x:.1f}°C  μmax=%{y:.4f}<extra></extra>"))

    col_m=COL_TSB if medium=="TSB" else COL_SJ
    fill_rgb="99,179,237" if medium=="TSB" else "104,211,145"
    fig.add_trace(go.Scatter(
        x=np.concatenate([ts,ts[::-1]]),
        y=np.concatenate([ci_hi,ci_lo[::-1]]),
        fill="toself",fillcolor=f"rgba({fill_rgb},0.18)",
        line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=ts,y=mu,mode="lines",line=dict(color=col_m,width=2.5),
        name=f"GPR mean ({medium})",
        customdata=np.stack([np.maximum(0,mu-1.96*sd),mu+1.96*sd,sd],axis=-1),
        hovertemplate=(f"<b>{medium}</b>  T=%{{x:.1f}}°C<br>"
                       "μmax = <b>%{y:.4f}</b> h⁻¹<br>"
                       "95% CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>")))

    df=m["df"]
    mask=df["medium"]==medium
    if medium=="TSB":
        mask &= (np.abs(df["NaCl_pct"]-nacl)<=max(0.3,nacl*0.3))
        mask &= (np.abs(df["PCS_conc_pct"]-pcs_c)<=max(0.01,pcs_c*0.3))
    tr=df[mask & df["mu_max"].notna()]
    if len(tr)>0:
        fig.add_trace(go.Scatter(
            x=tr["temperature_C"],y=tr["mu_max"],mode="markers",
            marker=dict(color=col_m,size=8,opacity=0.7,line=dict(color=BG,width=1.5)),
            name="Training data",
            customdata=tr[["strain","NaCl_pct","PCS_conc_pct"]].values,
            hovertemplate=("<b>Training point</b>  Strain: %{customdata[0]}<br>"
                           "T=%{x}°C  μmax=%{y:.4f} h⁻¹<extra></extra>")))

    x0_=enc_x(temp,med_enc,nacl,pcs_c,pcs_ev)
    rf_p,gpr_mu,gpr_sd,pt_lo,pt_hi=predict_pt(x0_,m)
    fig.add_trace(go.Scatter(
        x=[temp],y=[gpr_mu],mode="markers",
        marker=dict(color=COL_WARN,size=20,symbol="star",line=dict(color=TEXT_CLR,width=2)),
        name=f"★ {gpr_mu:.4f} h⁻¹",
        customdata=[[rf_p,pt_lo,pt_hi]],
        hovertemplate=(f"<b>Current</b>  T={temp:.1f}°C  {medium}<br>"
                       "GPR μmax = <b>%{y:.4f}</b> h⁻¹<br>"
                       "RF = %{customdata[0]:.4f}<br>"
                       "95% CI [%{customdata[1]:.4f}, %{customdata[2]:.4f}]<extra></extra>")))
    fig.add_vline(x=temp,line_dash="dash",line_color=COL_WARN,line_width=1.5,opacity=0.7)

    h=300 if compact else 400
    fig.update_layout(
        **PLOTLY_LAYOUT,font=_FONT,height=h,
        title=dict(text=f"① Growth Rate Curve — μmax vs Temperature ({medium})",
                   font=dict(color=TEXT_CLR,size=FS_CHART),x=0.01),
        xaxis=_xax(title="Temperature (°C)",range=[TEMP_MIN,TEMP_MAX]),
        yaxis=_yax(title="μmax (h⁻¹)",rangemode="tozero"),
        hovermode="x unified",
        legend=_legend(orientation="h",yanchor="bottom",y=1.02,
                       xanchor="left",x=0,font=dict(color=TEXT_CLR,size=FS_LEG)),
        margin=dict(l=65,r=30,t=65,b=55),
    )
    return fig,rf_p,gpr_mu,gpr_sd,pt_lo,pt_hi


def fig_timeline_dots(t_mean, t_fast, t_slow):
    """
    Option A timeline: three dots on a time axis.
    Labels placed above/below alternately to avoid overlap.
    Fast (red, bottom) — Mean (orange, top) — Slow (green, bottom)
    """
    # Cap extreme slow values for display axis
    t_slow_disp = t_slow if not np.isinf(t_slow) else t_fast * 4
    pad = (t_slow_disp - t_fast) * 0.18
    x_lo = max(0, t_fast - pad)
    x_hi = t_slow_disp + pad

    def fmt_short(h):
        if np.isinf(h): return "∞"
        d = h / 24
        if d >= 10:
            return f"{h:.0f} h<br>({d:.0f} d)"
        return f"{h:.0f} h<br>({d:.1f} d)"

    fig = go.Figure()

    # ── Track line ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_lo, x_hi], y=[0, 0], mode="lines",
        line=dict(color=MUTED, width=2), showlegend=False, hoverinfo="skip"))

    # Gradient fill: red zone (left) fading to green (right)
    n_fill = 60
    xs_fill = np.linspace(x_lo, x_hi, n_fill)
    for i in range(n_fill - 1):
        frac = i / (n_fill - 2)
        r = int(252 * (1 - frac) + 104 * frac)
        g = int(129 * (1 - frac) + 211 * frac)
        b_c = int(129 * (1 - frac) + 145 * frac)
        fig.add_shape(
            type="rect",
            x0=xs_fill[i], x1=xs_fill[i + 1],
            y0=-0.18, y1=0.18,
            fillcolor=f"rgba({r},{g},{b_c},0.18)",
            line_width=0)

    # ── Dots ────────────────────────────────────────────────────────────────
    # Fast (red) — label BELOW
    fig.add_trace(go.Scatter(
        x=[t_fast], y=[0], mode="markers",
        marker=dict(color=COL_CRIT, size=22, symbol="circle",
                    line=dict(color=TEXT_CLR, width=2.5)),
        name=f"Fast growth: {fmt_time(t_fast)}",
        hovertemplate=f"<b>Fast growth scenario</b><br>"
                      f"CI upper μmax<br>{fmt_time(t_fast)}<extra></extra>"))

    # Mean (orange) — label ABOVE
    fig.add_trace(go.Scatter(
        x=[t_mean], y=[0], mode="markers",
        marker=dict(color=COL_WARN, size=28, symbol="diamond",
                    line=dict(color=TEXT_CLR, width=2.5)),
        name=f"Mean: {fmt_time(t_mean)}",
        hovertemplate=f"<b>Mean scenario</b><br>"
                      f"GPR mean μmax<br>{fmt_time(t_mean)}<extra></extra>"))

    # Slow (green) — label BELOW
    if not np.isinf(t_slow):
        fig.add_trace(go.Scatter(
            x=[t_slow_disp], y=[0], mode="markers",
            marker=dict(color=COL_SAFE, size=22, symbol="circle",
                        line=dict(color=TEXT_CLR, width=2.5)),
            name=f"Slow growth: {fmt_time(t_slow)}",
            hovertemplate=f"<b>Slow growth scenario</b><br>"
                          f"CI lower μmax<br>{fmt_time(t_slow)}<extra></extra>"))

    # ── Annotations: alternating above/below ─────────────────────────────
    # Fast — below the dot
    fig.add_annotation(
        x=t_fast, y=-0.38,
        text=f"<b>{fmt_short(t_fast)}</b>",
        showarrow=True, arrowhead=0, arrowcolor=COL_CRIT,
        arrowwidth=1.5, ay=0, ax=0,
        font=dict(color=COL_CRIT, size=14),
        align="center", bgcolor="rgba(14,17,23,0.7)",
        bordercolor=COL_CRIT, borderwidth=1, borderpad=4)
    fig.add_annotation(
        x=t_fast, y=-0.72,
        text="⚠️ fastest",
        showarrow=False,
        font=dict(color=COL_CRIT, size=11),
        align="center")

    # Mean — above the dot
    fig.add_annotation(
        x=t_mean, y=0.38,
        text=f"<b>{fmt_short(t_mean)}</b>",
        showarrow=True, arrowhead=0, arrowcolor=COL_WARN,
        arrowwidth=1.5, ay=0, ax=0,
        font=dict(color=COL_WARN, size=14),
        align="center", bgcolor="rgba(14,17,23,0.7)",
        bordercolor=COL_WARN, borderwidth=1, borderpad=4)
    fig.add_annotation(
        x=t_mean, y=0.72,
        text="⏱ mean estimate",
        showarrow=False,
        font=dict(color=COL_WARN, size=11),
        align="center")

    # Slow — below the dot (offset label if close to mean)
    if not np.isinf(t_slow):
        fig.add_annotation(
            x=t_slow_disp, y=-0.38,
            text=f"<b>{fmt_short(t_slow)}</b>",
            showarrow=True, arrowhead=0, arrowcolor=COL_SAFE,
            arrowwidth=1.5, ay=0, ax=0,
            font=dict(color=COL_SAFE, size=14),
            align="center", bgcolor="rgba(14,17,23,0.7)",
            bordercolor=COL_SAFE, borderwidth=1, borderpad=4)
        fig.add_annotation(
            x=t_slow_disp, y=-0.72,
            text="✅ slowest",
            showarrow=False,
            font=dict(color=COL_SAFE, size=11),
            align="center")

    # Axis direction labels
    fig.add_annotation(x=x_lo + (x_hi-x_lo)*0.04, y=0,
        text="sooner →", showarrow=False,
        font=dict(color=MUTED, size=11), yshift=28)
    fig.add_annotation(x=x_hi - (x_hi-x_lo)*0.04, y=0,
        text="← later", showarrow=False,
        font=dict(color=MUTED, size=11), yshift=28)

    fig.update_layout(
        **PLOTLY_LAYOUT, font=_FONT,
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=_xax(title="Time (hours)", range=[x_lo, x_hi]),
        yaxis=dict(visible=False, range=[-1.1, 1.1],
                   gridcolor=GRID_CLR, zeroline=False),
        legend=_legend(orientation="h", yanchor="bottom", y=-0.22,
                       xanchor="center", x=0.5,
                       font=dict(color=TEXT_CLR, size=FS_LEG)),
        showlegend=True,
    )
    return fig


def fig_shap_waterfall(sv, bv, rf_p):
    order=np.argsort(np.abs(sv))[::-1]
    sv_o=sv[order]; fn_o=[FEAT_SHORT[i] for i in order]
    cum=[bv]
    for v in sv_o: cum.append(cum[-1]+v)
    max_abs=max(np.abs(sv_o).max(),1e-6)

    fig=go.Figure()
    for i,(v,fn) in enumerate(zip(sv_o,fn_o)):
        col=COL_POS if v>0 else COL_NEG
        fig.add_trace(go.Bar(
            x=[v],y=[fn],base=cum[i],orientation="h",
            marker=dict(color=col,opacity=0.85,line=dict(color=BG,width=0.8)),
            showlegend=False,
            hovertemplate=f"<b>{fn}</b><br>SHAP {v:+.5f} h⁻¹<br>"
                          f"{'↑ promotes growth' if v>0 else '↓ inhibits growth'}<extra></extra>"))
        off=max_abs*0.04
        fig.add_annotation(x=cum[i+1]+(off if v>=0 else -off),y=fn,
            text=f"<b>{v:+.5f}</b>",showarrow=False,
            xanchor="left" if v>=0 else "right",
            font=dict(color=TEXT_CLR,size=FS_KEY))

    fig.add_vline(x=bv,line_dash="dash",line_color=MUTED,line_width=1.5,
        annotation_text=f"baseline {bv:.4f}",annotation_position="top",
        annotation_font=dict(color=MUTED,size=FS_ANNOT))
    fig.add_vline(x=rf_p,line_dash="solid",line_color=TEXT_CLR,line_width=2.0,
        annotation_text=f"RF {rf_p:.4f}",annotation_position="bottom",
        annotation_font=dict(color=TEXT_CLR,size=FS_ANNOT))
    for col,lbl in [(COL_POS,"↑ Promotes growth"),(COL_NEG,"↓ Inhibits growth")]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
            marker=dict(color=col,size=12,symbol="square"),name=lbl))

    fig.update_layout(
        **PLOTLY_LAYOUT,font=_FONT,height=310,
        title=dict(text="SHAP Feature Contributions to μmax",
                   font=dict(color=TEXT_CLR,size=FS_CHART),x=0.01),
        xaxis=_xax(title="SHAP contribution (h⁻¹)"),
        yaxis=_yax(autorange="reversed",
                   tickfont=dict(color=TEXT_CLR,size=FS_KEY)),
        barmode="overlay",
        legend=_legend(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1),
        margin=dict(l=120,r=90,t=55,b=50))
    return fig


def fig_global_importance(sv_train):
    imp=np.abs(sv_train).mean(axis=0)
    order=np.argsort(imp)
    cols=[COL_POS if FEAT_SHORT[i]=="Temp (°C)" else COL_NEG for i in order]
    fig=go.Figure(go.Bar(
        x=imp[order],y=[FEAT_SHORT[i] for i in order],orientation="h",
        marker=dict(color=cols,opacity=0.85,line=dict(color=BG,width=0.8)),
        text=[f"{v:.5f}" for v in imp[order]],textposition="outside",
        textfont=dict(color=TEXT_CLR,size=FS_BAR),
        hovertemplate="<b>%{y}</b>  Mean |SHAP|=%{x:.5f} h⁻¹<extra></extra>"))
    fig.update_layout(
        **PLOTLY_LAYOUT,font=_FONT,height=270,
        title=dict(text="Global Feature Importance (n=118)",
                   font=dict(color=TEXT_CLR,size=FS_CHART),x=0.01),
        xaxis=_xax(title="Mean |SHAP| (h⁻¹)"),
        yaxis=_yax(tickfont=dict(color=TEXT_CLR,size=FS_KEY)),
        showlegend=False,margin=dict(l=120,r=90,t=55,b=45))
    return fig


def fig_about_benchmark(bench):
    order=sorted(bench.items(),key=lambda x:x[1])
    names=[k for k,v in order]; vals=[v for k,v in order]
    cols=[COL_RAT if k=="Ratkowsky" else(COL_GPR if k=="GPR" else COL_WARN) for k in names]
    fig=go.Figure(go.Bar(
        x=vals,y=names,orientation="h",
        marker=dict(color=cols,opacity=0.85,line=dict(color=BG,width=0.8)),
        text=[f"R²={v:.3f}" for v in vals],textposition="outside",
        textfont=dict(color=TEXT_CLR,size=FS_BAR),
        hovertemplate="<b>%{y}</b>  LOSO R²=%{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5,line_dash="dash",line_color=COL_WARN,line_width=1.5,opacity=0.6)
    fig.update_layout(
        **PLOTLY_LAYOUT,font=_FONT,height=310,
        title=dict(text="Fig 13 — ML vs Ratkowsky (LOSO)",
                   font=dict(color=TEXT_CLR,size=FS_CHART),x=0.01),
        xaxis=_xax(title="LOSO R²",range=[0,1.1]),
        yaxis=_yax(tickfont=dict(color=TEXT_CLR,size=FS_LEG)),
        showlegend=False,margin=dict(l=140,r=90,t=55,b=45))
    return fig


def fig_about_shap_bar(shap_imp):
    df_s=shap_imp.iloc[::-1]
    cols=[COL_POS if f=="Temp (°C)" else COL_NEG for f in df_s["Feature"]]
    fig=go.Figure(go.Bar(
        x=df_s["Mean_Abs_SHAP"],y=df_s["Feature"],orientation="h",
        marker=dict(color=cols,opacity=0.85,line=dict(color=BG,width=0.8)),
        text=[f"{v:.5f}" for v in df_s["Mean_Abs_SHAP"]],textposition="outside",
        textfont=dict(color=TEXT_CLR,size=FS_BAR),
        hovertemplate="<b>%{y}</b>  Mean |SHAP|=%{x:.5f}<extra></extra>"))
    fig.update_layout(
        **PLOTLY_LAYOUT,font=_FONT,height=310,
        title=dict(text="Fig 15 — Hurdle Factor Ranking",
                   font=dict(color=TEXT_CLR,size=FS_CHART),x=0.01),
        xaxis=_xax(title="Mean |SHAP| (h⁻¹)"),
        yaxis=_yax(tickfont=dict(color=TEXT_CLR,size=FS_LEG)),
        showlegend=False,margin=dict(l=140,r=100,t=55,b=45))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    m=load_all()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔧 Storage Conditions")
        medium=st.radio("Medium",["TSB","SJ"],horizontal=True,
            help="TSB = full hurdle data  |  SJ = temperature effect only")
        st.markdown("---")
        temp=st.slider("🌡 Temperature (°C)",TEMP_MIN,TEMP_MAX,8.0,0.5,
                       help=f"Training range: {TRAIN_TEMP_MIN}–{TRAIN_TEMP_MAX}°C")
        if medium=="TSB":
            nacl=st.slider("🧂 NaCl (%)",NACL_MIN,NACL_MAX,0.0,0.1)
            pcs_t=st.radio("🧪 PCS Type",["None","VTABB","JJT01"])
            pcs_c=(st.slider("PCS Concentration (%)",PCS_MIN,PCS_MAX,0.026,0.005)
                   if pcs_t!="None" else 0.0)
            if pcs_t=="None": st.caption("PCS Concentration: 0%")
        else:
            nacl=pcs_c=0.0; pcs_t="None"
            st.markdown("**🧂 NaCl** — 🔒 TSB only")
            st.markdown("**🧪 PCS** — 🔒 TSB only")
            st.warning("SJ mode — temperature effect only.\nμmax ~35% lower than TSB.")

        st.markdown("---")
        st.markdown("## 📊 Scenario Parameters")
        st.caption("TSB medium basis — research reference only")
        log_initial=st.slider("Initial level (log CFU/g)",0.0,4.0,DEFAULT_INITIAL_LOG,0.5)
        log_limit=st.slider("Safety threshold (log CFU/g)",3.0,7.0,DEFAULT_LIMIT_LOG,0.5)
        lag_mode=st.radio("Lag phase assumption",
                          ["Conservative (0 h)","Median (5 h)","Custom"])
        lag_h=(st.slider("Custom lag (h)",0.0,48.0,5.0,1.0) if lag_mode=="Custom"
               else (LAG_MEDIAN_H if "Median" in lag_mode else 0.0))
        st.caption(f"Δlog = {log_limit-log_initial:.1f}  |  lag = {lag_h:.0f} h")
        st.markdown("---")
        st.caption("LOSO R²:  GPR 0.892 ± 0.040  |  RF 0.865 ± 0.053")

    # ── Encode & predict ──────────────────────────────────────────────────────
    med_enc=m["med_enc_tsb"] if medium=="TSB" else m["med_enc_sj"]
    pcs_ev=int(m["le_p"].transform([pcs_t])[0])
    x0=enc_x(temp,med_enc,nacl,pcs_c,pcs_ev)
    rf_p,gpr_mu,gpr_sd,ci_lo,ci_hi=predict_pt(x0,m)
    t_mean,t_fast,t_slow=time_to_threshold(gpr_mu,ci_lo,ci_hi,log_initial,log_limit,lag_h)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_pred,tab_strain,tab_about=st.tabs(["📊  Prediction","🔬  Strain Data","ℹ️  About"])

    with tab_pred:

        # ── App header (restored) ──────────────────────────────────────────
        st.markdown("<div style='margin-top:1.5rem'></div>",unsafe_allow_html=True)
        st.title("🦠 Aeromonas Growth Predictor")
        st.markdown(
            "<p style='font-size:17px;color:#a0aec0;margin-top:-8px;'>"
            "Predict how storage conditions affect <em>Aeromonas</em> growth rate — "
            "GPR uncertainty bounds · SHAP explanations · Hurdle optimisation<br>"
            "<span style='font-size:14px;'>"
            "Lee et al. (2023) · TSB laboratory medium · Research purpose only · "
            "Not directly applicable to actual food shelf-life</span></p>",
            unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>",unsafe_allow_html=True)
        st.divider()

        # ═══════════════════════════════════════
        # ① GROWTH RATE CURVE (moved to top)
        # ═══════════════════════════════════════
        st.markdown("<div style='margin-top:1.2rem'></div>",unsafe_allow_html=True)
        sj_cmp=(st.checkbox("Show TSB curve for comparison",value=True)
                if medium=="SJ" else False)

        fig_gc,rf_p,gpr_mu,gpr_sd,ci_lo,ci_hi=fig_growth_curve(
            m,temp,medium,nacl,pcs_c,pcs_ev,sj_cmp,compact=False)
        st.plotly_chart(fig_gc,use_container_width=True)

        if temp<TRAIN_TEMP_MIN or temp>TRAIN_TEMP_MAX:
            st.caption(f"⚠️ {temp:.1f}°C is outside the training range "
                       f"({TRAIN_TEMP_MIN}–{TRAIN_TEMP_MAX}°C). "
                       "GPR CI widens automatically in extrapolation zones.")

        st.markdown("<div style='margin-top:1.5rem'></div>",unsafe_allow_html=True)
        st.divider()

        st.markdown("<div style='margin-top:1.5rem'></div>",unsafe_allow_html=True)
        st.divider()

        # ═══════════════════════════════════════
        # ② TIME TO THRESHOLD
        # ═══════════════════════════════════════
        st.markdown("<div style='margin-top:1.2rem'></div>",unsafe_allow_html=True)
        st.markdown("### ② How long until the safety threshold is reached?")
        st.caption(
            f"10^{log_initial:.1f} → 10^{log_limit:.1f} CFU/g  ·  "
            f"lag = {lag_h:.0f} h  ·  {medium}  ·  T = {temp:.1f}°C"
            +(f"  ·  NaCl = {nacl:.1f}%  ·  PCS = {pcs_c:.3f}% ({pcs_t})"
              if medium=="TSB" else ""))

        st.markdown("<div style='margin-top:1rem'></div>",unsafe_allow_html=True)

        # Three headline metrics
        col_fast,col_mean,col_slow=st.columns(3)
        col_fast.metric("⚠️ Fast growth scenario", fmt_time(t_fast),
                        delta="CI upper μmax", delta_color="inverse",
                        help="Shortest time — CI upper bound (fastest growth)")
        col_mean.metric("⏱ Mean scenario", fmt_time(t_mean),
                        delta=f"μmax = {gpr_mu:.4f} h⁻¹", delta_color="off",
                        help="Based on GPR mean μmax")
        col_slow.metric("✅ Slow growth scenario", fmt_time(t_slow),
                        delta="CI lower μmax", delta_color="normal",
                        help="Longest time — CI lower bound (slowest growth)")

        st.markdown("<div style='margin-top:1.5rem'></div>",unsafe_allow_html=True)

        # Timeline dot chart
        st.plotly_chart(fig_timeline_dots(t_mean,t_fast,t_slow),
                        use_container_width=True)

        st.markdown("<div style='margin-top:0.5rem'></div>",unsafe_allow_html=True)

        # Formula box
        delta_log=log_limit-log_initial
        st.info(
            f"**Formula:** t = lag_h + (Δlog × ln10) / μmax  \n"
            f"Δlog = {log_limit:.1f} − {log_initial:.1f} = **{delta_log:.1f} log units**  ·  "
            f"lag = **{lag_h:.0f} h**  ·  μmax (GPR mean) = **{gpr_mu:.4f} h⁻¹**  \n"
            f"📌 TSB laboratory medium basis — relative comparisons across conditions are the key output."
        )

        if medium=="SJ":
            st.info("ℹ️ **SJ mode:** temperature effect only. "
                    "Switch to TSB for NaCl/PCS hurdle analysis.")

        st.markdown("<div style='margin-top:1.5rem'></div>",unsafe_allow_html=True)
        st.divider()

        # ═══════════════════════════════════════
        # ③ EXPLANATION TABS
        # ═══════════════════════════════════════
        st.markdown("<div style='margin-top:1.2rem'></div>",unsafe_allow_html=True)
        tab_shap,tab_opt=st.tabs([
            "③ 🔍  Why this result?",
            "③ 🔧  What if I change conditions?"
        ])

        with tab_shap:
            st.markdown("<div style='margin-top:1rem'></div>",unsafe_allow_html=True)
            col_wf,col_gi=st.columns([1.4,1])
            with col_wf:
                sv_in=m["exp"].shap_values(x0)
                sv1=(sv_in[0,:,0] if sv_in.ndim==3 else sv_in[0]).astype(float)
                st.plotly_chart(fig_shap_waterfall(sv1,m["base_val"],rf_p),
                                use_container_width=True)
                net=rf_p-m["base_val"]
                st.caption(
                    f"Baseline μmax (dataset mean) = **{m['base_val']:.4f} h⁻¹**  ·  "
                    f"RF predicted = **{rf_p:.4f} h⁻¹**  ·  Net SHAP = **{net:+.4f} h⁻¹**  \n"
                    "Red = promotes growth  ·  Blue = inhibits growth")
            with col_gi:
                st.plotly_chart(fig_global_importance(m["sv_train"]),
                                use_container_width=True)
                st.caption("Temperature ~4× above all others — primary μmax driver.  \n"
                           "PCS Conc > NaCl — bacteriocin suppresses μmax more per unit.")

        with tab_opt:
            st.markdown("<div style='margin-top:1rem'></div>",unsafe_allow_html=True)
            st.markdown("**Which single change reduces μmax most — "
                        "and extends time to threshold by how much?**")
            st.caption("Continuous grid search within training ranges. "
                       "TSB medium basis — relative comparison only.")

            b0=float(m["rf"].predict(x0)[0])
            t_cur,_,_=time_to_threshold(b0,b0*0.9,b0*1.1,log_initial,log_limit,lag_h)
            results=[]

            if medium=="TSB":
                candidates=[
                    ("Temperature",[(t_try,med_enc,nacl,pcs_c,pcs_ev,
                                     f"{temp:.1f}°C → {t_try:.1f}°C")
                                    for t_try in np.linspace(TEMP_MIN,TRAIN_TEMP_MAX,12)
                                    if abs(t_try-temp)>=0.3]),
                    ("NaCl",[(temp,med_enc,n_try,pcs_c,pcs_ev,
                              f"{nacl:.1f}% → {n_try:.1f}%")
                             for n_try in np.linspace(0,TRAIN_NACL_MAX,10)
                             if abs(n_try-nacl)>=0.1]),
                    ("PCS Conc",[(temp,med_enc,nacl,pc_try,pcs_ev,
                                  f"{pcs_c:.3f}% → {pc_try:.3f}%")
                                 for pc_try in np.linspace(0,TRAIN_PCS_MAX,10)
                                 if abs(pc_try-pcs_c)>=0.005]),
                    ("PCS Type",[(temp,med_enc,nacl,pcs_c,
                                  int(m["le_p"].transform([pt])[0]),
                                  f"{pcs_t} → {pt}")
                                 for pt in ["None","VTABB","JJT01"] if pt!=pcs_t]),
                ]
            else:
                candidates=[
                    ("Temperature",[(t_try,med_enc,0,0,m["pcs_enc_none"],
                                     f"{temp:.1f}°C → {t_try:.1f}°C")
                                    for t_try in np.linspace(TEMP_MIN,TRAIN_TEMP_MAX,12)
                                    if abs(t_try-temp)>=0.3]),
                ]

            for label,cases in candidates:
                for args in cases:
                    *xargs,change=args
                    p=float(m["rf"].predict(enc_x(*xargs))[0])
                    if p<b0:
                        t_new,_,_=time_to_threshold(p,p*0.9,p*1.1,
                                                    log_initial,log_limit,lag_h)
                        dt=(t_new-t_cur) if not np.isinf(t_new) else 0
                        results.append(dict(label=label,change=change,
                                            new_mu=p,delta_mu=p-b0,delta_t=dt))
            results.sort(key=lambda r: r["delta_mu"])

            if results:
                cols_opt=st.columns(min(3,len(results)))
                for i,s in enumerate(results[:3]):
                    rank=["🥇","🥈","🥉"][i]
                    with cols_opt[i]:
                        st.success(
                            f"{rank} **{s['label']}**  \n"
                            f"{s['change']}  \n"
                            f"μmax: `{b0:.4f}` → `{s['new_mu']:.4f}` h⁻¹  \n"
                            f"Δt ≈ **+{s['delta_t']/24:.1f} d** (TSB ref.)")
            else:
                st.success("✅ Already near minimum μmax within training range.")
            if medium=="TSB":
                st.caption("⚠️ NaCl × PCS co-application not tested in Lee et al. (2023).")


    # ══════════════════════════════════════
    # TAB 2 — Strain Data
    # ══════════════════════════════════════
    with tab_strain:
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("## Strain Reference Data")
        st.markdown(
            "Actual measured μmax values from Lee et al. (2023) — Table A "
            "(Temperature × Medium experiment).  \n"
            "NaCl and PCS experiments were conducted in TSB only — "
            "strain comparison is available for the temperature experiment only.  \n\n"
            "**How to read this:** The model prediction (Prediction tab) is trained on all "
            "8 strains together and outputs a single μmax estimate. "
            "This tab shows the actual inter-strain variability in the raw data — "
            "useful for understanding how much the model's single prediction "
            "covers or misses for any specific strain."
        )

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # ── Build Table A inline ─────────────────────────────────────────────
        _table_A = [
            ("A. media",              4,  "SJ",  0.019), ("A. media",              8,  "SJ",  0.032),
            ("A. media",             15,  "SJ",  0.078), ("A. media",              4,  "TSB", 0.025),
            ("A. media",              8,  "TSB", 0.051), ("A. media",             15,  "TSB", 0.091),
            ("A. bestiarum",          4,  "SJ",  0.021), ("A. bestiarum",          8,  "SJ",  0.030),
            ("A. bestiarum",         15,  "SJ",  0.075), ("A. bestiarum",          4,  "TSB", 0.026),
            ("A. bestiarum",          8,  "TSB", 0.045), ("A. bestiarum",         15,  "TSB", 0.082),
            ("A. piscicola",          4,  "SJ",  0.019), ("A. piscicola",          8,  "SJ",  0.026),
            ("A. piscicola",         15,  "SJ",  0.057), ("A. piscicola",          4,  "TSB", 0.026),
            ("A. piscicola",          8,  "TSB", 0.044), ("A. piscicola",         15,  "TSB", 0.103),
            ("A. salmonicida SU2",    4,  "SJ",  0.014), ("A. salmonicida SU2",    8,  "SJ",  0.024),
            ("A. salmonicida SU2",   15,  "SJ",  0.043), ("A. salmonicida SU2",    4,  "TSB", 0.024),
            ("A. salmonicida SU2",    8,  "TSB", 0.046), ("A. salmonicida SU2",   15,  "TSB", 0.096),
            ("A. salmonicida Nr.21",  4,  "SJ",  0.016), ("A. salmonicida Nr.21",  8,  "SJ",  0.026),
            ("A. salmonicida Nr.21", 15,  "SJ",  0.053), ("A. salmonicida Nr.21",  4,  "TSB", 0.021),
            ("A. salmonicida Nr.21",  8,  "TSB", 0.038), ("A. salmonicida Nr.21", 15,  "TSB", 0.088),
            ("A. hydrophila",         8,  "SJ",  0.028), ("A. hydrophila",         15,  "SJ",  0.058),
            ("A. hydrophila",         8,  "TSB", 0.038), ("A. hydrophila",         15,  "TSB", 0.096),
            ("A. dhakensis",          8,  "SJ",  0.020), ("A. dhakensis",          15,  "SJ",  0.080),
            ("A. dhakensis",          8,  "TSB", 0.033), ("A. dhakensis",          15,  "TSB", 0.112),
            ("A. caviae",            15,  "SJ",  0.064),
            ("A. caviae",             8,  "TSB", 0.035), ("A. caviae",             15,  "TSB", 0.082),
        ]
        df_strains = pd.DataFrame(_table_A, columns=["strain","temp_C","medium","mu_max"])

        STRAIN_ORDER = [
            "A. media","A. bestiarum","A. piscicola",
            "A. salmonicida SU2","A. salmonicida Nr.21",
            "A. hydrophila","A. dhakensis","A. caviae"
        ]
        TEMP_COLORS  = {4: "#63b3ed", 8: "#f6ad55", 15: "#fc8181"}
        NOTE_NG      = {"A. hydrophila":4,"A. dhakensis":4,"A. caviae":4}

        # ── Filter controls ──────────────────────────────────────────────────
        fc1, fc2 = st.columns([1, 1.5])
        with fc1:
            med_sel = st.radio("Medium", ["TSB","SJ","Both"],
                               horizontal=True, key="sd_med")
        with fc2:
            temp_sel = st.multiselect("Temperature (°C)", [4, 8, 15],
                                      default=[4, 8, 15], key="sd_temp")

        if not temp_sel:
            st.warning("Select at least one temperature.")
            st.stop()

        df_f = df_strains[df_strains["temp_C"].isin(temp_sel)]
        if med_sel != "Both":
            df_f = df_f[df_f["medium"] == med_sel]

        if df_f.empty:
            st.info("No data for this combination.")
            st.stop()

        # ── Bar chart ────────────────────────────────────────────────────────
        fig_sd = go.Figure()

        for t in sorted(temp_sel):
            for med in (["TSB","SJ"] if med_sel=="Both" else [med_sel]):
                sub = df_f[(df_f["temp_C"]==t) & (df_f["medium"]==med)]
                if sub.empty:
                    continue
                sub = sub.set_index("strain").reindex(STRAIN_ORDER).reset_index()
                y_vals  = sub["mu_max"].tolist()
                x_vals  = sub["strain"].tolist()

                # Text labels: value or "NG"
                text_labels = []
                for i_s, strain in enumerate(STRAIN_ORDER):
                    val = sub.loc[sub["strain"]==strain, "mu_max"]
                    if val.isna().all() or (strain in NOTE_NG and t == NOTE_NG[strain]):
                        text_labels.append("NG")
                    elif not val.isna().all():
                        text_labels.append(f"{val.values[0]:.4f}")
                    else:
                        text_labels.append("")

                fig_sd.add_trace(go.Bar(
                    x=x_vals, y=y_vals,
                    name=f"{t}°C {med}",
                    marker=dict(
                        color=TEMP_COLORS[t],
                        opacity=0.88 if med=="TSB" else 0.45,
                        pattern_shape="" if med=="TSB" else "/",
                        line=dict(color=BG, width=0.6)
                    ),
                    text=text_labels,
                    textposition="outside",
                    textfont=dict(color=TEXT_CLR, size=11),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"{t}°C  {med}<br>"
                        "μmax = <b>%{y:.4f}</b> h⁻¹<extra></extra>"
                    )
                ))

        fig_sd.update_layout(
            **PLOTLY_LAYOUT, font=_FONT, height=420,
            title=dict(
                text="Measured μmax by Strain — "
                     + med_sel + " | "
                     + ", ".join(str(t)+"°C" for t in sorted(temp_sel)),
                font=dict(color=TEXT_CLR, size=FS_CHART), x=0.01),
            xaxis=_xax(title="Strain"),
            yaxis=_yax(title="μmax (h⁻¹)", rangemode="tozero"),
            barmode="group",
            legend=_legend(orientation="h", yanchor="bottom", y=1.02,
                           xanchor="left", x=0,
                           font=dict(color=TEXT_CLR, size=12)),
            margin=dict(l=65, r=30, t=70, b=100),
        )
        fig_sd.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_sd, use_container_width=True)

        st.caption(
            "NG = No Growth observed at this condition.  "
            "A. hydrophila, A. dhakensis, A. caviae showed no growth at 4°C."
        )

        st.divider()

        # ── TSB vs SJ comparison (if Both or individual) ─────────────────────
        st.markdown("### TSB vs SJ — Medium Effect per Strain")
        st.markdown(
            "SJ (salmon juice) μmax is consistently lower than TSB. "
            "This panel quantifies the medium effect for each strain — "
            "available for temperature experiment only (Table A)."
        )

        df_tsb = df_strains[df_strains["medium"]=="TSB"]
        df_sj  = df_strains[df_strains["medium"]=="SJ"]

        fig_med = go.Figure()
        for t in [4, 8, 15]:
            sub_tsb = df_tsb[df_tsb["temp_C"]==t].set_index("strain").reindex(STRAIN_ORDER)
            sub_sj  = df_sj[df_sj["temp_C"]==t].set_index("strain").reindex(STRAIN_ORDER)
            ratio   = (sub_sj["mu_max"] / sub_tsb["mu_max"] * 100).round(1)

            fig_med.add_trace(go.Bar(
                x=STRAIN_ORDER, y=ratio.tolist(),
                name=f"{t}°C",
                marker=dict(color=TEMP_COLORS[t], opacity=0.85,
                            line=dict(color=BG, width=0.5)),
                text=[f"{v:.0f}%" if not np.isnan(v) else "NG"
                      for v in ratio],
                textposition="outside",
                textfont=dict(color=TEXT_CLR, size=11),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{t}°C<br>"
                    "SJ/TSB ratio = <b>%{y:.1f}%</b><extra></extra>"
                )
            ))

        fig_med.add_hline(y=100, line_dash="dot", line_color=MUTED,
                          line_width=1.5,
                          annotation_text="TSB baseline (100%)",
                          annotation_font=dict(color=MUTED, size=11))

        fig_med.update_layout(
            **PLOTLY_LAYOUT, font=_FONT, height=360,
            title=dict(
                text="SJ μmax as % of TSB μmax — by Strain and Temperature",
                font=dict(color=TEXT_CLR, size=FS_CHART), x=0.01),
            xaxis=_xax(title="Strain"),
            yaxis=_yax(title="SJ / TSB μmax (%)", range=[0, 140]),
            barmode="group",
            legend=_legend(orientation="h", yanchor="bottom", y=1.02,
                           xanchor="left", x=0),
            margin=dict(l=65, r=30, t=70, b=100),
        )
        fig_med.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_med, use_container_width=True)

        # Medium effect summary
        df_paired = df_strains.pivot_table(
            index=["strain","temp_C"], columns="medium", values="mu_max"
        ).reset_index()
        df_paired = df_paired.dropna(subset=["TSB","SJ"])
        df_paired["SJ/TSB (%)"] = (df_paired["SJ"] / df_paired["TSB"] * 100).round(1)
        overall_ratio = df_paired["SJ/TSB (%)"].mean()
        st.caption(
            f"Overall mean SJ/TSB ratio: **{overall_ratio:.1f}%** "
            f"(SJ μmax is on average {100-overall_ratio:.1f}% lower than TSB).  "
            "Inter-strain range visible in chart — some strains show stronger medium sensitivity."
        )

        st.divider()

        # ── Raw data table ───────────────────────────────────────────────────
        st.markdown("### Raw Data — Table A (Lee et al. 2023)")
        df_display = df_strains.copy()
        df_display.columns = ["Strain", "Temp (°C)", "Medium", "μmax (h⁻¹)"]
        df_display = df_display.sort_values(["Strain","Medium","Temp (°C)"])
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=350)
        st.caption(
            "Source: Lee et al. (2023) Table A — Temperature × Medium experiment.  "
            "NaCl and PCS experiments (Tables B–D) used TSB only — strain breakdown "
            "not available for those conditions."
        )

    # ── About tab ─────────────────────────────────────────────────────────────
    with tab_about:
        st.header("About this Project")
        st.markdown("""
This tool extends **Lee et al. (2023)** — *"The effect of food processing factors on the
growth kinetics of Aeromonas strains isolated from ready-to-eat seafood"*
(*Int. J. Food Microbiology* 384:109985) — beyond the Ratkowsky mechanistic model
using machine learning and SHAP explainability.

**Core question:**  *How long does it take for Aeromonas to reach a defined safety
threshold under a given set of storage conditions?*

> ⚠️ **Scope:** All predictions are based on **TSB (laboratory broth)** data.
> Results reflect laboratory conditions and **cannot be used directly as product shelf-life**.
> Designed for **relative comparison of hurdle conditions** and **research-purpose analysis**.
        """)

        st.subheader("Pipeline")
        st.markdown("""
| Step | File | Purpose |
|------|------|---------|
| 1 | `step1_data_extraction_eda.ipynb` | Data extraction, EDA, Ratkowsky reproduction |
| 2 | `step2_ml_modeling_v5.ipynb` | GPR + RF + XGBoost, LOSO CV |
| 3 | `step3_shap_analysis.ipynb` | SHAP feature attribution |
| 4 | **This app** | Interactive growth prediction + condition comparison |
        """)

        st.subheader("Key Design Decisions")
        st.markdown("""
**Growth scenario formula:** `t = lag_h + (Δlog × ln10) / μmax`  
GPR CI → fast/slow growth scenarios. For relative comparison — not absolute shelf-life.

**species_enc excluded** — LOSO design consistency. Removing it improved RF R² by +0.039.

**Medium: TSB vs SJ** — NaCl/PCS experiments used TSB only (Tables B–D).

**lag_h not predicted** — per-fold R²=0.579±0.365, too unstable.
Three scenario assumptions: conservative (0 h), median (5 h), custom.
        """)

        st.subheader("Validation")
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(fig_about_benchmark(m["bench"]),use_container_width=True)
        with c2: st.plotly_chart(fig_about_shap_bar(m["shap_imp"]),use_container_width=True)

        st.subheader("Why Two Models?")
        c1,c2=st.columns(2)
        with c1:
            st.info("**🔵 GPR**  LOSO R² = 0.892 ± 0.040  \n"
                    "Calibrated uncertainty → fast/slow growth scenarios.  \n"
                    "CI widens automatically in extrapolation zones.")
        with c2:
            st.info("**🔴 RandomForest + SHAP**  LOSO R² = 0.865 ± 0.053  \n"
                    "Exact Shapley values → which conditions drive μmax.  \n"
                    "Hurdle Optimizer uses RF predictions.")

        st.subheader("Key Findings")
        st.markdown("""
- **Temperature** — primary μmax driver, ~4× SHAP weight vs all others
- **PCS Conc > NaCl** — bacteriocin suppresses μmax more per unit
- **PCS threshold** — effective above ~0.125%; negligible below ~0.025%
- **SJ μmax ≈ 35% lower** than TSB — medium matrix effect
- **lag_h** — per-fold R²=0.579±0.365 — too unstable to predict
- **Ymax** — R²<0 per-fold — species-determined ceiling
        """)

        st.subheader("Limitations")
        st.warning("""
- **All data from TSB laboratory broth** — not actual food product
- **Growth scenario ≠ shelf-life** — food matrix effects not captured
- SJ data: temperature only — hurdle effects in salmon matrix absent
- n=118, 8 strains — limited taxonomic coverage
- Temperatures tested: 4, 8, 15°C — intermediate = GPR interpolation
- NaCl × PCS never co-tested — interaction unquantifiable
- lag_h not predicted — treated as scenario assumption
- ⚠️ Research and educational purposes only
        """)


if __name__=="__main__":
    main()
