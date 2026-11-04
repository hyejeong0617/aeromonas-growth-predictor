"""Interactive scientific case study. Reads reviewed CSVs only; never fits models."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DOI = 'https://doi.org/10.1016/j.ijfoodmicro.2022.109985'
GITHUB = 'https://github.com/hyejeong0617/aeromonas-growth-predictor'
COLORS = ['#167D8D', '#5066A1', '#B58341']
UNIT = 'μmax (log10 CFU/h)'
st.set_page_config(page_title='Aeromonas | Scientific ML Case Study', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {max-width:1160px;padding-top:2.5rem;padding-bottom:3rem;}
h1 {letter-spacing:-.045em;font-weight:700!important;}
h2 {letter-spacing:-.025em;}
p,li {line-height:1.65;}
.kicker {color:#167D8D;font-size:.76rem;letter-spacing:.16em;font-weight:700;text-transform:uppercase;}
.intro {font-size:1.18rem;color:#4B6068;max-width:760px;margin-bottom:1.6rem;}
.workflow {border-left:3px solid #167D8D;padding:1rem 1.2rem;background:#F1F6F5;margin:1.2rem 0;line-height:1.8;}
.fold {display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:6px;margin:1rem 0;}
.fold span {background:#E5F0EE;color:#235F62;border:1px solid #B6D4CE;border-radius:6px;padding:16px 0;text-align:center;font-weight:600;}
.fold span.test {background:#344A72;color:white;border-color:#344A72;}
@media(max-width:600px){.block-container{padding:1.1rem;}h1{font-size:2rem!important;}.intro{font-size:1rem;}.fold{gap:3px;}.fold span{padding:12px 0;font-size:.8rem;}}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv(relative, modified):
    return pd.read_csv(ROOT / relative)

def data(relative, required=False, columns=()):
    path = ROOT / relative
    try:
        frame = read_csv(relative, path.stat().st_mtime_ns)
        if not set(columns).issubset(frame.columns):
            raise ValueError('Required columns are absent')
        return frame
    except (OSError, ValueError, pd.errors.ParserError):
        if required:
            st.error('The reviewed experimental dataset is unavailable. Restore the source data to view this case study.')
            st.stop()
        st.info('This reviewed result is currently unavailable. Other sections remain accessible.')
        return None

def plot(fig, key):
    fig.update_layout(template='plotly_white', font=dict(family='Arial, sans-serif', size=13, color='#243C46'),
                      margin=dict(l=15,r=15,t=35,b=25), paper_bgcolor='rgba(0,0,0,0)',
                      colorway=COLORS, legend=dict(orientation='h',y=-.2), hoverlabel=dict(font_size=13))
    st.plotly_chart(fig, use_container_width=True, key=key, config={'displayModeBar':False, 'scrollZoom':False})

st.markdown('<div class="kicker">Food microbiology / Scientific machine learning</div>', unsafe_allow_html=True)
st.title('Predicting Aeromonas Growth with Machine Learning')
st.markdown('<div class="intro">From published food microbiology research to strain-aware validation and interpretable machine learning.</div>', unsafe_allow_html=True)
st.markdown('<div class="workflow"><strong>Published research (2023)</strong> &nbsp;→&nbsp; <strong>ML extension (2026)</strong> &nbsp;→&nbsp; <strong>Interactive case study</strong></div>', unsafe_allow_html=True)
st.markdown(f'[Publication · Lee et al. (2023)]({DOI})  ·  [Source repository]({GITHUB})')
st.divider()
section = st.radio('Explore the case study', ['01 · Original research','02 · ML approach','03 · Results & insights'], horizontal=True, label_visibility='collapsed', key='section')
raw = data('outputs/aeromonas_kinetics_master.csv', True, ['strain','temperature_C','growth','mu_max','PCS_conc_pct','NaCl_pct','experiment'])
status_names = {'Y':'Growth observed', 'NL':'Growth observed · no lag', 'NG':'No growth observed'}
treatments = {'Temperature':'Temperature & medium','NaCl':'Added NaCl','PCS_VTABB':'PCS · VTABB','PCS_JJT01':'PCS · JJT01'}

if section.startswith('01'):
    st.header('Understanding Aeromonas Growth Under Food Processing Conditions')
    st.write('Lee et al. (2023), International Journal of Food Microbiology, examined Aeromonas strains isolated from ready-to-eat seafood. Growth experiments measured responses to temperature, added salt, purified condensed smoke (PCS), and culture medium.')
    st.markdown(f'**{len(raw)} experimental conditions · {raw.mu_max.notna().sum()} growth-positive records · {raw.strain.nunique()} strain groups**')
    st.caption('Eight strains across seven species. Media: tryptic soy broth (TSB) and salmon juice (SJ). Measurements are published kinetic summaries, not individual growth-curve time points.')
    st.subheader('Explore the observed growth rates')
    c1,c2,c3 = st.columns(3)
    temps = c1.multiselect('Temperature (°C)', sorted(raw.temperature_C.unique()), default=sorted(raw.temperature_C.unique()), key='temperatures')
    strains = c2.multiselect('Strain', sorted(raw.strain.unique()), default=sorted(raw.strain.unique()), format_func=lambda s:s.replace('_',' '), key='strains')
    treatment = c3.selectbox('Experimental treatment', ['All experiments']+list(treatments.values()), index=1, key='treatment')
    sub=raw[raw.temperature_C.isin(temps)&raw.strain.isin(strains)].copy()
    if treatment!='All experiments':sub=sub[sub.experiment.map(treatments)==treatment]
    sub['Strain']=sub.strain.str.replace('_',' ')
    sub['Status']=sub.growth.map(status_names).fillna('Not reported')
    sub['Treatment']=sub.experiment.map(treatments)
    measured=sub[sub.growth.isin(['Y','NL'])&sub.mu_max.notna()]
    if sub.empty:st.info('No conditions match these filters. Select at least one temperature and strain.')
    elif measured.empty:st.info('No measured growth rates in this selection. See the condition records below for NG observations.')
    else:
        fig=px.scatter(measured,x='temperature_C',y='mu_max',color='Treatment',symbol='Status',
                       hover_data=['Strain','medium','NaCl_pct','PCS_conc_pct'],
                       labels={'temperature_C':'Temperature (°C)','mu_max':UNIT,'medium':'Medium','NaCl_pct':'Added NaCl (%)','PCS_conc_pct':'PCS (%)'},color_discrete_sequence=COLORS+['#776A95'])
        fig.for_each_trace(lambda t: t.update(name=t.name.replace('Growth observed · no lag','NL').replace('Growth observed','Y').replace('Temperature & medium','Temp / medium')))
        fig.update_traces(marker=dict(size=10,opacity=.75));fig.update_xaxes(tickvals=[4,8,15]);plot(fig,'observed')
    st.caption(f'Selection: {len(sub)} conditions · {len(measured)} measured rates · {(sub.growth=="NG").sum()} NG. Overlapping points may share the same measured rate.')
    st.caption('**Takeaway:** Measured growth rates tend to increase with temperature in the studied conditions; other treatment effects require condition-specific comparison.')
    st.info('NG = no growth observed; its rate is missing, never plotted as zero. NL = growth with no lag; the measured rate is retained. Other missing values remain unreported. NA in Ymax means no asymptote.')
    with st.expander('Condition records & source definitions'):
        cols=['Strain','temperature_C','Treatment','medium','NaCl_pct','PCS_conc_pct','Status','mu_max','lag_h','mu_max_source_flag']
        st.dataframe(sub[cols].rename(columns={'temperature_C':'Temperature (°C)','medium':'Medium','NaCl_pct':'Added NaCl (%)','PCS_conc_pct':'PCS (%)','mu_max':UNIT,'lag_h':'Lag (h)','mu_max_source_flag':'Source rate flag'}),hide_index=True,use_container_width=True)
        st.write('PCS treatments are VTABB and JJT01 at 0.026%, 0.13% and author-confirmed 0.26%. The 0.26% correction supersedes conflicting published table labels; it is not a legal concentration limit.')
        st.write('NaCl values mean added salt. Total salt in TSB = added NaCl + 0.5 percentage points. No baseline salt correction is applied to SJ.')
        st.caption('Parameter SD/SE terminology and the source Ymax denominator remain ambiguous. These are preserved in the provenance record; no speculative conversion was applied.')

elif section.startswith('02'):
    st.header('Can Machine Learning Predict Growth in Unseen Strains?')
    st.write('The later Python ML extension asks whether environmental conditions can predict growth rates for a strain excluded from training. It extends the published experimental summaries; these ML results are not findings reported in the original paper.')
    st.markdown('<div class="workflow">Published experimental summaries → structured dataset → ML models → LOSO validation → interpretation</div>',unsafe_allow_html=True)
    st.subheader('Three predictive models, one scientific reference')
    st.markdown('**GPR** estimates a response with predictive uncertainty. **Random Forest** captures nonlinear patterns and supports SHAP interpretation. **XGBoost** provides a regularized boosted-tree comparison. **Ratkowsky** is a temperature-based scientific reference with a narrower input set.')
    st.caption('Five predictors: temperature, medium, added NaCl, PCS concentration and PCS type. Strain identity is excluded. Hyperparameters were reused without tuning during revalidation.')
    st.subheader('Leave one strain out (LOSO)')
    st.write('**7 strains for training → 1 unseen strain for testing.** Move the slider to see which strain group is held out in each fold.')
    held=st.select_slider('Illustrative held-out strain group', options=list(range(1,9)),value=8,key='held_out')
    st.markdown('<div class="fold">'+''.join(f'<span class="{"test" if i==held else "train"}">S{i}</span>' for i in range(1,9))+'</div>',unsafe_allow_html=True)
    st.markdown(f'**Training: 7 strain groups** · **Testing: S{held} (held out)**')
    st.caption('Pale teal = training; dark blue = test. S1–S8 are illustrative group labels. Each strain is held out once across eight folds.')
    st.write('All records from the held-out strain stay outside training. GPR scaling is fitted within each training fold. Training-only category encodings were checked to match the existing vocabulary in every main LOSO fold.')
    summary=data('outputs/step2_loso_performance_summary.csv',columns=['target','model','R2_mean','R2_std'])
    if summary is not None:
        m=summary[summary.target=='mu_max']
        fig=px.bar(m,x='model',y='R2_mean',error_y='R2_std',text=m.R2_mean.map(lambda x:f'{x:.5f}'),labels={'model':'','R2_mean':'Mean fold R²'},color='model',color_discrete_sequence=COLORS)
        fig.update_layout(showlegend=False);fig.update_yaxes(range=[0,1.05]);plot(fig,'comparison')
        st.caption('μmax · 118 growth-positive records · eight folds. Bars: unweighted mean fold R²; error bars: fold standard deviation. This is not pooled out-of-fold R².')
        st.write('**Takeaway:** GPR achieved the highest mean held-out-strain R² among the three evaluated models. This is internal cross-validation, not independent external validation.')
    with st.expander('Ratkowsky: a deliberately limited comparison'):
        st.write('A separate 41-observation Temperature-experiment subset uses the same eight held-out strain splits. Ratkowsky uses temperature only; ML uses five feature columns, but only temperature and medium vary in this subset. Predictors and fitting objectives differ.')
        b=data('outputs/step2_ml_vs_ratkowsky_benchmark.csv',columns=['Model','R²','RMSE'])
        if b is not None:st.dataframe(b.rename(columns={'R²':'Pooled OOF R²','MAE':'MAE (log10 CFU/h)','RMSE':'RMSE (log10 CFU/h)'}),hide_index=True,use_container_width=True)
        st.caption('Pooled OOF metrics on 41 observations cannot be compared directly with the mean fold metrics above. This is not an identical-information comparison.')
else:
    st.header('What Did the Models Learn?')
    st.write('Growth-rate predictions generalize across held-out strains within this experiment. Interpretation remains bounded by the small dataset and its growth-positive scope.')
    score=data('outputs/step2_loso_performance_summary.csv',columns=['target','model','R2_mean'])
    if score is not None:
        gpr=score[(score.target=='mu_max') & (score.model=='GPR')]
        if not gpr.empty:
            a,b,c=st.columns(3)
            a.metric('GPR mean LOSO R²', f'{gpr.iloc[0].R2_mean:.3f}')
            b.metric('Held-out strain folds', '8')
            interval_rows=data('audit/phase2/gpr_intervals.csv', columns=['covered'])
            if interval_rows is not None and len(interval_rows):
                covered_count=int(interval_rows.covered.astype(str).str.lower().eq('true').sum())
                c.metric('95% interval coverage', f'{covered_count}/{len(interval_rows)} ({covered_count/len(interval_rows):.2%})')
            st.caption('LOSO R² is the mean across eight folds; interval coverage is empirical within the same experiment, not independent calibration.')
    st.subheader('Observed vs predicted μmax')
    model=st.selectbox('Model', ['GPR','RandomForest','XGBoost'],format_func=lambda x:'Random Forest' if x=='RandomForest' else x,key='model')
    oof=data('audit/phase2/oof_predictions.csv',columns=['target','model','row','actual','prediction'])
    pooled=data('audit/phase2/pooled_oof_metrics.csv',columns=['target','model','R2','MAE','RMSE'])
    if oof is not None:
        d=oof[(oof.target=='mu_max')&(oof.model==model)].copy()
        d=d.merge(raw[['strain','temperature_C']].rename_axis('row').reset_index(),on='row',how='left')
        fig=px.scatter(d,x='actual',y='prediction',color=d.temperature_C.astype(str),hover_data=['strain'],labels={'actual':f'Observed {UNIT}','prediction':f'Predicted {UNIT}','color':'Temperature (°C)'},color_discrete_sequence=COLORS)
        top=max(d.actual.max(),d.prediction.max())*1.05
        fig.add_trace(go.Scatter(x=[0,top],y=[0,top],mode='lines',line=dict(color='#88979E',dash='dash'),name='Perfect agreement',hoverinfo='skip'))
        plot(fig,'predictions')
        st.caption('Every point is an out-of-fold prediction for a strain absent from training—not a training fit.')
    if pooled is not None:
        m=pooled[(pooled.target=='mu_max')&(pooled.model==model)]
        if not m.empty:
            v=m.iloc[0];st.markdown(f'**Pooled OOF R² {v.R2:.3f}** · MAE {v.MAE:.4f} · RMSE {v.RMSE:.4f} log10 CFU/h')
    st.write('**Takeaway:** Each plotted prediction comes from a fold that excluded the corresponding strain from training.')
    st.divider()
    st.subheader('Which features contribute to RF predictions?')
    rank=data('outputs/step3_hurdle_ranking.csv',columns=['Feature','Mean_Abs_SHAP'])
    if rank is not None:
        r=rank.sort_values('Mean_Abs_SHAP')
        fig=px.bar(r,x='Mean_Abs_SHAP',y='Feature',orientation='h',labels={'Mean_Abs_SHAP':'Mean |SHAP| (log10 CFU/h)','Feature':''},color_discrete_sequence=[COLORS[0]])
        plot(fig,'shap')
        st.write('**Takeaway:** Temperature has the largest mean absolute SHAP attribution, followed by PCS concentration. These contributions describe the fitted RF, not causal effects.')
        st.caption('Full-data RF interpretation · 118 growth-positive records · five environmental features. SHAP additivity verified; no interpretation is recalculated in this app.')
    st.subheader('How much uncertainty did GPR capture?')
    intervals=data('audit/phase2/gpr_intervals.csv',columns=['actual','prediction','lower','upper','covered'])
    if intervals is not None:
        g=intervals.sort_values('actual').reset_index(drop=True)
        covered=g.covered.astype(str).str.lower().eq('true').sum()
        st.markdown(f'**{covered}/{len(g)} observations inside nominal 95% intervals ({covered/len(g):.2%}).**')
        with st.expander('Inspect the saved prediction intervals'):
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=g.index,y=g.upper,mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=g.index,y=g.lower,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(22,125,141,.18)',name='Nominal 95% interval',hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=g.index,y=g.prediction,mode='lines',name='GPR OOF prediction',line=dict(color=COLORS[0])))
            fig.add_trace(go.Scatter(x=g.index,y=g.actual,mode='markers',name='Observed',marker=dict(size=5,color='#344A72')))
            fig.update_layout(xaxis_title='Observations sorted by observed rate',yaxis_title=UNIT);plot(fig,'intervals')
        st.caption('Empirical coverage in internal LOSO evaluation—not independent external calibration. The observations are small in number and experimentally correlated.')
    st.divider()
    st.subheader('What this evidence can—and cannot—support')
    st.markdown('''- A small, correlated dataset supports a bounded study of unseen-strain prediction.
- Regression excludes NG conditions; it does not predict whether growth will occur.
- No independent food-product validation has been performed. SHAP does not establish causality.
- No validated shelf-life, inhibition-threshold or regulatory decision support is provided.''')
    st.caption('The case study focuses on μmax. Lag predictions are variable and Ymax generalization is poor; neither is presented as a validated application.')
st.divider()
st.caption('Research and educational use · Original publication and later ML extension are distinct contributions. Experimental provenance and Phase 2 validation define the scope of this case study.')
st.markdown(f'[Read the publication]({DOI}) · [Explore the source]({GITHUB})')
