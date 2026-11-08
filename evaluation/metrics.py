import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
from scipy.special import expit


def fit_logistic5(prediction, target):
    """Fit beta1*(sigmoid(beta2*(x-beta3))-.5)+beta4*x+beta5.

    All finite evaluation samples participate in least squares. X/Y are
    standardized for stability; this is the same five-parameter family.
    The fit uses evaluation labels, not an inference-time calibration model.
    """
    x=np.asarray(prediction,dtype=np.float64).reshape(-1)
    y=np.asarray(target,dtype=np.float64).reshape(-1)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('logistic5 requires equally sized finite arrays')
    info={'status':'not_fitted','samples':int(x.size),'fit_scope':'current_evaluation_condition',
          'parameters_standardized':None,'fallback':'clipped_raw_prediction','output_range':[0.0,1.0]}
    def fallback():
        info['clipped_count']=int(((x<0)|(x>1)).sum())
        return np.clip(x,0.,1.),info
    if x.size<6:
        info['status']='insufficient_samples';return fallback()
    xm,xs,ym,ys=float(x.mean()),float(x.std()),float(y.mean()),float(y.std())
    if xs<=np.finfo(float).eps or ys<=np.finfo(float).eps:
        info['status']='constant_input_or_target';return fallback()
    z=(x-xm)/xs; t=(y-ym)/ys
    info.update(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
    slope=float(np.mean(z*t))
    def value(p):
        a,b,c,d,e=p
        return a*(expit(b*(z-c))-.5)+d*z+e
    def jacobian(p):
        a,b,c,d,e=p
        s=expit(b*(z-c));g=s*(1-s)
        return np.column_stack((s-.5,a*g*(z-c),-a*b*g,z,np.ones_like(z)))
    best=None;attempts=[]
    def curve(z,a,b,c,d,e):
        return a*(expit(b*(z-c))-.5)+d*z+e
    # Use SciPy's direct five-parameter curve fitter with bounded, diverse starts.
    # Positive amplitude and linear slope preserve increasing loss-to-loss mapping.
    bounds=([0.,1e-4,-10.,0.,-5.],[10.,50.,10.,5.,5.])
    for steepness in (0.5,2.0,8.0):
        for center in np.quantile(z,[.2,.5,.8]):
            initial=[1.0,steepness,float(center),max(slope*.2,1e-3),0.0]
            try:
                params,_,fit_output,message,status=curve_fit(
                    curve,z,t,p0=initial,bounds=bounds,method='trf',x_scale='jac',
                    max_nfev=1200,ftol=1e-8,xtol=1e-8,gtol=1e-8,full_output=True)
                residual=curve(z,*params)-t
                objective=float(np.mean(residual**2))
                success=status in (1,2,3,4) and np.isfinite(objective)
                attempts.append(dict(success=bool(success),status=int(status),
                                     message=str(message),nfev=int(fit_output.get('nfev',0)),
                                     objective=objective))
                if success and (best is None or objective<best[0]):best=(objective,params,info)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                attempts.append(dict(success=False,message=str(error)))
    if best is None:
        info.update(status='optimizer_failed',attempts=attempts);return fallback()
    objective,params,fit_info=best
    mapped=curve(z,*params)*ys+ym
    if not np.isfinite(mapped).all() or mapped.std()<=np.finfo(float).eps:
        info['status']='invalid_fit';return fallback()
    info['clipped_count']=int(((mapped<0)|(mapped>1)).sum())
    mapped=np.clip(mapped,0.,1.)
    if mapped.std()<=np.finfo(float).eps:
        info['status']='constant_after_clip';return fallback()
    info.update(status='fitted',parameters_standardized=params.tolist(),
                optimizer='scipy.optimize.curve_fit (TRF)',nfev=int(fit_info.get('nfev',0)),
                objective_standardized=float(objective),
                successful_starts=sum(a['success'] for a in attempts),total_starts=len(attempts),
                attempts=attempts,fallback=None)
    return mapped,info

def _safe(fn):
    try: return float(fn())
    except Exception: return float("nan")

def _auroc(labels, scores):
    labels=np.asarray(labels,dtype=int); scores=np.asarray(scores,dtype=float)
    pos=scores[labels==1]; neg=scores[labels==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    return float(((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg)))

def _auprc(labels, scores):
    labels=np.asarray(labels,dtype=int); order=np.argsort(-np.asarray(scores,dtype=float)); y=labels[order]
    if y.sum()==0: return float("nan")
    precision=np.cumsum(y)/(np.arange(len(y))+1); return float((precision*y).sum()/y.sum())

def evaluate_predictions(y_true, y_pred, failure_true, risk_prob, logistic_details=None):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); failure_true=np.asarray(failure_true); risk_prob=np.asarray(risk_prob)
    bins=np.linspace(0,1,11); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(risk_prob>=lo)&(risk_prob<hi)
        if m.any(): ece += m.mean()*abs(risk_prob[m].mean()-failure_true[m].mean())
    mapped,fit=fit_logistic5(y_pred,y_true)
    if logistic_details is not None:logistic_details.update(fit)
    return {"srcc":_safe(lambda:spearmanr(y_true,y_pred).statistic),
            "plcc":_safe(lambda:pearsonr(y_true,mapped).statistic),
            "plcc_raw":_safe(lambda:pearsonr(y_true,y_pred).statistic),
            "plcc_fit_success":float(fit['status']=='fitted'),
            "mae":float(np.mean(np.abs(y_true-y_pred))), "auroc":_auroc(failure_true,risk_prob), "auprc":_auprc(failure_true,risk_prob), "brier":float(np.mean((risk_prob-failure_true)**2)), "ece":float(ece)}

def macro_average(per_condition):
    values=list(per_condition.values())
    if not values: return {}
    metric_names=tuple(values[0].keys())
    return {name:float(np.nanmean([metrics[name] for metrics in values])) for name in metric_names}
