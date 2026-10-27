"""Ограниченные операторы для проверки предложений двух ИИ. Не штатный поиск."""
from dataclasses import asdict, dataclass, field, replace
import math
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from contracts import ExperimentSpec, fingerprint, parse_experiment
from metrics import normalized_probabilities
from models import sample_weights, train_model
from preprocessing import FeatureBuilder, lightgbm_fit_metadata

VARIANTS={"sentinel","median","conditional","missing_pattern","lgb_f1",
          "linear_residual","linear_tree","log_smear","huber","conflict_weight"}


@dataclass
class ProposalSpec:
    model: str = "proposal"
    variant: str = "median"
    base_config: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    seed: int = 42
    hypothesis: str = "Проверка предложения ИИ"

    def validate(self,task):
        if self.model!="proposal" or self.variant not in VARIANTS or not isinstance(self.base_config,dict):
            raise ValueError("Недопустимый исследовательский рецепт")
        base=parse_experiment(self.base_config,task)
        if base.model not in {"catboost","lightgbm"}:
            raise ValueError("Недопустимая базовая модель рецепта")
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or not 0<=self.seed<2**31:
            raise ValueError("Некорректный seed")
        if not isinstance(self.hypothesis,str) or len(self.hypothesis)>1200:
            raise ValueError("Некорректная гипотеза")
        for values in (self.columns,self.groups):
            if not isinstance(values,list) or len(values)>256 or len(set(values))!=len(values):
                raise ValueError("Неверный список столбцов")
            if any(not isinstance(c,str) or c not in task.feature_columns or c in task.id_columns+[task.target] for c in values):
                raise ValueError("Неизвестный или запрещённый столбец")
        if set(self.columns)&set(task.categorical_columns):
            raise ValueError("Ожидаются числовые столбцы преобразования")
        if self.variant in {"lgb_f1","conflict_weight"} and (not task.classification or base.model!="lightgbm"):
            raise ValueError("Рецепт требует LightGBM-классификацию")
        if self.variant in {"linear_residual","linear_tree","log_smear","huber"} and task.classification:
            raise ValueError("Регрессионный рецепт неприменим к классам")
        if self.variant in {"linear_residual","linear_tree","log_smear","huber"} and base.model!="lightgbm":
            raise ValueError("Регрессионному рецепту нужен LightGBM")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def key(self):
        value=self.to_dict();value.pop("hypothesis")
        value["base_config"]=dict(value["base_config"])
        value["base_config"].pop("hypothesis",None)
        return fingerprint(value)

    @classmethod
    def from_dict(cls,data,task):
        return cls(**data).validate(task)


class NumericPolicy:
    """Политики X: статистики только fit, оригинальные нули не маскируются глобально."""
    def __init__(self,variant,columns,groups=()):
        self.variant,self.columns,self.groups=variant,list(columns),list(groups)

    def fit(self,frame):
        if any(c not in frame for c in self.columns+self.groups):
            raise ValueError("Нет столбцов политики")
        cleaned={c:pd.to_numeric(frame[c],errors="coerce").replace([np.inf,-np.inf],np.nan) for c in self.columns}
        self.medians={c:float(v.median()) if v.notna().any() else 0. for c,v in cleaned.items()}
        self.group_medians={}
        if self.variant=="conditional" and self.groups:
            clean=frame[self.groups+self.columns].copy()
            for c in self.columns:
                clean[c]=cleaned[c]
            for c in self.groups:
                clean[c]=clean[c].fillna("__MISSING__").astype(str)
            for c in self.columns:
                stats=clean.groupby(self.groups,dropna=False)[c].agg(["median","count"])
                self.group_medians[c]=stats["median"].where(stats["count"]>=20)
        self.pattern_columns=[c for c in self.columns if frame[c].isna().any()][:8]
        return self

    def transform(self,frame):
        result=frame.copy()
        if self.variant=="missing_pattern":
            values=frame[self.columns].apply(pd.to_numeric,errors="coerce")
            result["__row_missing_count"]=values.isna().sum(axis=1).astype(float)
            result["__row_zero_count"]=values.eq(0).sum(axis=1).astype(float)
            pattern=np.zeros(len(frame),dtype=float)
            for i,c in enumerate(self.pattern_columns):
                pattern+=frame[c].isna().to_numpy()*2**i
            result["__missing_pattern"]=pattern
            return result
        for c in self.columns:
            values=pd.to_numeric(frame[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
            result[c+"__was_missing"]=values.isna().astype(float)
            if self.variant=="sentinel":
                result[c+"__was_zero"]=values.eq(0).astype(float)
                result[c]=values.mask(values.eq(0))
            else:
                if self.variant=="conditional" and self.groups:
                    keys=frame[self.groups].copy()
                    for g in self.groups:
                        keys[g]=keys[g].fillna("__MISSING__").astype(str)
                    index=pd.MultiIndex.from_frame(keys) if len(self.groups)>1 else pd.Index(keys[self.groups[0]])
                    fill=self.group_medians[c].reindex(index).to_numpy(dtype=float)
                    fill=np.where(np.isfinite(fill),fill,self.medians[c])
                    result[c]=np.where(values.isna(),fill,values)
                else:
                    result[c]=values.fillna(self.medians[c])
        return result


def conflict_weights(frame,y):
    """P(train-label | exact train-X), без valid и без удаления редких классов."""
    groups=pd.util.hash_pandas_object(frame,index=False).to_numpy()
    temp=pd.DataFrame({"g":groups,"y":y})
    total=temp.groupby("g")["y"].transform("size").to_numpy()
    same=temp.groupby(["g","y"])["y"].transform("size").to_numpy()
    weights=np.clip(same/total,.1,1.)
    return weights/weights.mean(),{"conflicting_rows":int(np.sum(same<total)),"minimum_raw_weight":float(np.min(same/total))}


def macro_f1_metric(y_true,prediction):
    if np.ndim(prediction)==1:
        prediction=np.column_stack([1-prediction,prediction])
    classes=prediction.shape[1]
    matrix=np.bincount(np.asarray(y_true,dtype=int)*classes+prediction.argmax(axis=1),minlength=classes**2).reshape(classes,classes)
    denominator=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(classes,dtype=float),where=denominator>0)
    return "macro_f1",float(f1.mean()),True


def custom_lgb(task,base,profile,train,test,y,tr,va,threads,refit,fixed,deadline,variant):
    import lightgbm as lgb
    started=time.monotonic()
    builder=FeatureBuilder(task,base,profile).fit(train.iloc[tr])
    xtr=builder.transform(train.iloc[tr],categorical_codes=True)
    xva=builder.transform(train.iloc[va],categorical_codes=True) if len(va) else None
    xte=builder.transform(test,categorical_codes=True)
    diagnostics={}
    if variant=="linear_tree":
        numeric=xtr.select_dtypes(include=[np.number]).columns.tolist()
        median=xtr[numeric].median().fillna(0)
        scaler=StandardScaler().fit(xtr[numeric].fillna(median))
        for frame in (xtr,xva,xte):
            if frame is not None and len(frame):
                frame[numeric]=scaler.transform(frame[numeric].fillna(median))
        diagnostics["numeric_preprocessor"]={"columns":numeric,"fit_rows":len(tr)}
    iterations=int(fixed or base.iterations)
    params=dict(n_estimators=iterations,learning_rate=base.learning_rate,num_leaves=base.leaves,
        min_child_samples=base.min_child_samples,reg_lambda=base.regularization,colsample_bytree=.9,
        subsample=.85,subsample_freq=1,random_state=base.seed,n_jobs=threads,verbosity=-1,
        deterministic=True,force_col_wise=True)
    if variant=="linear_tree":
        params["linear_tree"]=True
    if variant=="huber":
        params.update(objective="huber",alpha=.9,metric="rmse")
    if variant=="lgb_f1":
        params["metric"]="None"
    model=(lgb.LGBMClassifier if task.classification else lgb.LGBMRegressor)(**params)
    weight=sample_weights(y[tr],len(task.classes),base.class_weight_power) if task.classification else None
    if variant=="conflict_weight":
        extra,info=conflict_weights(train.iloc[tr][task.feature_columns],y[tr])
        diagnostics.update(info)
        weight=extra if weight is None else weight*extra
        weight=weight/np.mean(weight)
    fit={"sample_weight":weight, **lightgbm_fit_metadata(xtr)}
    stopped=[False]
    def stop(env):
        if time.monotonic()>=deadline:
            stopped[0]=True
            if env.evaluation_result_list and not refit:
                raise lgb.callback.EarlyStopException(env.iteration,env.evaluation_result_list)
            raise TimeoutError("Не завершён фиксированный fit")
    stop.order=5
    callbacks=[stop]
    if len(va) and not refit:
        fit["eval_set"]=[(xva,y[va])]
        callbacks.append(lgb.early_stopping(120,first_metric_only=True,verbose=False))
        if variant=="lgb_f1":
            fit["eval_metric"]=macro_f1_metric
    fit["callbacks"]=callbacks
    model.fit(xtr,y[tr],**fit)
    best=int(model.best_iteration_ or iterations)
    diagnostics["evaluation_names"]={k:list(v) for k,v in getattr(model,"evals_result_",{}).items()}
    trained=time.monotonic()
    def predict(frame):
        if frame is None or not len(frame):
            return np.empty((0,len(task.classes))) if task.classification else np.empty(0)
        if task.classification:
            prediction=np.zeros((len(frame),len(task.classes)))
            prediction[:,np.asarray(model.classes_,dtype=int)]=model.predict_proba(frame)
            return normalized_probabilities(prediction,len(task.classes))
        return np.asarray(model.predict(frame),dtype=float)
    valid,predicted=predict(xva),predict(xte)
    details={"best_iterations":best,"n_train":len(tr),"training_seconds":trained-started,
        "prediction_seconds":time.monotonic()-trained,"feature_count":len(xtr.columns),
        "evaluation_used_for_fit":bool(len(va) and not refit),"training_stopped_for_budget":stopped[0],
        "diagnostics":diagnostics}
    return {"model":model,"builder":builder},valid,predicted,details


def ridge_pipeline(task,frame):
    cats=[c for c in task.categorical_columns if c in frame]
    nums=[c for c in task.feature_columns if c not in cats]
    encoder=ColumnTransformer([
        ("numeric",make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),StandardScaler(with_mean=False)),nums),
        ("categorical",OneHotEncoder(handle_unknown="ignore",min_frequency=3,max_categories=128,sparse_output=True),cats)],sparse_threshold=1.)
    return make_pipeline(encoder,Ridge(alpha=10.,solver="lsqr",tol=1e-5))


def train_proposal(task,config,profile,train,test,y,tr,va,threads,refit=False,fixed_iterations=None,time_limit=600):
    started=time.monotonic();deadline=started+time_limit
    config.validate(task)
    base=parse_experiment(config.base_config,task)
    base=replace(base,seed=config.seed)
    variant=config.variant
    if variant in {"sentinel","median","conditional","missing_pattern"}:
        policy=NumericPolicy(variant,config.columns,config.groups).fit(train.iloc[tr][task.feature_columns])
        xtr=policy.transform(train.iloc[tr][task.feature_columns])
        xva=policy.transform(train.iloc[va][task.feature_columns])
        xte=policy.transform(test[task.feature_columns])
        augmented=train.copy()
        for col in xtr:
            if col not in augmented:
                augmented[col]=np.nan
            augmented.loc[train.index[tr],col]=xtr[col].to_numpy()
            if len(va):
                augmented.loc[train.index[va],col]=xva[col].to_numpy()
        te=test.copy()
        for col in xte:
            te[col]=xte[col].to_numpy()
        augmented_task=replace(task,feature_columns=list(xtr.columns))
        bundle,valid,predicted,details=train_model(augmented_task,base,profile,augmented,te,y,tr,va,threads,refit,fixed_iterations,max(.1,deadline-time.monotonic()))
        bundle["policy"]=policy
        details["diagnostics"]={"policy":variant,"columns":config.columns,"groups":config.groups,"fit_rows":len(tr)}
    elif variant in {"lgb_f1","linear_tree","huber","conflict_weight"}:
        bundle,valid,predicted,details=custom_lgb(task,base,profile,train,test,y,tr,va,threads,refit,fixed_iterations,deadline,variant)
    elif variant=="linear_residual":
        linear=ridge_pipeline(task,train.iloc[tr][task.feature_columns])
        features=train[task.feature_columns].copy()
        for c in task.categorical_columns:
            features[c]=features[c].fillna("__MISSING__").astype(str)
        for c in task.feature_columns:
            if c not in task.categorical_columns:
                features[c]=pd.to_numeric(features[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        linear.fit(features.iloc[tr],y[tr])
        residual=y.copy()
        residual[tr]=y[tr]-linear.predict(features.iloc[tr])
        residual[va]=(y[va]-linear.predict(features.iloc[va])) if len(va) and not refit else 0.
        bundle,valid,predicted,details=train_model(task,base,profile,train,test,residual,tr,va,threads,refit,fixed_iterations,max(.1,deadline-time.monotonic()))
        te=test[task.feature_columns].copy()
        for c in task.categorical_columns:
            te[c]=te[c].fillna("__MISSING__").astype(str)
        for c in task.feature_columns:
            if c not in task.categorical_columns:
                te[c]=pd.to_numeric(te[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        valid+=linear.predict(features.iloc[va]) if len(va) else np.empty(0)
        predicted+=linear.predict(te)
        bundle["linear"]=linear
        details["diagnostics"]={"ridge_alpha":10.,"residual_fit_rows":len(tr)}
    elif variant=="log_smear":
        if np.min(y[tr])<0:
            raise ValueError("log1p требует неотрицательный таргет")
        logbase=replace(base,target_transform="log1p")
        bundle,valid,predicted,details=train_model(task,logbase,profile,train,test,y,tr,va,threads,refit,fixed_iterations,max(.1,deadline-time.monotonic()))
        time_col=profile.get("time_column")
        if not time_col:
            raise ValueError("Для smearing нужна внутренняя временная проверка")
        periods=pd.to_numeric(train.iloc[tr][time_col]).to_numpy()
        latest=np.max(periods)
        inner_tr=tr[periods<latest];inner_va=tr[periods==latest]
        if min(len(inner_tr),len(inner_va))<20:
            raise ValueError("Недостаточно внутренней истории для smearing")
        _,calibrated,_,_=train_model(task,logbase,profile,train,test,y,inner_tr,inner_va,threads,
            True,details["best_iterations"],max(.1,deadline-time.monotonic()))
        factor=float(np.mean((y[inner_va]+1)/np.maximum(calibrated+1,1e-8)))
        if not math.isfinite(factor) or factor<=0:
            raise ValueError("Некорректный smearing factor")
        valid=(valid+1)*factor-1;predicted=(predicted+1)*factor-1
        bundle["smearing_factor"]=factor
        details["diagnostics"]={"smearing_factor":factor,"inner_train":len(inner_tr),"inner_calibration":len(inner_va),"fit_period_max":float(np.max(periods[periods<latest])),"calibration_period":float(latest)}
    else:
        raise ValueError("Неизвестный рецепт")
    details["training_seconds_total"]=time.monotonic()-started
    details["component_fits"] = 2 if variant in {"linear_residual", "log_smear"} else 1
    return bundle,valid,predicted,details


def prior_bias(prediction,prior,strength):
    scores=np.log(np.maximum(prediction,1e-12))-strength*np.log(np.maximum(prior,1e-12))[None,:]
    scores-=scores.max(axis=1,keepdims=True)
    return normalized_probabilities(np.exp(scores),len(prior))


def fit_sigmoid_calibration(prediction,y):
    parameters=[]
    for k in range(prediction.shape[1]):
        label=np.asarray(y)==k
        if len(np.unique(label))<2:
            parameters.append([1.,0.]);continue
        x=logit(np.clip(prediction[:,k],1e-6,1-1e-6)).reshape(-1,1)
        model=LogisticRegression(C=1000.,solver="lbfgs",max_iter=300).fit(x,label.astype(int))
        parameters.append([float(model.coef_[0,0]),float(model.intercept_[0])])
    return parameters


def sigmoid_calibration(prediction,parameters):
    values=logit(np.clip(prediction,1e-6,1-1e-6))
    params=np.asarray(parameters)
    result=expit(values*params[:,0]+params[:,1])
    return normalized_probabilities(result,result.shape[1])
