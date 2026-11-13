import json
from pathlib import Path
import numpy as np
import pandas as pd
from matched_data import save,sha

def load_metadata(rows,anatomy,out):
    path=anatomy/'data/raw/phenotypic.csv'
    if not path.is_file():raise ValueError('Missing previous phenotype download: '+str(path))
    source=pd.read_csv(path,usecols=['SUB_ID','FILE_ID','AGE_AT_SCAN','SEX'])
    source=source[source.FILE_ID!='no_filename'].copy();source['number']=source.SUB_ID.map(lambda x:int(float(x)))
    if source.number.duplicated().any():raise ValueError('Duplicate phenotype IDs')
    source=source.set_index('number')
    original={r['subject_id']:r for r in json.loads((anatomy/'data/cohort.json').read_text())}
    frame=pd.read_csv(anatomy/'data/anatomy.csv').set_index('subject_id')
    icv_name='EstimatedTotalIntraCranialVol'
    if icv_name not in frame:raise ValueError('Expected verified ICV measurement missing')
    meta=[];audit=[]
    for r in rows:
        sid=r['subject_id'];number=int(sid.split('::')[1]);p=source.loc[number]
        age=float(p.AGE_AT_SCAN);sex=float(p.SEX);icv=float(frame.loc[sid,icv_name])
        if not np.isfinite([age,sex,icv]).all() or age<=0 or age>110 or sex not in (1.,2.) or icv<=0:raise ValueError('Invalid age/sex/ICV '+sid)
        old=original[sid]
        if old.get('FILE_ID') and old['FILE_ID']!=p.FILE_ID:raise ValueError('Phenotype FILE_ID differs '+sid)
        if old.get('age_months') is not None and abs(float(old['age_months'])/12-age)>1.:raise ValueError('Phenotype age differs '+sid)
        values=[age,float(sex==1),np.log1p(icv),float(r['mean_fd']),np.log(float(r['usable_frames'])),float(r['tensor_tr'])]
        if not np.isfinite(values).all():raise ValueError('Invalid confound fields '+sid)
        meta.append(values)
        audit.append(dict(subject_id=sid,partition=r['partition'],site_id=r['site_id'],asd_true=r['asd_true'],age_years=age,sex_abide=int(sex),icv_mm3=icv,mean_fd=float(r['mean_fd']),usable_frames=int(r['usable_frames']),tensor_tr=float(r['tensor_tr']),low_motion_mean_fd_le_0p5=float(r['mean_fd'])<=.5,missing_anatomy_measures=int(frame.loc[sid].isna().sum()),anatomical_spatial_qc='not certified'))
    audit=pd.DataFrame(audit);audit.to_csv(out/'metadata_qc_inventory.csv',index=False)
    summary=[]
    for (partition,site,group),part in audit.groupby(['partition','site_id','asd_true']):
        summary.append(dict(partition=partition,site_id=site,asd_true=int(group),n=len(part),mean_age=float(part.age_years.mean()),mean_fd=float(part.mean_fd.mean()),n_low_motion=int(part.low_motion_mean_fd_le_0p5.sum())))
    pd.DataFrame(summary).to_csv(out/'metadata_by_site_group.csv',index=False)
    save(out/'metadata_contract.json',dict(source=str(path),sha256=sha(path),fields=['AGE_AT_SCAN','SEX','EstimatedTotalIntraCranialVol','mean_fd','usable_frames','tensor_tr','site_id'],diagnostic_assessments_used=False,iq_medication_comorbidity_used=False,low_motion_threshold=.5,qc_limit='Mean FD threshold is a prespecified sensitivity analysis, not complete motion or anatomical QC. No outcome-based exclusion.'))
    return np.asarray(meta,float)
