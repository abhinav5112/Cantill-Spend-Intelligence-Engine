#!/usr/bin/env python3
"""Config-driven procurement spend normalization and taxonomy classification."""
from __future__ import annotations
import argparse, json, re, unicodedata, hashlib, warnings, joblib
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from rapidfuzz import process, fuzz
warnings.filterwarnings('ignore')

EXCLUSIONS={'CANCELLED','NOT_APPROVED','NON_PROCUREMENT','DUPLICATE'}

def norm(x):
    if pd.isna(x): return ''
    s=unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+',' ',s).strip()
def digits(x): return re.sub(r'\D','',str(x or ''))
def money(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().replace(' ','')
    if not s: return np.nan
    # handle common decimal/thousands conventions
    if ',' in s and '.' in s:
        if s.rfind(',')>s.rfind('.'): s=s.replace('.','').replace(',','.')
        else: s=s.replace(',','')
    elif ',' in s:
        tail=s.split(',')[-1]
        s=s.replace(',','.' if len(tail) in (1,2) else '')
    try: return float(re.sub(r'[^0-9.\-]','',s))
    except: return np.nan
def vat_fraction(x, default=.15):
    if pd.isna(x) or str(x).strip()=='': return default
    s=str(x).strip().replace('%','')
    try:
        v=float(s); return v/100 if v>1 else v
    except: return default
def stable_key(prefix, value):
    return prefix+'-'+hashlib.md5(norm(value).encode()).hexdigest()[:10]
def read_csv(path): return pd.read_csv(path,low_memory=False,dtype=str,keep_default_na=True)
def category_paths(tree):
    cols=[c for c in tree.columns if c.lower().startswith('category_') or c.lower() in ('l1','l2','l3','l4')]
    if not cols: cols=list(tree.columns)
    return tree[cols].fillna('').astype(str).apply(lambda r: tuple(x.strip() for x in r if x and x.lower()!='nan'),axis=1).drop_duplicates().tolist()

def taxonomy_classifier(tree, train_text=None, train_labels=None):
    paths = category_paths(tree)
    labels = [' > '.join(p) for p in paths]

    if train_text is not None and train_labels is not None and len(set(train_labels)) > 1:
        vec = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=3,
            max_features=12000,
            sublinear_tf=True,
            strip_accents='unicode'
        )

        X = vec.fit_transform(train_text)
        classes = sorted(set(train_labels))
        centroids = []

        for label in classes:
            ids = [i for i, y in enumerate(train_labels) if y == label]
            centroids.append(
                np.asarray(X[ids].mean(axis=0)).ravel()
            )

        centroids = np.vstack(centroids)

        # Normalize category centroids
        centroid_norms = np.linalg.norm(
            centroids, axis=1, keepdims=True
        )

        centroids = np.divide(
            centroids,
            centroid_norms,
            out=np.zeros_like(centroids),
            where=centroid_norms > 0
        )

        return ('supervised', vec, (classes, centroids), paths)

    # Zero-shot taxonomy classification
    corpus = [' '.join(p) for p in paths]

    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer='word',
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode'
    )

    mat = vec.fit_transform(corpus)

    return ('zero', vec, mat, paths)

def classify(texts, model):
    mode,vec,obj,paths=model
    q=vec.transform(texts)
    if mode=='supervised':
        classes,centroids=obj
        sims=q @ centroids.T
        idx=np.asarray(sims.argmax(axis=1)).ravel(); conf=np.asarray(sims.max(axis=1)).ravel()
        class_to_path={ ' > '.join(p):p for p in paths }
        return [class_to_path.get(classes[i],tuple(classes[i].split(' > '))) for i in idx], np.clip(conf,0,1)
    sims=cosine_similarity(q,obj)
    idx=sims.argmax(axis=1); conf=sims.max(axis=1)
    return [paths[i] for i in idx], np.clip(conf,0,1)
def enrich_text(df, fields):
    return df.reindex(columns=fields).fillna('').astype(str).apply(lambda r:' '.join(norm(x) for x in r if norm(x)),axis=1).tolist()
def classify_rows(df, tree, text_fields, label_df=None, label_id=None, label_cols=None):
    texts=enrich_text(df,text_fields)
    if label_df is not None and label_id and label_cols:
        lab=label_df.dropna(subset=label_cols,how='all').copy()
        lab['__path']=lab[label_cols].fillna('').astype(str).apply(lambda r:' > '.join(x for x in r if x and x!='nan'),axis=1)
        lab=lab[lab['__path'].isin([' > '.join(p) for p in category_paths(tree)])]
        byid={str(r[label_id]):r['__path'] for _,r in lab.iterrows()}
        train_idx=[i for i,x in enumerate(df[label_id].astype(str)) if x in byid]
        if train_idx:
            model=taxonomy_classifier(tree,[texts[i] for i in train_idx],[byid[str(df.iloc[i][label_id])] for i in train_idx])
            return classify(texts,model)
    return classify(texts,taxonomy_classifier(tree))
def match_supplier_keys(names, ids=None):
    names=[norm(x) for x in names]; ids=ids or ['']*len(names)
    unique=list(dict.fromkeys(n for n in names if n)); resolved={}
    for n in unique:
        hit=process.extractOne(n,list(resolved.keys()),scorer=fuzz.ratio) if resolved else None
        resolved[n]=resolved[hit[0]] if hit and hit[1]>=96 else stable_key('SUP',n)
    keys=[]
    for n,ident in zip(names,ids):
        keys.append(resolved.get(n,stable_key('SUP',str(ident))) if n else stable_key('SUP',str(ident)))
    return keys
def item_keys(codes, descriptions):
    keys=[]; seen={}
    for c,d in zip(codes,descriptions):
        nc=norm(c); nd=norm(d)
        if nc and nc not in ('nan','none'): k=stable_key('ITEM','code:'+nc)
        elif nd:
            k=stable_key('ITEM','desc:'+nd)
        else: k=stable_key('ITEM','unknown')
        keys.append(k)
    return keys
def exclusion(status='', text='', supplier='', account='', amount=None, doc=''):
    s=norm(status); d=norm(doc)
    if any(x in s for x in ('cancel','void')) or d in ('ca','cancel','cancelled'): return 'CANCELLED'
    if s and not any(x in s for x in ('approved','paid')): return 'NOT_APPROVED'
    blob=' '.join([norm(text),norm(supplier),norm(account),d])
    patterns=[r'intercompany',r'inter company',r'royalt',r'tax payment',r'vat recover',r'withholding tax',r'corporate income tax',r'interest expense',r'bank charge',r'financing',r'dividend',r'revenue',r'rebate received',r'partner billing',r'affiliate charge']
    if any(re.search(p,blob) for p in patterns): return 'NON_PROCUREMENT'
    if any(x in norm(account) for x in ['ic payable','intercompany','tax payable','royalt','interest expense']): return 'NON_PROCUREMENT'
    return ''
def run_a(cfg, root, outdir):
    erp=read_csv(root/cfg['erp_file']); tool=read_csv(root/cfg['tool_file']); tree=read_csv(root/cfg['taxonomy_file']); labels=read_csv(root/cfg['labels_file']) if cfg.get('labels_file') else None
    # Match tool records to ERP receipts via PO, amount (net), supplier similarity and text.
    erp['_po']=erp.po_number.map(digits); erp['_amt']=erp.value_zar.map(money)
    tool['_po']=tool.po_reference.map(digits); tool['_gross']=tool.gross_amount.map(money)
    tool['_vat']=tool.vat_rate.map(vat_fraction); tool['_net']=tool._gross/(1+tool._vat)
    erp['_supplier_norm']=erp.supplier_name.map(norm); tool['_supplier_norm']=tool.supplier_name_as_captured.map(norm)
    erp['_text']=enrich_text(erp,['material_short_text','service_short_text','material_number'])
    tool['_text']=enrich_text(tool,['line_text'])
    matched=set(); dup_tool=set(); tool_match={}
    po_index={}
    for j,p in enumerate(erp._po):
        if p: po_index.setdefault(p,[]).append(j)
    # only use robust PO candidate matching; score with supplier and amount.
    for ti,row in tool.iterrows():
        po=row['_po']; candidates=[]
        if po:
            candidates.extend(po_index.get(po, []))

        best=None; bestscore=-1
        for ei in candidates:
            er=erp.iloc[ei]
            ss=SequenceMatcher(None,row['_supplier_norm'],er['_supplier_norm']).ratio() if row['_supplier_norm'] and er['_supplier_norm'] else 0
            amt=abs(row['_net']-er['_amt'])/max(abs(er['_amt']),1) if pd.notna(row['_net']) and pd.notna(er['_amt']) else 9
            ts=SequenceMatcher(None,row['_text'][:150],er['_text'][:150]).ratio()
            score=(.5*(1-min(amt,1)) + .3*ss + .2*ts)
            if (amt<=.03 and (ss>=.45 or ts>=.5) or amt<=.005 and (ss>=.25 or ts>=.3)) and score>bestscore:
                best=ei; bestscore=score
        if best is not None: dup_tool.add(ti); matched.add(best); tool_match[ti]=best
    # Tool invoices duplicated within export: invoice + text + normalized net amount.
    seen=set()
    for ti,row in tool.iterrows():
        sig=(norm(row.invoice_number),norm(row.line_text),round(row['_net'],2) if pd.notna(row['_net']) else None)
        if sig in seen: dup_tool.add(ti)
        else: seen.add(sig)
    erp_text=erp['_text'].tolist(); tool_text=tool['_text'].tolist()
    # supervised category model from ERP labels, labels are aligned via record id.
    catmap={str(r.record_id):[r.get('category_l1',''),r.get('category_l2',''),r.get('category_l3','')] for _,r in labels.iterrows()} if labels is not None else {}
    label_frame=labels.copy() if labels is not None else None
    erp_ids=['ERP:'+str(x) for x in erp.line_id]
    # train on label rows; use direct text and extra operational metadata



    model_path = Path(cfg.get('model_file', 'models/company_A.joblib'))

    cat_train = labels.copy() if labels is not None else None

    if cat_train is not None:
        cat_train = cat_train[
            cat_train['category_l1'].notna() &
            (cat_train['category_l1'].astype(str).str.strip() != '')
        ].copy()

        text_lookup = dict(zip(erp_ids, erp_text))
        cat_train['__text'] = cat_train['record_id'].astype(str).map(text_lookup)
        cat_train = cat_train.dropna(subset=['__text'])

    if cat_train is not None and len(cat_train) > 10:
        paths = ['category_l1', 'category_l2', 'category_l3']

        model = taxonomy_classifier(
            tree,
            cat_train['__text'].tolist(),
            cat_train[paths].fillna('').astype(str).apply(
                lambda r: ' > '.join(
                    x for x in r if x and x != 'nan'
                ), axis=1
            ).tolist()
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    elif model_path.exists():
        model = joblib.load(model_path)

    else:
        model = taxonomy_classifier(tree)
    epreds,econf=classify(erp_text,model)
    tpreds,tconf=classify(tool_text,model)
    records=[]
    sup_keys=match_supplier_keys(erp.supplier_name.tolist(),erp.vendor_id.tolist())
    item_ids=item_keys(erp.material_number.tolist(),erp['_text'].tolist())
    for i,r in erp.iterrows():
        reason=exclusion('',r['_text'],r.supplier_name,r.cost_centre,r['_amt'])
        amt=r['_amt'] if pd.notna(r['_amt']) else 0
        p=epreds[i]
        records.append(dict(record_id='ERP:'+str(r.line_id),include=0 if reason else 1,exclusion_reason=reason,supplier_key=sup_keys[i],item_key=item_ids[i],spend=0 if reason else amt,l1=p[0] if len(p)>0 else '',l2=p[1] if len(p)>1 else '',l3=p[2] if len(p)>2 else '',l4='',confidence=float(econf[i])))
    # Tool supplier matching to known ERP suppliers using fuzzy name; use common golden key where close.
    erp_names=list(dict.fromkeys(norm(x) for x in erp.supplier_name.tolist()))
    name_to_key={}
    for j,n in enumerate(norm(x) for x in erp.supplier_name.tolist()):
        if n: name_to_key.setdefault(n,sup_keys[j])
    for i,r in tool.iterrows():
        reason=exclusion(r.status,r.line_text,r.supplier_name_as_captured,'',r['_net'])
        if i in dup_tool: reason='DUPLICATE'
        sn=norm(r.supplier_name_as_captured); best=-1; key=None
        hit=process.extractOne(sn,erp_names,scorer=fuzz.ratio) if sn and erp_names else None
        if hit:
            best=hit[1]/100; key=name_to_key.get(hit[0])
        if best<.90: key=stable_key('SUP',sn)
        ik=item_ids[tool_match[i]] if i in tool_match else stable_key('ITEM','desc:'+norm(r.line_text))
        p=tpreds[i]
        records.append(dict(record_id='TOOL:'+str(r.tool_record_id),include=0 if reason else 1,exclusion_reason=reason,supplier_key=key,item_key=ik,spend=0 if reason else (float(r['_net']) if pd.notna(r['_net']) else 0),l1=p[0] if len(p)>0 else '',l2=p[1] if len(p)>1 else '',l3=p[2] if len(p)>2 else '',l4='',confidence=float(tconf[i])))
    out=pd.DataFrame(records); out['spend']=pd.to_numeric(out.spend).round(2); out['confidence']=pd.to_numeric(out.confidence).clip(0,1).round(4)
    out.to_csv(outdir/cfg.get('output_name','abhinav__A.csv'),index=False)
    return out, {'erp_rows':len(erp),'tool_rows':len(tool),'tool_matched_or_duped':len(dup_tool),'included_spend':out.loc[out.include==1,'spend'].sum()}
def run_b(cfg, root, outdir):
    df=read_csv(root/cfg['invoice_file']); tree=read_csv(root/cfg['taxonomy_file']); fx=read_csv(root/cfg['fx_file']); labels=read_csv(root/cfg['labels_file']) if cfg.get('labels_file') else None
    # Normalize currencies and convert to USD; fx rates are currency units per USD.
    df['_amount']=df.line_amount.map(money); df['_date']=pd.to_datetime(df.posting_date,errors='coerce')
    fx['month']=fx.month.astype(str); fx['units_per_usd']=fx.units_per_usd.map(money)
    fxmap={(r.currency,r.month):r.units_per_usd for _,r in fx.iterrows()}
    usd=[]
    for _,r in df.iterrows():
        cur=str(r.currency).upper(); month=r['_date'].strftime('%Y-%m') if pd.notna(r['_date']) else ''
        rate=money(r.currency_rate)
        if cur=='USD': rate=1
        if not rate or pd.isna(rate) or rate<=0: rate=fxmap.get((cur,month),np.nan)
        usd.append(r['_amount']/rate if pd.notna(r['_amount']) and pd.notna(rate) and rate>0 else 0)
    df['_usd']=usd
    texts=enrich_text(df,['line_description','material_number','gl_account_description','gl_account','cost_centre','supplier_type','document_type','po_type'])
    # labels train the supervised classifier; include only labelled spend rows with a complete category path.
    train=None
    if labels is not None:
        label_lookup=dict(zip(labels.line_id,labels.include.astype(str)))
        lab=labels[(labels.include.astype(str)=='1') & labels.category_l1.notna()].copy()
        text_lookup=dict(zip(df.line_id,texts)); lab['__text']=lab.line_id.map(text_lookup)
        train=lab
    model_path=Path(cfg.get('model_file','models/company_B.joblib'))
    if train is not None and len(train)>10 and train.__text.notna().sum()>10:
        paths=['category_l1','category_l2','category_l3']
        train=train.dropna(subset=['__text'])
        model=taxonomy_classifier(tree,train.__text.tolist(),train[paths].fillna('').astype(str).apply(lambda r:' > '.join(x for x in r if x and x!='nan'),axis=1).tolist())
        model_path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,model_path)
    elif model_path.exists(): model=joblib.load(model_path)
    else: model=taxonomy_classifier(tree)
    preds,conf=classify(texts,model)
    # Learn account-level exclusion rules from analyst seed labels; require repeated high-purity evidence.
    excluded_accounts=set(); excluded_descriptions=set()
    rules_path=Path(cfg.get('rules_file','models/company_B_rules.json'))
    if labels is not None:
        seed=labels.merge(df[['line_id','gl_account','gl_account_description']],on='line_id',how='inner')
        if len(seed):
            seed['include']=pd.to_numeric(seed['include'],errors='coerce')
            for col,target in [('gl_account',excluded_accounts),('gl_account_description',excluded_descriptions)]:
                stats=seed.groupby(col,dropna=True).include.agg(['mean','count'])
                target.update(str(k) for k,v in stats.iterrows() if v['count']>=2 and v['mean']<=.10)
            rules_path.parent.mkdir(parents=True,exist_ok=True)
            json.dump({'excluded_accounts':sorted(excluded_accounts),'excluded_descriptions':sorted(excluded_descriptions)},open(rules_path,'w'),indent=2)
    if not excluded_accounts and not excluded_descriptions and rules_path.exists():
        saved=json.load(open(rules_path)); excluded_accounts=set(saved.get('excluded_accounts',[])); excluded_descriptions=set(saved.get('excluded_descriptions',[]))
    out=[]
    for i,r in df.iterrows():
        reason=exclusion('',r.line_description,r.vendor_id,r.gl_account_description,r['_usd'],r.document_type)
        if not reason and (str(r.gl_account) in excluded_accounts or str(r.gl_account_description) in excluded_descriptions):
            reason='NON_PROCUREMENT'
        if not reason and re.search(r'\b(intercompany|inter company|royalt|tax payable|interest expense|financing)\b',norm(str(r.gl_account_description or '')+' '+str(r.gl_account or ''))):
            reason='NON_PROCUREMENT'
        p=preds[i]; key=stable_key('SUP',r.vendor_id); ik=stable_key('ITEM',r.material_number if norm(r.material_number) else r.line_description)
        out.append(dict(record_id=str(r.line_id),include=0 if reason else 1,exclusion_reason=reason,supplier_key=key,item_key=ik,spend=0 if reason else r['_usd'],l1=p[0] if len(p)>0 else '',l2=p[1] if len(p)>1 else '',l3=p[2] if len(p)>2 else '',l4='',confidence=float(conf[i])))
    out=pd.DataFrame(out); out['spend']=pd.to_numeric(out.spend).fillna(0).round(2); out['confidence']=pd.to_numeric(out.confidence).clip(0,1).round(4)
    out.to_csv(outdir/cfg.get('output_name','abhinav__B.csv'),index=False)
    return out, {'rows':len(df),'included_spend_usd':out.loc[out.include==1,'spend'].sum(),'excluded':int((out.include==0).sum())}
def run_c(cfg, root, outdir):
    df=read_csv(root/cfg['invoice_file']); tree=read_csv(root/cfg['taxonomy_file'])
    fields=cfg.get('description_columns',[])
    texts=enrich_text(df,fields)
    date_col=cfg.get('date_column'); currency_col=cfg.get('currency_column'); rate_col=cfg.get('currency_rate_column')
    fxmap={}
    if cfg.get('fx_file'):
        fx=read_csv(root/cfg['fx_file']); fx['units_per_usd']=fx['units_per_usd'].map(money)
        fxmap={(str(r.currency).upper(),str(r.month)):r.units_per_usd for _,r in fx.iterrows()}
    amounts=df[cfg['amount_column']].map(money); spends=[]
    dates=pd.to_datetime(df[date_col],errors='coerce') if date_col else pd.Series([pd.NaT]*len(df))
    for i,r in df.iterrows():
        amount=amounts.iloc[i]; cur=str(r[currency_col]).upper() if currency_col else 'USD'
        rate=money(r[rate_col]) if rate_col else np.nan
        if cur=='USD': rate=1
        if (pd.isna(rate) or rate<=0) and cur!='USD' and fxmap and pd.notna(dates.iloc[i]): rate=fxmap.get((cur,dates.iloc[i].strftime('%Y-%m')),np.nan)
        spends.append(amount/rate if pd.notna(amount) and pd.notna(rate) and rate>0 else 0)
    preds,conf=classify(texts,taxonomy_classifier(tree))
    out=[]
    for i,r in df.iterrows():
        status=str(r[cfg['status_column']]) if cfg.get('status_column') in df.columns else ''
        supplier=str(r[cfg['supplier_column']]) if cfg.get('supplier_column') in df.columns else ''
        account=str(r[cfg['account_column']]) if cfg.get('account_column') in df.columns else ''
        doc=str(r[cfg['document_type_column']]) if cfg.get('document_type_column') in df.columns else ''
        reason=exclusion(status,texts[i],supplier,account,spends[i],doc)
        code=str(r[cfg['item_code_column']]) if cfg.get('item_code_column') in df.columns else ''
        p=preds[i]
        out.append(dict(record_id=str(r[cfg.get('id_column','line_id')]),include=0 if reason else 1,exclusion_reason=reason,
            supplier_key=stable_key('SUP',str(r[cfg['supplier_id_column']])) if cfg.get('supplier_id_column') in df.columns else stable_key('SUP',supplier),
            item_key=stable_key('ITEM',code if norm(code) else texts[i]),spend=0 if reason else spends[i],
            l1=p[0] if len(p)>0 else '',l2=p[1] if len(p)>1 else '',l3=p[2] if len(p)>2 else '',l4=p[3] if len(p)>3 else '',confidence=float(conf[i])))
    out=pd.DataFrame(out); out['spend']=pd.to_numeric(out.spend).fillna(0).round(2); out['confidence']=pd.to_numeric(out.confidence).clip(0,1).round(4)
    out.to_csv(outdir/cfg.get('output_name','abhinav__C.csv'),index=False)
    return out,{'rows':len(df),'included_spend_usd':out.loc[out.include==1,'spend'].sum(),'excluded':int((out.include==0).sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output-dir',default='outputs'); ap.add_argument('--root',default='Cantill_DataML_Task_Kit')
    args=ap.parse_args(); cfg=json.load(open(args.config)); root=Path(args.root); outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    if cfg['company']=='A': out,stats=run_a(cfg,root,outdir)
    elif cfg['company']=='B': out,stats=run_b(cfg,root,outdir)
    elif cfg['company']=='C': out,stats=run_c(cfg,root,outdir)
    else: raise ValueError('company must be A, B or C')
    print(json.dumps(stats,indent=2)); print('Wrote',len(out),'records to',outdir)
if __name__=='__main__': main()
