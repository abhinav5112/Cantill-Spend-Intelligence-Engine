"""Reproduce the A taxonomy-only evaluation with labels withheld from inference."""
import argparse
import pandas as pd
from spend_engine import enrich_text, taxonomy_classifier, classify

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='Cantill_DataML_Task_Kit'); a=p.parse_args()
    root=a.root
    data=pd.read_csv(f'{root}/company_A/erp_goods_receipts.csv',low_memory=False)
    labels=pd.read_csv(f'{root}/company_A/labels_A.csv')
    tree=pd.read_csv(f'{root}/company_A/taxonomy_A.csv')
    # No labels enter vectorization, fitting or prediction.
    texts=enrich_text(data,['material_short_text','service_short_text','material_number','material_type','spend_type','uom','po_uom','opex_capex'])
    predictions,_=classify(texts,taxonomy_classifier(tree))
    actual=labels[['category_l1','category_l2','category_l3']].fillna('').astype(str).apply(lambda r:tuple(x for x in r if x and x!='nan'),axis=1).tolist()
    print(f'L1 accuracy: {sum(x[0]==y[0] for x,y in zip(actual,predictions))/len(actual):.4f}')
    print(f'L1+L2 accuracy: {sum(tuple(x[:2])==tuple(y[:2]) for x,y in zip(actual,predictions))/len(actual):.4f}')
    print(f'Leaf accuracy: {sum(x==y for x,y in zip(actual,predictions))/len(actual):.4f}')
if __name__=='__main__': main()