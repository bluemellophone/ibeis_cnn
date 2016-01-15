python -m ibeis_cnn --tf netrun --db PZ_Master1 --acfg default:is_known=True,qmin_pername=2,view=primary,species=primary,minqual=ok --ensuredata --vtd
python -m ibeis_cnn --tf netrun --db PZ_Master1 --acfg default:is_known=True,qmin_pername=2,view=primary,species=primary,minqual=ok --weights=new --arch=siaml2_128 --train --monitor
