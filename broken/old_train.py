# Build Dataset Aliases
python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --colorspace='gray' --num-top=None --controlled=True --aliasexit
python -m ibeis_cnn --tf pz_patchmatch --db liberty --colorspace='gray' --aliasexit
python -m ibeis_cnn --tf pz_patchmatch --db PZ_MTEST --colorspace='gray' --num-top=None --controlled=True --aliasexit
python -m ibeis_cnn --tf pz_patchmatch --db NNP_Master3 --colorspace='gray' --num-top=None --controlled=True --aliasexit
python -m ibeis_cnn --tf pz_patchmatch --db GZ_ALL --colorspace='gray' --num-top=None --controlled=True --aliasexit
python -m ibeis_cnn --tf pz_patchmatch --db NNP_MasterGIRM_core --colorspace='gray' --num-top=None --controlled=True --aliasexit

# --- TRAINING ---

# Train liberty
python -m ibeis_cnn --tf pz_patchmatch --ds liberty --weights=new --arch=siaml2_128 --train --monitor

# Train NNP_Master
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=nnp3-2:epochs0011 --arch=siaml2 --train --monitor
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor

python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor

# Train COMBO
python -m ibeis_cnn --tf pz_patchmatch --ds combo --weights=new --arch=siaml2_128 --train --monitor

# Train Liberty (Make sure that our structures continue to work on liberty data)
python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=new --arch=siaml2_128 --train --monitor

# Continue training
python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=current --arch=siaml2_128 --train --monitor --learning-rate=.03
python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=current --arch=siaml2_128 --test

# --- MONITOR TRAINING ---

# Hyperparameter settings
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor

# Grevys
python -m ibeis_cnn --tf pz_patchmatch --ds gz-gray --weights=new --arch=siaml2 --train --weight_decay=0.0001 --monitor

# THIS DID WELL VERY QUICKLY
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor --learning_rate=.1 --weight_decay=0.0005
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor --DEBUG_AUGMENTATION

python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2 --train --monitor

python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2 --train --monitor
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siam2streaml2 --train --monitor

python -m ibeis_cnn --tf pz_patchmatch --db NNP_Master3 --weights=new --arch=siaml2 --train --monitor --colorspace='bgr' --num_top=None

python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2 --test
python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2 --test
--monitor --colorspace='bgr' --num_top=None

# --- INITIALIZED-TRAINING ---
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --arch=siaml2 --weights=gz-gray:current --train --monitor

# --- TESTING ---
python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=liberty:current --arch=siaml2_128 --test

# test combo
python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --weights=combo:hist_eras007_epochs0098_gvmylbm --arch=siaml2_128 --testall
python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --weights=combo:current --arch=siaml2_128 --testall

python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=liberty:current --arch=siaml2_128 --test

python -m ibeis_cnn --tf pz_patchmatch --ds gz-gray --arch=siaml2 --weights=gz-gray:current --test
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --arch=siaml2 --weights=gz-gray:current --test
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --arch=siaml2 --weights=gz-gray:current --test
python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster --arch=siaml2 --weights=gz-gray:current --test

# Test NNP_Master on in sample
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=current --arch=siaml2 --test
python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=nnp3-2 --arch=siaml2 --test

# Test NNP_Master3 weights on out of sample data
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0011 --arch=siaml2 --test

# Now can use the alias
python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2 --test
python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster --weights=nnp3-2:epochs0021 --arch=siaml2 --test

