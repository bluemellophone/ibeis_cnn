
OLD COMMAND LINE:
    TrainingCommandLine:
        python -m ibeis_cnn.train --test-pz_patchmatch --train
        python -m ibeis_cnn.train --test-pz_patchmatch --vtd --max_examples=3 --learning_rate .0000001 --train
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --nocache-train

        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --train
        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_Master0 --train

    TestingCommandLine:
        python -m ibeis_cnn.train --test-pz_patchmatch --test='current'
        python -m ibeis_cnn.train --test-pz_patchmatch --vtd

        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --num-top=20
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --vtd

        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --test --weights=current
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --test --weights=nnp
        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_MTEST --test --weights=nnp
        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_MTEST --test --weights=nnp --checkpoint=11
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --test --weights=nnp --checkpoint=12
        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_MTEST --test --weights=new
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --test --weights=new

        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --test --weights=liberty
        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_MTEST --test --weights=liberty --checkpoint=lib30

        python -m ibeis_cnn.train --test-pz_patchmatch --db liberty --test
        python -m ibeis_cnn.train --test-pz_patchmatch --db liberty --test --checkpoint=hist_eras1_epochs14_mzdgzqtjprzddqie

        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_Master0 --test --checkpoint master21
        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_Master0 --test --checkpoint master51

        python -m ibeis_cnn.train --test-pz_patchmatch --db PZ_MTEST --weights=pzmaster --test --checkpoint master21

        python -m ibeis_cnn.train --test-pz_patchmatch --db mnist --weights=new --train
        python -m ibeis_cnn.train --test-pz_patchmatch --db mnist --weights=current --test

        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --weights=new --train --arch=siaml2
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --weights=new --train --arch=siaml2 --num_top=None

        python -m ibeis_cnn.train --test-pz_patchmatch --db nnp3-2 --arch=siaml2 --weights=new  --train
        python -m ibeis_cnn.train --test-pz_patchmatch --db NNP_Master3 --num_top=None --arch=siaml2 --weights=current  --test

