
r"""
    RENAME:
        patchmatch?

    CommandLine:
        THEANO_FLAGS='device=gpu1'

        # --- UTILITY
        python -m ibeis_cnn --tf get_juction_dpath --show

        # --- LIBERTY EXAMPLES ---

        # Build / Ensure Liberty dataset
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --ensuredata --colorspace='gray'

        # Train on liberty dataset
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --train --weights=new --arch=siaml2_128 --monitor

        # Continue liberty training using previous learned weights
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --train --weights=current --arch=siaml2_128 --monitor --learning-rate=.03

        # Test liberty accuracy
        python -m ibeis_cnn --tf pz_patchmatch --ds liberty --test --weights=liberty:current --arch=siaml2_128 --test

        # Initialize a second database using IBEIS
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_MTEST --colorspace='gray' --num-top=None --controlled=True --ensuredata

        # Test accuracy of another dataset using weights from liberty
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --test --weights=liberty:current --arch=siaml2_128 --testall  # NOQA


        # --- DATASET BUILDING ---

        # Build Dataset Aliases
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_MTEST --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db NNP_Master3 --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db GZ_ALL --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db NNP_MasterGIRM_core --colorspace='gray' --num-top=None --controlled=True --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master1 --acfg_name timectrl --ensuredata
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master1 --acfg_name timectrl:pername=None --ensuredata

        # --- TRAINING ---

        python -m ibeis_cnn --tf pz_patchmatch --ds timectrl_pzmaster1 --train --weights=new --arch=siaml2_128  --monitor  # NOQA
        python -m ibeis_cnn --tf pz_patchmatch --ds timectrl_pzmaster1 --train --weights=new --arch=siaml2_128  --monitor  --learning_rate=.01 --weight_decay=0.005 # NOQA

        # Train NNP_Master
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=nnp3-2:epochs0011 --arch=siaml2_128 --train --monitor
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2_128 --train --weight_decay=0.0001 --monitor
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2_128 --train --weight_decay=0.0001 --monitor

        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor

        # Train COMBO
        python -m ibeis_cnn --tf pz_patchmatch --ds combo --weights=new --arch=siaml2_128 --train --monitor

        # Hyperparameter settings
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor

        # Grevys
        python -m ibeis_cnn --tf pz_patchmatch --ds gz-gray --weights=new --arch=siaml2_128 --train --weight_decay=0.0001 --monitor

        # THIS DID WELL VERY QUICKLY
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor --learning_rate=.1 --weight_decay=0.0005
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor --DEBUG_AUGMENTATION

        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=new --arch=siaml2_128 --train --monitor

        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siaml2_128 --train --monitor
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=new --arch=siam2streaml2 --train --monitor

        python -m ibeis_cnn --tf pz_patchmatch --db NNP_Master3 --weights=new --arch=siaml2_128 --train --monitor --colorspace='bgr' --num_top=None

        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2_128 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster-bgr --weights=nnp3-2-bgr:epochs0023_rttjuahuhhraphyb --arch=siaml2_128 --test
        --monitor --colorspace='bgr' --num_top=None

        # --- INITIALIZED-TRAINING ---
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --arch=siaml2_128 --weights=gz-gray:current --train --monitor

        # --- TESTING ---
        python -m ibeis_cnn --tf pz_patchmatch --db liberty --weights=liberty:current --arch=siaml2_128 --test

        # test combo
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --weights=combo:hist_eras007_epochs0098_gvmylbm --arch=siaml2_128 --testall
        python -m ibeis_cnn --tf pz_patchmatch --db PZ_Master0 --weights=combo:current --arch=siaml2_128 --testall

        python -m ibeis_cnn --tf pz_patchmatch --ds gz-gray --arch=siaml2_128 --weights=gz-gray:current --test
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --arch=siaml2_128 --weights=gz-gray:current --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --arch=siaml2_128 --weights=gz-gray:current --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster --arch=siaml2_128 --weights=gz-gray:current --test

        # Test NNP_Master on in sample
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=current --arch=siaml2_128 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds nnp3-2 --weights=nnp3-2 --arch=siaml2_128 --test

        # Test NNP_Master3 weights on out of sample data
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2_128 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0011 --arch=siaml2_128 --test

        # Test liberty on timectrl PZ_Master1
        python -m ibeis_cnn --tf pz_patchmatch --ds timectrl_pzmaster1 --testall --weights=liberty:current --arch=siaml2_128  # NOQA

        # Now can use the alias
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmtest --weights=nnp3-2:epochs0021 --arch=siaml2_128 --test
        python -m ibeis_cnn --tf pz_patchmatch --ds pzmaster --weights=nnp3-2:epochs0021 --arch=siaml2_128 --test

    Ignore:
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/ibeis_cnn_weights.pickle'
        weights_fpath = '/media/raid/work/NNP_Master3/_ibsdb/_ibeis_cache/nets/train_patchmetric((2462)&ju%uw7bta19cunw)/arch_d788de3571330d42/training_state.cPkl'

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> pz_patchmatch()
        >>> ut.show_if_requested()
"""


#def train_mnist():
#    r"""
#    CommandLine:
#        python -m ibeis_cnn.train --test-train_mnist

#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis_cnn.train import *  # NOQA
#        >>> result = train_mnist()
#        >>> print(result)
#    """
#    hyperparams = ut.argparse_dict(
#        {
#            'batch_size': 128,
#            'learning_rate': .001,
#            'momentum': .9,
#            'weight_decay': 0.0005,
#        }
#    )
#    dataset = ingest_data.grab_mnist_category_dataset()
#    data_shape = dataset.data_shape
#    input_shape = (None, data_shape[2], data_shape[0], data_shape[1])

#    # Choose model
#    model = models.MNISTModel(
#        input_shape=input_shape, output_dims=dataset.output_dims,
#        training_dpath=dataset.training_dpath, **hyperparams)

#    # Initialize architecture
#    model.initialize_architecture()

#    # Load previously learned weights or initialize new weights
#    if model.has_saved_state():
#        model.load_model_state()
#    else:
#        model.reinit_weights()

#    config = dict(
#        learning_rate_schedule=15,
#        max_epochs=120,
#        show_confusion=False,
#        run_test=None,
#        show_features=False,
#        print_timing=False,
#    )

#    X_train, y_train = dataset.load_subset('train')
#    X_valid, y_valid = dataset.load_subset('valid')
#    #X_test, y_test = utils.load_from_fpath_dicts(data_fpath_dict, label_fpath_dict, 'test')
#    harness.train(model, X_train, y_train, X_valid, y_valid, dataset, config)
