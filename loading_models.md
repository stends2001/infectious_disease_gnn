It seems that there are basically maximal threshold values, above whcih my temporal GCN cannot predict. That means that, for quite specific nodes really, the peaks are never really predicted right, but rather underpredicted!



n_epochs = 10
lr       = 0.0001

ml1_id = SpatialGCNModel(name = f'spatial_gcn-identity_graph', dataloader=epidata_nuts2_id)
ml1_id.set_model_hparams()
ml1_id.set_global_hparams(lr = lr, n_epochs=n_epochs, scheduler_kwargs={'step_size':20, 'gamma':0.95}, min_delta = 0.00001, loss = 'mse')
ml1_id.run_snapshot(debug=True)
ml1_id.train(verbose=2)
ml1_id.forecast()
ml1_id.show_forecasts(dataset='test', target_h = 0)


ml_trial = SpatialGCNModel(name = f'spatial_gcn-identity_graph_reloaded', dataloader=epidata_nuts2_id)
ml_trial.set_model_hparams()
ml_trial.set_global_hparams(lr = lr, n_epochs=n_epochs, scheduler_kwargs={'step_size':20, 'gamma':0.95}, min_delta = 0.00001, loss = 'mse')
ml_trial.load_weights(weights_path = 'config/checkpoints/0001.pt')
ml_trial.forecast()
ml_trial.show_forecasts(dataset='test', target_h = 0)