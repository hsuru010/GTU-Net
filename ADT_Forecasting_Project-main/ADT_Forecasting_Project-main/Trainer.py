

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         TRAINER                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

if __name__ == '__main__':
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    import torch
    import numpy as np
    import pickle

    from config import dataloader_config, model_name, models_dict, model_params
    from LightningModule import ADT_Lightning_Module
    from Dataloaders import time_series_module


    torch.set_float32_matmul_precision('high')


    """Define EarlyStopping callback for training."""
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=8,
        verbose=True,
        mode='min'
    )

    """Define Lightning Trainer.
    """
    trainer = L.Trainer(
        max_epochs=200,
        callbacks=[early_stopping_callback],
        accelerator="auto",
        devices=[0],
        log_every_n_steps=10,
    )



    #---------------------------------------------#
    #                  TRAINING                   #
    #---------------------------------------------#

    # define model using LightningModule
    model = ADT_Lightning_Module(models_dict, model_name, **model_params)

    # load data
    data_module = time_series_module(dataloader_config)
    data_module.setup()

    # prepate datasets
    train_set, train_loader = data_module.train_dataloader()

    val_set, val_loader = data_module.val_dataloader()
    test_set, test_loader, test_mean, test_std = data_module.test_dataloader()

    # launch training
    model_training = trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training completed!")
    torch.save(model.state_dict(), "trained_model.pth")

    # validate training
    model_validation = trainer.validate(model, dataloaders=val_loader)

    # testing results
    model_testing = trainer.test(model, dataloaders=test_loader)

    # make predictions
    predictions = trainer.predict(model, dataloaders=test_loader)

    # 保存预测结果和相关变量
    with open('test_mean_std.pkl', 'wb') as f:
        pickle.dump({'test_mean': test_mean, 'test_std': test_std}, f)

    torch.save(test_set, 'test_set.pt')
    torch.save(predictions, 'predictions.pt')


    print("Predictions and related variables have been saved.")

