history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch)