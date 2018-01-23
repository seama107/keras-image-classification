import time
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import History

checkpoint_dir = "models"

def create_model_dir(model):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    timestr = time.strftime("%m%d-%H%M")
    model_name = "CNN_" + timestr
    model_dir = os.path.join(checkpoint_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, "model.h5"))
    return model_dir

def create_callbacks(model_dir, history=True, checkp=True, earlyStop=False):
    callbacks = []
    weights_filepath = os.path.join(model_dir,
        "weights-improvement-{epoch:03d}-{loss:.4f}.hdf5")

    checkpoint = ModelCheckpoint(weights_filepath, save_best_only=True,
        verbose=1)
    esCallback = EarlyStopping(min_delta=0, patience=10, verbose=1)
    hisCallback = History()

    if history:
        callbacks.append(hisCallback)
    if checkp:
        callbacks.append(checkpoint)
    if earlyStop:
        callbacks.append(esCallback)
    return callbacks

def train_model(model, X_train, y_train, X_test, y_test, batch_size, callbacks_list,
 num_epochs, history, initial_epoch=0):
    for e in range(num_epochs):
        epochs = e + initial_epoch
        try:
            print("\nEPOCH {}\n".format(epochs))
            hist = model.fit(X_train, y_train, validation_data=(X_test,y_test),
                batch_size=batch_size, epochs=epochs+1, callbacks=callbacks_list,
                initial_epoch=epochs)
            for k, v in hist.history.items():
                history[k] = history[k] + v
        except KeyboardInterrupt:
            print("Exiting training loop")
            break
    return history
