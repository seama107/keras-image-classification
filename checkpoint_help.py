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

def create_callbacks(model_dir, history=False, checkp=True, earlyStop=False):
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


# Config and Hyperparams

# Building a model
## Blah Blah Blah


# Now, the important part. Add the following:
# Replace <YOUR_MODEL>

model_dir = create_model_dir(<YOUR_MODEL>)
print("Saved model data to:", os.path.abspath(model_dir))
history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
callbacks = create_callbacks(model_dir)

#  Update the call to .fit() or .fit_generator(), and add the arguement
# 'callbacks=callbacks'
model.fit(X_train, y_train, callbacks=callbacks)
