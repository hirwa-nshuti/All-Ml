This notebook contains the classification of MNIST fashion dataset.

During the Neural network training it is hard to precise the number of
epochs to use and sometimes this can lead us to overfitting. Due to that
case the early stopping of the neural network can be the best solution as
the model has to stop when the needed performance is reached.

Here is the implementation of callback class:
```
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs={}):
        if(logs.get('accuracy') > 0.85):
            print("\nAccuracy is high now cancelling training!")
            self.model.stop_training = True
```


When fitting the model you need to instantiate callback 
object and add a callback parameter to model fit function: </br>
```callbacks = myCallback()
model.fit(train_data, train_labels, epochs=5, callbacks=[callbacks])
```


For more information on Callbacks you can visit the [Keras Callbacks API](https://keras.io/api/callbacks/).
To know other available callbacks APIs
