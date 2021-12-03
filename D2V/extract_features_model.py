import tensorflow as tf
from D2V.modules import FunctionF, FunctionH, FunctionG, PoolF, PoolG

def Dataset2VecModel(configuration):
    nonlinearity_d2v = configuration['nonlinearity_d2v']
    # Function F
    units_f = configuration['units_f']
    nhidden_f = configuration['nhidden_f']
    architecture_f = configuration['architecture_f']
    resblocks_f = configuration['resblocks_f']

    # Function G
    units_g = configuration['units_g']
    nhidden_g = configuration['nhidden_g']
    architecture_g = configuration['architecture_g']

    # Function H
    units_h = configuration['units_h']
    nhidden_h = configuration['nhidden_h']
    architecture_h = configuration['architecture_h']
    resblocks_h = configuration['resblocks_h']
    #
    batch_size = configuration["batch_size"]
    trainable = False
    # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
    x = tf.keras.Input(shape=(2), dtype=tf.float32)
    # Number of sampled classes from triplets
    nclasses = tf.keras.Input(shape=(batch_size), dtype=tf.int32, batch_size=1)
    # Number of sampled features from triplets
    nfeature = tf.keras.Input(shape=(batch_size), dtype=tf.int32, batch_size=1)
    # Number of sampled instances from triplets
    ninstanc = tf.keras.Input(shape=(batch_size), dtype=tf.int32, batch_size=1)
    # Encode the predictor target relationship across all instances
    layer = FunctionF(units=units_f, nhidden=nhidden_f, nonlinearity=nonlinearity_d2v, architecture=architecture_f,
                      resblocks=resblocks_f, trainable=trainable)(x)
    # Average over instances
    layer = PoolF(units=units_f)(layer, nclasses[0], nfeature[0], ninstanc[0])
    # Encode the interaction between features and classes across the latent space
    layer = FunctionG(units=units_g, nhidden=nhidden_g, nonlinearity=nonlinearity_d2v, architecture=architecture_g,
                      trainable=trainable)(layer)
    # Average across all instances
    layer = PoolG(units=units_g)(layer, nclasses[0], nfeature[0])
    # Extract the metafeatures
    metafeatures = FunctionH(units=units_h, nhidden=nhidden_h, nonlinearity=nonlinearity_d2v,
                             architecture=architecture_h, trainable=trainable, resblocks=resblocks_h)(layer)
    # define hierarchical dataset representation model
    dataset2vec = tf.keras.Model(inputs=[x, nclasses, nfeature, ninstanc], outputs=metafeatures)
    return dataset2vec