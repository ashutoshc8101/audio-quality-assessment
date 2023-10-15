import tensorflow as tf
from .multi_head_attention import create_padding_mask
from .encoder import encoder

def transformer(
  time_steps,
  num_layers,
  units,
  d_model,
  num_heads,
  dropout,
  output_size,
  projection,
  name="transformer"
):

  inputs = tf.keras.Input(shape=(None,d_model), name="inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(tf.dtypes.cast(

      #Like our input has a dimension of length X d_model but the masking is applied to a vector
      # We get the sum for each row and result is a vector. So, if result is 0 it is because in that position was masked
      tf.math.reduce_sum(
      inputs,
      axis=2,
      keepdims=False,
      name=None
  ), tf.int32))


  enc_outputs = encoder(
      time_steps=time_steps,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
      projection=projection,
      name='encoder'
  )(inputs=[inputs, enc_padding_mask])

  #We reshape for feeding our FC in the next step
  outputs=tf.reshape(enc_outputs,(-1,time_steps*d_model))

  #We predict our class
  outputs = tf.keras.layers.Dense(units=output_size,use_bias=True, name="outputs")(outputs)

  return tf.keras.Model(inputs=[inputs], outputs=outputs, name='audio_class')

