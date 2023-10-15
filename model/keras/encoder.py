import tensorflow as tf
from .positional_encoding import PositionalEncoding
from .multi_head_attention import MultiHeadAttention

def encoder_layer(units, d_model, num_heads, dropout,name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None,d_model ), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention1 = MultiHeadAttention(
    d_model, num_heads, name="attention1")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })

  attention1 = tf.keras.layers.Dropout(rate=dropout)(attention1)
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention1)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention2")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })

  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
    epsilon=1e-6)(inputs + attention2)

  attention3 = MultiHeadAttention(
    d_model, num_heads, name="attention3")({
        'query': attention1,
        'key': attention2,
        'value': attention1,
        'mask': padding_mask
    })

  attention3 = tf.keras.layers.Dropout(rate=dropout)(attention3)
  attention3 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention3)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention3)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention3)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(
  time_steps,
  num_layers,
  units,
  d_model,
  num_heads,
  dropout,
  projection,
  name="encoder"
):

  inputs = tf.keras.Input(shape=(None,d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  projection=tf.keras.layers.Dense( d_model,use_bias=True, activation='linear')(inputs)

  projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  projection = PositionalEncoding(time_steps, d_model)(projection)

  outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

  for i in range(num_layers):
    outputs = encoder_layer(
        units = units,
        d_model = d_model,
        num_heads = num_heads,
        dropout = dropout,
        name = "encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
    inputs = [inputs, padding_mask], outputs=outputs, name=name)
