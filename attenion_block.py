import math
import tensorflow as tf


def split_heads(x, hidden_size, num_heads):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
      hidden_size: Int,The unit size of hidden layer
      num_heads: Int,The number of heads to split

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        # Calculate depth of last dimension after it has been split.
        depth = (hidden_size // num_heads)

        # Split the last dimension
        x = tf.reshape(x, [batch_size, length, num_heads, depth])

        # Transpose the result
        return tf.transpose(x, [0, 2, 1, 3])


def combine_heads(x, hidden_size):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
      hidden_size: Int,The unit size of hidden layer
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, [batch_size, length, hidden_size])


def multi_head_attention(x, y, hidden_size, num_heads, attention_dropout=0.3, scope='multi_head_attention',
                         training=False):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      hidden_size:Int,the size of units of layer
      num_heads: Int,the number of heads
      attention_dropout:Float,The probability to set the units to zeros when training
      scope: (optional),the scope's name the this block
      training: bool,true for train,false for inference
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    if hidden_size % num_heads != 0:
        raise ValueError("Hidden size must be evenly divisible by the number of "
                         "heads.")
    with tf.name_scope(name='multi_head_attention'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linearly project the query (q), key (k) and value (v) using different
            # learned projections. This is in preparation of splitting them into
            # multiple heads. Multi-head attention uses multiple queries, keys, and
            # values rather than regular attention (which uses a single q, k, v).
            q = tf.layers.dense(x, hidden_size, use_bias=False, name="q")
            k = tf.layers.dense(y, hidden_size, use_bias=False, name="k")
            v = tf.layers.dense(y, hidden_size, use_bias=False, name="v")

            # Split q, k, v into heads.
            q = split_heads(q, hidden_size, num_heads)
            k = split_heads(k, hidden_size, num_heads)
            v = split_heads(v, hidden_size, num_heads)

            # Scale q to prevent the dot product between q and k from growing too large.
            depth = (hidden_size // num_heads)
            q *= depth ** -0.5

            # Calculate dot product attention
            logits = tf.matmul(q, k, transpose_b=True)
            weights = tf.nn.softmax(logits, name="attention_weights")
            if training:
                weights = tf.nn.dropout(weights, 1.0 - attention_dropout)
            attention_output = tf.matmul(weights, v)

            # Recombine heads --> [batch_size, length, hidden_size]
            attention_output = combine_heads(attention_output, hidden_size)

            # Run the combined outputs through another linear projection layer.
            attention_output = tf.layers.dense(attention_output, hidden_size, use_bias=False, name="output_transform")
            return attention_output


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically unstable
    # in float16.
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal
