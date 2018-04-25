import tensorflow as tf
import numpy as np

epsilon = 0.000001

def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        [batch_size, dim, atoms]

    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def margin_loss(v_length, y, lambda_val=0.5, m_plus=0.9, m_minus=0.1):
    # 1. The margin loss

    # max_l = max(0, m_plus-||v_c||)^2
    max_l = tf.square(tf.maximum(0., m_plus - v_length))
    # max_r = max(0, ||v_c||-m_minus)^2
    max_r = tf.square(tf.maximum(0., v_length - m_minus))

    # calc T_c: [batch_size, 10]
    # T_c = Y, is my understanding correct? Try it.
    T_c = y
    # [batch_size, 10], element-wise multiply
    L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

    return margin_loss

def capsule(input, output_dim, output_atoms, iter_routing):
    '''
    Args:
        input: input Tensor, shape [batch_size, input_dim, input_atoms]
        input_dim:
    Returns:
        output tensor
    '''
    # dynamic batch_size
    batch_size = tf.shape(input)[0]
    input_dim = input.shape[1]
    input_atoms = input.shape[2]
    
    with tf.variable_scope('routing'):
        W = tf.get_variable('Weight', shape=(input_dim, output_dim, input_atoms, output_atoms), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias", shape=(output_dim, output_atoms))
        print('W:', W)
        print('biases:', biases)
        
        # about the reason of using 'batch_size', see issue #21
        # b_IJ: [batch_size, input_dim, ouput_dim],
        #b_IJ = tf.constant(np.zeros([batch_size, input_dim, output_dim], dtype=np.float32))
        b_IJ = tf.zeros([batch_size, input_dim, output_dim], dtype=np.float32)
        print('b_IJ:', b_IJ)
        assert b_IJ.get_shape().as_list() == [None, input_dim, output_dim] 
    
        # Eq.2, calc u_hat
        # do tiling for input and W before matmul
        # input: [batch_size, input_dim, output_dim, 1, input_atoms]
        #     W: [batch_size, input_dim, output_dim, input_atoms, output_atoms]
        #input = tf.expand_dims(tf.expand_dims(input, 2), 3)
        input = tf.reshape(input, [-1, input_dim, 1, 1, input_atoms])
        input = tf.tile(input, [1, 1, output_dim, 1, 1])
        W = tf.tile(tf.expand_dims(W, 0), [batch_size, 1, 1, 1, 1])
        print('input:', input)
        print('W:', W)
        assert input.get_shape().as_list() == [None, input_dim, output_dim, 1, input_atoms] 
        assert W.get_shape().as_list() == [None, input_dim, output_dim, input_atoms, output_atoms] 
        
        # u_hat: [batch_size, input_dim, output_dim, output_atoms]
        u_hat = tf.matmul(input, W)
        u_hat = tf.squeeze(u_hat, axis=-2)
        print('u_hat:', u_hat)
        assert u_hat.get_shape().as_list() == [None, input_dim, output_dim, output_atoms] 
        
        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

        # line 3,for r iterations do
        for r_iter in range(iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # c_IJ: [batch_size, input_dim, output_dim]
                c_IJ = tf.nn.softmax(b_IJ, dim=1)
                # c_IJ: [batch_size, input_dim, output_dim, 1]
                c_IJ = tf.expand_dims(c_IJ, axis=-1)
                print("c_IJ:", c_IJ)

                if r_iter < iter_routing - 1:
                    # Inner iterations, do not apply backpropagation
                    # s_J: [batch_size, input_dim, output_dim, output_atoms]
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    print("s_J:", s_J)
                    #s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) + biases
                    s_J = tf.reduce_sum(s_J, axis=1) + biases
                    # s_J [batch_size, output_dim, output_atoms]
                    v_J = squash(s_J)
                    print("v_J", v_J)

                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                    #v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                    #u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_b=True)
                    # u_produce_v [batch_size, input_dim, output_dim, output_dim]
                    #assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                    # v_J_tiled: [batch_size, input_dim, output_dim, output_atoms]
                    v_J_tiled = tf.tile(tf.expand_dims(v_J, 1), [1, input_dim, 1, 1])
                    u_produce_v = tf.multiply(u_hat_stopped, v_J_tiled)
                    print('u_produce_v', u_produce_v)
                    b_IJ += tf.reduce_sum(u_produce_v, axis=-1)

                elif r_iter == iter_routing - 1:
                    # At last iteration, use `u_hat` in order to receive gradients from the following graph
                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat)
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1) + biases
                    #assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                    # line 6:
                    # squash using Eq.1,
                    v_J = squash(s_J)
                    #assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

        return(v_J)
        # v_J [batch_size, output_dim, output_atoms]

if __name__ == '__main__':
    a = [1, 1, 2, 2, 3, 3, 4, 4]
    b1 = np.reshape(a, [-1, 2, 2])
    b2 = np.reshape(a[0:4], [-1, 2, 2])
    print("b1 shape", b1.shape, "b2 shape", b2.shape)

    i = tf.placeholder(tf.float32, [None, 2, 2])
    t = capsule(i, output_dim=2, output_atoms=3, iter_routing=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t1 = sess.run(t, feed_dict={i:b1})
        t2 = sess.run(t, feed_dict={i:b2})
        print("t1=", t1, "t2=", t2)
        print("t1 shape", t1.shape, "t2 shape", t2.shape)
