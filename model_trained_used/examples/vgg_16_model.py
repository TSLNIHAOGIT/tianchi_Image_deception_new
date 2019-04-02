import tensorflow as tf

saver = tf.train.import_meta_graph('vgg_16/vgg_16.ckpt.meta')
with tf.Session() as sess:
    all_var=tf.all_variables()
    for each in all_var:
        print(each)
    # model_ckpt=tf.train.latest_checkpoint('vgg_16')
    # print('model_ckpt',  model_ckpt)
    # saver.restore(sess,'vgg_16/vgg_16.ckpt' )


    # Access the graph
    # graph = tf.get_default_graph()
    ## Prepare the feed_dict for feeding data for fine-tuning

    # Access the appropriate output for fine-tuning
    # fc7 = graph.get_tensor_by_name('fc7:0')

    # # use this if you only want to change gradients of the last layer
    # fc7 = tf.stop_gradient(fc7)  # It's an identity function
    # fc7_shape = fc7.get_shape().as_list()
    #
    # new_outputs = 2
    # weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
    # biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    # output = tf.matmul(fc7, weights) + biases
    # pred = tf.nn.softmax(output)
    #
    # # Now, you run this with fine-tuning data in sess.run()
