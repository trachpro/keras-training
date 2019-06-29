import tensorflow as tf

def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    list_op = graph.get_operations()
    for op in graph.get_operations():
        print(op.name)


printTensors("/media/trongpq/HDD/ai_data/driver-ocr/checkpoints/digit/201904160859_lenet5_12/weights-181-0.99.pb")