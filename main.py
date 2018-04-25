#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a custom Estimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data
from capsule import capsule, margin_loss

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=30, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def my_model(features, labels, mode, params):
    input_dim = params['input_dim']
    input_atoms = params['input_atoms']
    output_dim = params['output_dim']
    output_atoms = params['output_atoms']
    iter_routing = params['iter_routing']
    print('input_dim:%d, output_dim:%d' %(input_dim, output_dim))

    X = tf.feature_column.input_layer(features, params['feature_columns'])
    print('X:', X)
    X = tf.reshape(X, [-1, input_dim, input_atoms])
    print('X:', X)

    v_J = capsule(X, output_dim, output_atoms, iter_routing) 
    epsilon = 0.000001
    v_length = tf.sqrt(tf.reduce_sum(tf.square(v_J), axis=2) + epsilon)

    # Compute predictions.
    softmax_v = tf.nn.softmax(v_length, dim=1)
    argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
    predicted_classes = argmax_idx
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': softmax_v,
            #'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    Y = tf.one_hot(labels, depth=3, axis=1, dtype=tf.float32)
    loss = margin_loss(v_length, Y)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key=key,
            hash_bucket_size=100, dtype=tf.int32),
            dimension=8))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': feature_columns,
            'input_dim': 4,
            'input_atoms':8,
            'output_dim': 3, # 3 classes
            'output_atoms':3,
            'iter_routing':2,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
            input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size), steps=1)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
