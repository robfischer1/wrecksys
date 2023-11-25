import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs

from wrecksys_ai.config import ConfigFile
from wrecksys_ai.io import load_datasets
from wrecksys_ai.model.assets import layers, losses, callback_list, metrics_list

model_config = ConfigFile().data.model
books = tf.range(1, model_config.vocabulary_size + 1)

train, test = load_datasets()


class Rex(tfrs.Model):
    def __init__(self):
        super().__init__()
        self._query_model = layers.ContextEncoder(
            vocab_size=model_config.vocabulary_size,
            embedding_dim=model_config.embedding_dimensions,
            rnn_dim=model_config.rnn_dimensions
        )

        self._candidate_model = layers.BookEncoder(
            vocab_size=model_config.vocabulary_size,
            embedding_dim=model_config.embedding_dimensions
        )

        self._task = tfrs.tasks.Retrieval(
            loss=losses.GlobalSoftmax(),
            metrics=metrics_list([1, 5, 10])
        )

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        label_id = inputs.pop('label_id')
        label_embedding = self._candidate_model(label_id)
        context_embedding = self._query_model(inputs)
        return self._task(context_embedding, label_embedding, compute_metrics=not training)


"""model = Rex()
steps = model_config.train_size // model_config.batch_size
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train, epochs=1, steps_per_epoch=steps)"""

test_config = dict(model_config)
print(test_config)