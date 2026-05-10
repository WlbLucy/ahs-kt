import tensorflow as tf


class DifficultyEncoder(tf.keras.layers.Layer):
    def __init__(self, model_config):
        super().__init__()
        self.question_difficulty_embedding = tf.keras.layers.Embedding(
            input_dim=model_config.num_question_difficulty + 1,
            output_dim=model_config.difficulty_dim,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.concept_difficulty_embedding = tf.keras.layers.Embedding(
            input_dim=model_config.num_concept_difficulty + 1,
            output_dim=model_config.difficulty_dim,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.projection = tf.keras.layers.Dense(model_config.difficulty_dim, activation="relu")
        self.normalization = tf.keras.layers.LayerNormalization()

    def call(self, question_difficulty, concept_difficulty, training=False):
        question_emb = self.question_difficulty_embedding(question_difficulty, training=training)
        concept_emb = self.concept_difficulty_embedding(concept_difficulty, training=training)
        difficulty_state = tf.concat([question_emb, concept_emb], axis=-1)
        difficulty_state = self.projection(difficulty_state, training=training)
        return self.normalization(difficulty_state, training=training)


class BehaviorEncoder(tf.keras.layers.Layer):
    def __init__(self, model_config):
        super().__init__()
        self.use_behavior_cluster = bool(model_config.use_behavior_cluster)
        self.behavior_condition_on_difficulty = bool(model_config.behavior_condition_on_difficulty)
        self.use_relative_speed = bool(getattr(model_config, "use_relative_speed", True))
        self.use_soft_behavior_prototypes = bool(getattr(model_config, "use_soft_behavior_prototypes", True))
        self.num_behavior_clusters = int(model_config.num_behavior_clusters)
        self.cluster_embedding = None
        if self.use_behavior_cluster:
            self.cluster_embedding = tf.keras.layers.Embedding(
                input_dim=model_config.num_behavior_clusters,
                output_dim=model_config.behavior_dim,
                embeddings_initializer=tf.keras.initializers.GlorotNormal(),
            )
        self.numeric_projection = tf.keras.layers.Dense(model_config.behavior_dim, activation="relu")
        self.behavior_projection = tf.keras.layers.Dense(model_config.behavior_dim, activation="relu")
        self.gate_projection = tf.keras.layers.Dense(model_config.behavior_dim, activation="sigmoid")
        self.normalization = tf.keras.layers.LayerNormalization()

    def _cluster_state(self, behavior_cluster, behavior_soft_membership, training=False):
        if not self.use_behavior_cluster or self.cluster_embedding is None:
            return None

        hard_cluster_state = self.cluster_embedding(behavior_cluster, training=training)
        if not self.use_soft_behavior_prototypes or behavior_soft_membership is None:
            return hard_cluster_state

        prototype_indices = tf.range(self.num_behavior_clusters, dtype=tf.int32)
        prototype_embeddings = self.cluster_embedding(prototype_indices, training=training)
        soft_cluster_state = tf.linalg.matmul(behavior_soft_membership, prototype_embeddings)
        soft_available = tf.cast(
            tf.reduce_sum(tf.abs(behavior_soft_membership), axis=-1, keepdims=True) > 0.0,
            soft_cluster_state.dtype,
        )
        return soft_available * soft_cluster_state + (1.0 - soft_available) * hard_cluster_state

    def call(
        self,
        attempts,
        hints,
        speed,
        behavior_cluster,
        difficulty_state,
        speed_relative_student=None,
        speed_relative_question=None,
        behavior_soft_membership=None,
        training=False,
    ):
        numeric_parts = [attempts, hints, speed]
        if self.use_relative_speed:
            if speed_relative_student is None:
                speed_relative_student = tf.zeros_like(speed)
            if speed_relative_question is None:
                speed_relative_question = tf.zeros_like(speed)
            numeric_parts.extend([speed_relative_student, speed_relative_question])
        numeric_inputs = tf.stack(numeric_parts, axis=-1)
        numeric_state = self.numeric_projection(numeric_inputs, training=training)
        fusion_parts = [numeric_state]
        cluster_state = self._cluster_state(behavior_cluster, behavior_soft_membership, training=training)
        if cluster_state is not None:
            fusion_parts.append(cluster_state)
        behavior_state = self.behavior_projection(tf.concat(fusion_parts, axis=-1), training=training)
        behavior_state = self.normalization(behavior_state, training=training)
        if not self.behavior_condition_on_difficulty:
            return behavior_state
        gate = self.gate_projection(tf.concat([behavior_state, difficulty_state], axis=-1), training=training)
        return gate * behavior_state + (1.0 - gate) * difficulty_state
