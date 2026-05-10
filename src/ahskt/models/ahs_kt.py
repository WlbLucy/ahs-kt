import tensorflow as tf

from .encoders import BehaviorEncoder, DifficultyEncoder


class AHSKTModel(tf.keras.Model):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.fusion_mode = str(getattr(model_config, "fusion_mode", "early"))
        self.aux_residual_scale = float(getattr(model_config, "aux_residual_scale", 0.25))
        self.difficulty_mode = str(getattr(model_config, "difficulty_mode", "embedding"))
        self.difficulty_bias_scale = float(getattr(model_config, "difficulty_bias_scale", 0.1))
        self.difficulty_feature_source = str(getattr(model_config, "difficulty_feature_source", "question_concept"))
        self.question_global_easiness = float(getattr(model_config, "question_global_easiness", 0.5))
        self.concept_global_easiness = float(getattr(model_config, "concept_global_easiness", 0.5))
        self.question_embedding = tf.keras.layers.Embedding(
            input_dim=model_config.num_questions + 1,
            output_dim=model_config.embedding_dim,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.concept_embedding = tf.keras.layers.Embedding(
            input_dim=model_config.num_concepts + 1,
            output_dim=model_config.embedding_dim,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.response_embedding = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=model_config.embedding_dim,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.difficulty_encoder = DifficultyEncoder(model_config)
        self.behavior_encoder = BehaviorEncoder(model_config)
        self.input_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="relu")
        self.dropout = tf.keras.layers.Dropout(model_config.dropout)
        self.sequence_encoder = tf.keras.layers.GRU(
            model_config.hidden_dim,
            return_sequences=True,
            dropout=model_config.dropout,
        )
        self.context_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.target_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.output_projection = tf.keras.layers.Dense(1)
        self.base_output_projection = tf.keras.layers.Dense(1)
        self.difficulty_context_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.difficulty_target_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.difficulty_gate_projection = tf.keras.layers.Dense(1, activation="sigmoid")
        self.difficulty_output_projection = tf.keras.layers.Dense(1)
        self.difficulty_bias_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.difficulty_bias_gate_projection = tf.keras.layers.Dense(1, activation="sigmoid")
        self.difficulty_bias_output_projection = tf.keras.layers.Dense(1)
        self.smoothed_difficulty_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.smoothed_difficulty_gate_projection = tf.keras.layers.Dense(1, activation="sigmoid")
        self.smoothed_difficulty_output_projection = tf.keras.layers.Dense(1)
        self.behavior_context_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.behavior_target_projection = tf.keras.layers.Dense(model_config.hidden_dim, activation="tanh")
        self.behavior_gate_projection = tf.keras.layers.Dense(1, activation="sigmoid")
        self.behavior_output_projection = tf.keras.layers.Dense(1)

    @staticmethod
    def _shift_to_next_step(tensor):
        zero_step = tf.zeros_like(tensor[:, :1, ...])
        return tf.concat([tensor[:, 1:, ...], zero_step], axis=1)

    @staticmethod
    def _zero_state(reference_tensor, hidden_dim):
        batch_size = tf.shape(reference_tensor)[0]
        sequence_length = tf.shape(reference_tensor)[1]
        return tf.zeros((batch_size, sequence_length, hidden_dim), dtype=reference_tensor.dtype)

    def _normalized_next_difficulty_features(self, batch):
        next_question_difficulty = tf.cast(self._shift_to_next_step(batch["question_difficulty"]), tf.float32)
        next_concept_difficulty = tf.cast(self._shift_to_next_step(batch["concept_difficulty"]), tf.float32)
        question_denominator = float(max(1, int(self.model_config.num_question_difficulty) - 1))
        concept_denominator = float(max(1, int(self.model_config.num_concept_difficulty) - 1))
        question_easiness = (next_question_difficulty - 1.0) / question_denominator
        concept_easiness = (next_concept_difficulty - 1.0) / concept_denominator
        if self.difficulty_feature_source == "question_only":
            return tf.stack(
                [
                    question_easiness,
                    1.0 - question_easiness,
                    tf.square(question_easiness),
                    tf.sqrt(tf.maximum(question_easiness, 1e-6)),
                ],
                axis=-1,
            )
        if self.difficulty_feature_source == "concept_only":
            return tf.stack(
                [
                    concept_easiness,
                    1.0 - concept_easiness,
                    tf.square(concept_easiness),
                    tf.sqrt(tf.maximum(concept_easiness, 1e-6)),
                ],
                axis=-1,
            )
        mean_easiness = 0.5 * (question_easiness + concept_easiness)
        disagreement = tf.abs(question_easiness - concept_easiness)
        interaction = question_easiness * concept_easiness
        return tf.stack(
            [question_easiness, concept_easiness, mean_easiness, disagreement, interaction],
            axis=-1,
        )

    def _smoothed_target_difficulty_features(self, batch):
        next_question_easiness = tf.cast(self._shift_to_next_step(batch["question_easiness"]), tf.float32)
        next_concept_easiness = tf.cast(self._shift_to_next_step(batch["concept_easiness"]), tf.float32)
        next_question_confidence = tf.cast(self._shift_to_next_step(batch["question_confidence"]), tf.float32)
        next_concept_confidence = tf.cast(self._shift_to_next_step(batch["concept_confidence"]), tf.float32)

        if self.difficulty_feature_source == "question_only":
            question_delta = next_question_easiness - self.question_global_easiness
            return tf.stack(
                [
                    next_question_easiness,
                    next_question_confidence,
                    question_delta,
                    next_question_confidence * question_delta,
                    (1.0 - next_question_confidence) * question_delta,
                ],
                axis=-1,
            )

        if self.difficulty_feature_source == "concept_only":
            concept_delta = next_concept_easiness - self.concept_global_easiness
            return tf.stack(
                [
                    next_concept_easiness,
                    next_concept_confidence,
                    concept_delta,
                    next_concept_confidence * concept_delta,
                    (1.0 - next_concept_confidence) * concept_delta,
                ],
                axis=-1,
            )

        mean_easiness = 0.5 * (next_question_easiness + next_concept_easiness)
        mean_confidence = 0.5 * (next_question_confidence + next_concept_confidence)
        mean_delta = mean_easiness - (0.5 * (self.question_global_easiness + self.concept_global_easiness))
        disagreement = tf.abs(next_question_easiness - next_concept_easiness)
        return tf.stack(
            [
                mean_easiness,
                mean_confidence,
                mean_delta,
                mean_confidence * mean_delta,
                disagreement,
            ],
            axis=-1,
        )

    def _compute_legacy_logits(
        self,
        sequence_state,
        question_emb,
        concept_emb,
        difficulty_state,
        behavior_state,
        training=False,
    ):
        if not bool(self.model_config.use_target_interaction):
            logits = self.output_projection(
                tf.concat([sequence_state, difficulty_state, behavior_state], axis=-1),
                training=training,
            )
            return tf.squeeze(logits, axis=-1)

        next_question_emb = self._shift_to_next_step(question_emb)
        next_concept_emb = self._shift_to_next_step(concept_emb)
        next_difficulty_state = self._shift_to_next_step(difficulty_state)

        context_state = self.context_projection(
            tf.concat([sequence_state, difficulty_state, behavior_state], axis=-1),
            training=training,
        )
        target_state = self.target_projection(
            tf.concat([next_question_emb, next_concept_emb, next_difficulty_state], axis=-1),
            training=training,
        )

        interaction_logit = tf.reduce_sum(context_state * target_state, axis=-1, keepdims=True)
        calibration_logit = self.output_projection(
            tf.concat([context_state, target_state, behavior_state, difficulty_state], axis=-1),
            training=training,
        )
        logits = interaction_logit + calibration_logit
        return tf.squeeze(logits, axis=-1)

    def _compute_residual_logits(
        self,
        batch,
        sequence_state,
        question_emb,
        concept_emb,
        difficulty_state,
        behavior_state,
        training=False,
    ):
        if not bool(self.model_config.use_target_interaction):
            logits = self.output_projection(
                tf.concat([sequence_state, difficulty_state, behavior_state], axis=-1),
                training=training,
            )
            return tf.squeeze(logits, axis=-1)

        next_question_emb = self._shift_to_next_step(question_emb)
        next_concept_emb = self._shift_to_next_step(concept_emb)
        next_difficulty_state = self._shift_to_next_step(difficulty_state)

        base_context = self.context_projection(sequence_state, training=training)
        base_target = self.target_projection(
            tf.concat([next_question_emb, next_concept_emb], axis=-1),
            training=training,
        )
        base_interaction_logit = tf.reduce_sum(base_context * base_target, axis=-1, keepdims=True)
        base_calibration_logit = self.base_output_projection(
            tf.concat([base_context, base_target], axis=-1),
            training=training,
        )
        logits = base_interaction_logit + base_calibration_logit

        if bool(self.model_config.use_difficulty_features):
            if self.difficulty_mode == "scalar_bias":
                difficulty_features = self._normalized_next_difficulty_features(batch)
                difficulty_bias_context = self.difficulty_bias_projection(
                    tf.concat([base_context, base_target, behavior_state, difficulty_features], axis=-1),
                    training=training,
                )
                difficulty_bias_gate = self.difficulty_bias_gate_projection(
                    tf.concat([difficulty_bias_context, difficulty_features], axis=-1),
                    training=training,
                )
                difficulty_bias_logit = self.difficulty_bias_output_projection(
                    tf.concat([difficulty_bias_context, difficulty_features], axis=-1),
                    training=training,
                )
                logits += self.difficulty_bias_scale * difficulty_bias_gate * difficulty_bias_logit
            elif self.difficulty_mode == "smoothed_target_calibration":
                difficulty_features = self._smoothed_target_difficulty_features(batch)
                difficulty_hidden = self.smoothed_difficulty_projection(
                    tf.concat([base_target, difficulty_features], axis=-1),
                    training=training,
                )
                difficulty_gate = self.smoothed_difficulty_gate_projection(
                    tf.concat([base_context, behavior_state, difficulty_features], axis=-1),
                    training=training,
                )
                difficulty_logit = self.smoothed_difficulty_output_projection(
                    tf.concat([base_context, base_target, behavior_state, difficulty_hidden, difficulty_features], axis=-1),
                    training=training,
                )
                logits += self.difficulty_bias_scale * difficulty_gate * difficulty_logit
            else:
                difficulty_context = self.difficulty_context_projection(
                    tf.concat([sequence_state, difficulty_state], axis=-1),
                    training=training,
                )
                difficulty_target = self.difficulty_target_projection(
                    tf.concat([next_question_emb, next_concept_emb, next_difficulty_state], axis=-1),
                    training=training,
                )
                difficulty_gate = self.difficulty_gate_projection(
                    tf.concat([base_context, difficulty_context, next_difficulty_state], axis=-1),
                    training=training,
                )
                difficulty_interaction = tf.reduce_sum(difficulty_context * difficulty_target, axis=-1, keepdims=True)
                difficulty_calibration = self.difficulty_output_projection(
                    tf.concat([difficulty_context, difficulty_target, base_target], axis=-1),
                    training=training,
                )
                logits += self.aux_residual_scale * difficulty_gate * (difficulty_interaction + difficulty_calibration)

        if bool(self.model_config.use_behavior_features):
            behavior_context = self.behavior_context_projection(
                tf.concat([sequence_state, behavior_state], axis=-1),
                training=training,
            )
            behavior_target = self.behavior_target_projection(
                tf.concat([next_question_emb, next_concept_emb, next_difficulty_state], axis=-1),
                training=training,
            )
            behavior_gate = self.behavior_gate_projection(
                tf.concat([base_context, behavior_state, difficulty_state], axis=-1),
                training=training,
            )
            behavior_interaction = tf.reduce_sum(behavior_context * behavior_target, axis=-1, keepdims=True)
            behavior_calibration = self.behavior_output_projection(
                tf.concat([behavior_context, behavior_target, behavior_state], axis=-1),
                training=training,
            )
            logits += self.aux_residual_scale * behavior_gate * (behavior_interaction + behavior_calibration)

        return tf.squeeze(logits, axis=-1)

    def call(self, batch, training=False):
        question_emb = self.question_embedding(batch["question_ids"], training=training)
        concept_emb = self.concept_embedding(batch["concept_ids"], training=training)
        response_emb = self.response_embedding(batch["responses"], training=training)

        if bool(self.model_config.use_difficulty_features):
            if self.difficulty_mode == "embedding":
                difficulty_state = self.difficulty_encoder(
                    batch["question_difficulty"],
                    batch["concept_difficulty"],
                    training=training,
                )
            else:
                difficulty_state = self._zero_state(question_emb, self.model_config.difficulty_dim)
        else:
            difficulty_state = self._zero_state(question_emb, self.model_config.difficulty_dim)

        if bool(self.model_config.use_behavior_features):
            behavior_state = self.behavior_encoder(
                batch["attempts"],
                batch["hints"],
                batch["speed"],
                batch["behavior_cluster"],
                difficulty_state,
                speed_relative_student=batch["speed_relative_student"],
                speed_relative_question=batch["speed_relative_question"],
                behavior_soft_membership=batch["behavior_soft_membership"],
                training=training,
            )
        else:
            behavior_state = self._zero_state(question_emb, self.model_config.behavior_dim)

        fused_inputs = tf.concat([question_emb, concept_emb, response_emb], axis=-1)
        if self.fusion_mode == "early":
            fused_inputs = tf.concat([fused_inputs, difficulty_state, behavior_state], axis=-1)
        hidden_inputs = self.input_projection(fused_inputs, training=training)
        hidden_inputs = self.dropout(hidden_inputs, training=training)
        mask = tf.cast(batch["mask"], tf.bool)
        sequence_state = self.sequence_encoder(hidden_inputs, mask=mask, training=training)
        if self.fusion_mode == "late_residual":
            return self._compute_residual_logits(
                batch=batch,
                sequence_state=sequence_state,
                question_emb=question_emb,
                concept_emb=concept_emb,
                difficulty_state=difficulty_state,
                behavior_state=behavior_state,
                training=training,
            )
        return self._compute_legacy_logits(
            sequence_state=sequence_state,
            question_emb=question_emb,
            concept_emb=concept_emb,
            difficulty_state=difficulty_state,
            behavior_state=behavior_state,
            training=training,
        )
