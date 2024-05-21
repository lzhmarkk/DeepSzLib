import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel


class LlamaEncoder(LlamaModel):
    def __init__(self, hidden_size, num_hidden_layers,
                 num_attention_heads, max_position_embeddings,
                 intermediate_size, hidden_act):
        config = LlamaConfig(hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_attention_heads=num_attention_heads,
                             max_position_embeddings=max_position_embeddings,
                             intermediate_size=intermediate_size,
                             hidden_act=hidden_act,
                             use_cache=False
                             )

        super().__init__(config)

        self.embed_tokens = None

    def forward(self, inputs_embeds):
        # (B, T, D)
        batch_size, seq_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # embed positions
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, 0
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = ()
        for decoder_layer in self.layers:
            all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        all_hidden_states += (hidden_states,)
        return all_hidden_states
