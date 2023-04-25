from torch import nn
from transformers import BartForConditionalGeneration, AutoConfig, BartConfig


# /home/mixiangw/anaconda3/envs/pytorch/lib/python3.7/site-packages/transformers/models/bart
class Sep2SepModel(nn.Module):
    def __init__(self, model_name, vocab_size, label_smoothing=0.01):
        super(Sep2SepModel, self).__init__()
        self.new_config = BartConfig(
            vocab_size=vocab_size,
            max_position_embeddings=1024,
            encoder_layers=6,
            encoder_ffn_dim=1024,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_ffn_dim=1024,
            decoder_attention_heads=8,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            activation_function="gelu",
            d_model=256,
            dropout=0.1,
            attention_dropout=0.0,
            scale_embedding=False,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=4,
            decoder_start_token_id=2,
            forced_eos_token_id=2,
        )
        print(self.new_config)
        # 从配置项中拉取预训练的 模型
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name, config=self.new_config,
                                                                       ignore_mismatched_sizes=True)
        print('hidden_size:', self.bart_model.config.hidden_size)  # 输出默认的隐藏维度大小，为768
        print('d_model', self.bart_model.config.d_model)  # 输出默认的维度大小，为768

        self.label_smoothing = label_smoothing
        print(self.new_config.hidden_size, self.new_config.vocab_size)
        # 掩蔽 tokens 分类
        self.fc = nn.Linear(self.new_config.hidden_size, self.new_config.vocab_size)

    def forward(self, input_ids, mlm_labels=None, masks=None, decoded_inputs=None, lm_labels=None, dec_masks=None):
        outputs = self.bart_model(input_ids, attention_mask=masks.float(), decoder_input_ids=decoded_inputs,
                                  labels=lm_labels, decoder_attention_mask=dec_masks)
        lm_loss = outputs.loss
        loss = lm_loss
        mlm_loss = 0.
        if mlm_labels is not None:
            prediction_scores = self.fc(outputs.encoder_last_hidden_state)
            # 忽略值为-100的损失计算
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(prediction_scores.view(-1, self.new_config.vocab_size), mlm_labels.view(-1))
            loss = loss + 0.1 * mlm_loss
        return loss, lm_loss, mlm_loss


if __name__ == '__main__':
    model = Sep2SepModel("facebook/bart-base", vocab_size=1400)

    # for name, param in model.named_parameters():
    #     print(name)
