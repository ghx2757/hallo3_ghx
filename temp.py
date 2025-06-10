
# ...existing code...
from transformers import HubertModel
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
class HubertCustomModel(HubertModel):
    """
    HubertCustomModel 是一个自定义模型类，继承自 transformers 库的 HubertModel。
    它继承了 HubertModel 的所有功能，并添加了特征提取和编码的方法。

    Attributes:
        base_model (HubertModel): 基础 HubertModel 对象。

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            前向传播，输入原始音频，返回模型输出。

        feature_extract(input_values, seq_len):
            使用基础模型提取特征。

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            对提取的特征进行编码，返回编码后的特征。
    """
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)
        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# ...existing code...

'''

好的，根据你的要求，我们将创建一个新的 HubertModel 类，它将继承自 transformers.HubertModel 并复用 Wav2VecModel 中的大部分逻辑。

Wav2Vec2Model 和 HubertModel 在 transformers 库中具有非常相似的内部结构和方法名称（例如 feature_extractor, feature_projection, encoder, _mask_hidden_states 等），这使得替换过程相对直接。

需要注意的是 linear_interpolation 函数，它在原始代码中是从 transformers.models.wav2vec2.modeling_wav2vec2 导入的。虽然 Hubert 模型也处理类似的数据，但这个特定的辅助函数是 Wav2Vec2 模块的一部分。为了保持与原始 Wav2VecModel 提供的逻辑完全一致，我们将保留对 linear_interpolation 的导入和使用。在实际应用中，你可能需要确认 HubertModel 的 forward 内部是否已经包含了类似的处理，或者这个 linear_interpolation 在 Hubert 上是否仍然适用。但根据你的要求，我们将直接复制其行为。

import torch
from transformers import HubertModel as BaseHubertModel # Rename to avoid conflict with our custom class name
from transformers.modeling_outputs import BaseModelOutput

# The original Wav2VecModel uses linear_interpolation which is from the Wav2Vec2 modeling file.
# To exactly replicate the original logic, we need to import it.
# In a real-world scenario, you might check if Hubert's internal processing handles this differently
# or if a more generic interpolation function is available/needed.
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices, # Not explicitly used in the provided methods, but often comes with others
    _sample_negative_indices, # Not explicitly used in the provided methods
    linear_interpolation
)

class HubertModel(BaseHubertModel):
    """
    HubertModel is a custom model class that extends the HubertModel class from the transformers library.
    It inherits all the functionality of the HubertModel and adds additional methods for feature extraction and encoding,
    mimicking the structure provided for Wav2VecModel.
    ...

    Attributes:
        # Since this class directly extends BaseHubertModel, 'self' itself is the base model.
        # The original 'base_model' attribute in Wav2VecModel's docstring was slightly misleading
        # given its inheritance structure, so we omit it here as 'self' serves that purpose.

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            Forward pass of the HubertModel.
            It takes input_values, seq_len, and other optional parameters as input and returns the output of the base model.

        feature_extract(input_values, seq_len):
            Extracts features from the input_values using the base model's feature extractor.

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            Encodes the extracted features using the base model's encoder and returns the encoded features.
    """
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the Hubert model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the Hubert model.
        """
        # The original Wav2VecModel explicitly set output_attentions to True.
        # We retain this behavior for direct translation.
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Feature extraction step, identical to the Wav2VecModel's approach
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            # This method is inherited from the base HubertModel
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # Feature projection and masking, identical to Wav2VecModel
        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # Encoder pass, identical to Wav2VecModel
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        # Optional adapter layer, identical to Wav2VecModel
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        """
        Extracts features from the input values and returns the extracted features.

        Parameters:
        input_values (torch.Tensor): The input values to be processed.
        seq_len (torch.Tensor): The sequence lengths of the input values.

        Returns:
        extracted_features (torch.Tensor): The extracted features from the input values.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Encodes the input features into the output space.

        Args:
            extract_features (torch.Tensor): The extracted features from the audio signal.
            attention_mask (torch.Tensor, optional): Attention mask to be used for padding.
            mask_time_indices (torch.Tensor, optional): Masked indices for the time dimension.
            output_attentions (bool, optional): If set to True, returns the attention weights.
            output_hidden_states (bool, optional): If set to True, returns all hidden states.
            return_dict (bool, optional): If set to True, returns a BaseModelOutput instead of the tuple.

        Returns:
            The encoded output features.
        """
        # Retaining the explicit setting of self.config.output_attentions = True
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
'''