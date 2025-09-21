import torch
from safetensors.torch import load_file
from collections import OrderedDict
from codec_encoder import CodecEncoder_Transformer
from extra_layers import SemanticEncoder
from vocos import CodecDecoderVocos
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

class ModelCheckpointLoader:
    """
    A class to load a model checkpoint from a .safetensors file and extract
    specific state dictionaries for different parts of the model.

    Args:
        model_path (str): The path to the model.safetensors file.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.state_dict = self._load_model()
        self.filtered_state_dict_codec = self._filter_state_dict('CodecEnc.')
        self.filtered_state_dict_gen = self._filter_state_dict('generator.')
        self.filtered_state_dict_fc_post_a = self._filter_state_dict('fc_post_a.')
        self.filtered_state_dict_semantic_encoder = self._filter_state_dict('SemanticEncoder_module.')
        self.filtered_state_dict_fc_prior = self._filter_state_dict('fc_prior.')

    def _load_model(self):
        """
        Loads the model checkpoint from the specified path.
        """
        try:
            ckpt = load_file(f"{self.model_path}")
            return ckpt
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            return None
    def _filter_state_dict(self, prefix: str):
        """
        Filters the full state dictionary to extract a specific part based on a prefix.

        Args:
            prefix (str): The prefix to filter by (e.g., 'CodecEnc.').

        Returns:
            OrderedDict: The filtered state dictionary.
        """
        if self.state_dict is None:
            return OrderedDict()

        filtered_dict = OrderedDict()
        for key, value in self.state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                filtered_dict[new_key] = value
        return filtered_dict
    def _get_model(self, device: str, dtype: torch.dtype = None):
        """
        Initializes the model components and loads the corresponding state dictionaries,
        moving them to the specified device and data type.
        """
        # Reformat the state dict for a single linear layer
        fc_prior_state_dict = OrderedDict()
        fc_prior_state_dict['weight'] = self.filtered_state_dict_fc_prior.get('weight')
        if 'bias' in self.filtered_state_dict_fc_prior:
             fc_prior_state_dict['bias'] = self.filtered_state_dict_fc_prior.get('bias')

        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_prior.load_state_dict(fc_prior_state_dict)
        self.fc_prior.to(device=device, dtype=dtype)

        fc_post_a_state_dict = OrderedDict()
        fc_post_a_state_dict['weight'] = self.filtered_state_dict_fc_post_a.get('weight')
        if 'bias' in self.filtered_state_dict_fc_post_a:
             fc_post_a_state_dict['bias'] = self.filtered_state_dict_fc_post_a.get('bias')

        self.fc_post_a = nn.Linear(2048, 1024)
        self.fc_post_a.load_state_dict(fc_post_a_state_dict)
        self.fc_post_a.to(device=device, dtype=dtype)

        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.SemanticEncoder_module.load_state_dict(self.filtered_state_dict_semantic_encoder)
        self.SemanticEncoder_module.to(device=device, dtype=dtype)

        self.CodecEnc = CodecEncoder_Transformer()
        self.CodecEnc.load_state_dict(self.filtered_state_dict_codec)
        self.CodecEnc.to(device=device, dtype=dtype)

        self.generator = CodecDecoderVocos()
        self.generator.load_state_dict(self.filtered_state_dict_gen)
        self.generator.to(device=device, dtype=dtype)

        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0",
            output_hidden_states=True
        )
        self.semantic_model.eval()
        self.semantic_model.to(device=device, dtype=dtype)
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

class XCodec2Model():
    def __init__(self, model_path: str, device: str, dtype: torch.dtype = None):
        
        model_checkpoint = ModelCheckpointLoader(model_path)
        model_checkpoint._get_model(device, dtype)

        self.semantic_model = model_checkpoint.semantic_model
        self.SemanticEncoder_module = model_checkpoint.SemanticEncoder_module
        self.CodecEnc = model_checkpoint.CodecEnc
        self.generator = model_checkpoint.generator
        self.fc_prior = model_checkpoint.fc_prior
        self.fc_post_a = model_checkpoint.fc_post_a
        self.feature_extractor = model_checkpoint.feature_extractor
        self.device = 'cuda'

    def forward(self, input_waveform, sample_rate=16000):

        input_features = self.feature_extractor(
            input_waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)  # [batch, frames, feat_dim]

        semantic_output = self.semantic_model(input_features)
        semantic_hidden_16 = semantic_output.hidden_states[16]  # 取第16层
        semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
        semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

        wav = input_waveform.unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
        vq_emb = self.CodecEnc(wav)  # [batch, time//down, 1024] 只是示例
        vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

        if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
            vq_emb = vq_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]

        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 1024 + 1024, frames]

        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        _, vq_code, _ = self.generator(concat_emb, vq=True)
        vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)

        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)

        recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]
        return recon_audio

    def encode_code(self, input_waveform, sample_rate=16000):
        
        with torch.no_grad():
            input_features = self.feature_extractor(
                input_waveform, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)  # [batch, frames, feat_dim]

            semantic_output = self.semantic_model(input_features)
            semantic_hidden_16 = semantic_output.hidden_states[16]  
            semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
            semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

            wav = input_waveform.unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
            vq_emb = self.CodecEnc(wav) 
            vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

            if vq_emb.shape[-1] != semantic_encoded.shape[-1]:
                min_len = min(vq_emb.shape[-1], semantic_encoded.shape[-1])
                vq_emb = vq_emb[:, :, :min_len]
                semantic_encoded = semantic_encoded[:, :, :min_len]

            concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

            concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

            _, vq_code, _ = self.generator(concat_emb, vq=True)
            return vq_code

    def decode_code(self, vq_code):
        with torch.no_grad():
            vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
            vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]

            vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)  # [batch, 1024, frames]

            recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]  # [batch, time]
            return recon_audio
