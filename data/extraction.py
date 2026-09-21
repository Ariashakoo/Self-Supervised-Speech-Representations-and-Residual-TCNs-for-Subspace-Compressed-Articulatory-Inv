import gc
import torch
import numpy as np
from transformers import WavLMModel
from utils.signal_processing import fix_audio

@torch.no_grad()
def extract_wavlm_features(audio_arrays, device, orig_sr=None, target_sr=16000):
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    wavlm_model.eval()

    all_features = []
    hidden_size = getattr(wavlm_model.config, "hidden_size", 1024)

    for i, audio in enumerate(audio_arrays):
        x = fix_audio(audio, orig_sr=orig_sr, target_sr=target_sr)
        
        if len(x) < 400:
            all_features.append(np.zeros((1, hidden_size), dtype=np.float32))
            continue

        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
        outputs = wavlm_model(x_tensor)
        feats = outputs.last_hidden_state.squeeze(0).float().cpu().numpy()
        
        all_features.append(feats.astype(np.float32))
        del x_tensor, outputs
        
        if i % 20 == 0:
            torch.cuda.empty_cache()

    wavlm_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return all_features