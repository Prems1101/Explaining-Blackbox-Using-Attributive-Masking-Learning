import torch
import torch.nn as nn
from transformers import AutoModel


class AttributionModel(nn.Module):
    """
    Attribution model Gθ .

    Architecture:
    -  backbone 
    - yv encoded via class embeddings Z and z0 token prepended to input
    - Shared MLP head: d → d → 1 per token (tanh hidden, sigmoid output)
    """

    def __init__(self, backbone_name="distilbert-base-uncased", num_classes=2, hidden_dim=768):
        super().__init__()

        self.hidden_dim = hidden_dim

        
        self.backbone = AutoModel.from_pretrained(backbone_name)

        
        self.z0 = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode_yv(self, y_probs):
        """zv = z0 + sum_j( F(xv)[j] * z_j )  — """
        weighted = torch.einsum("bc,cd->bd", y_probs, self.class_embeddings)
        return self.z0.unsqueeze(0) + weighted  # [B, D]

    def forward(self, input_ids, attention_mask, y_probs):
        """
        input_ids:      [B, T]
        attention_mask: [B, T]
        y_probs:        [B, C]
        Returns scores: [B, T]  in [0, 1]
        """
        B = input_ids.shape[0]

        
        zv = self.encode_yv(y_probs).unsqueeze(1) 

        
        token_embeds = self.backbone.embeddings.word_embeddings(input_ids)  

        # Prepend zv token ("append zv to xv and forward through backbone")
        combined = torch.cat([zv, token_embeds], dim=1)     
        extra_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        ext_mask = torch.cat([extra_mask, attention_mask], dim=1)  

    
        hidden = self.backbone(inputs_embeds=combined, attention_mask=ext_mask).last_hidden_state

        
        token_hidden = hidden[:, 1:, :]  

        scores = self.mlp(token_hidden).squeeze(-1)  
        return scores
