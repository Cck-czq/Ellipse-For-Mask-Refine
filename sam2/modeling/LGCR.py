from torch import nn
import torch
class PureLGCR(nn.Module):
    def __init__(self):
        super(PureLGCR,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.model(x)
        return x

def lgcr_for_channel():
    import torch
    import numpy as np
    model = PureLGCR().to("cuda")
    model.train()

    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 加载 + 强制开启梯度（解决你所有的报错！）
    text_feat = torch.from_numpy(np.load("text_feat.npy")).cuda().float()
    feat = torch.from_numpy(np.load("fpn2.npy")).cuda().float()

    # 强制开启梯度，切断SAM计算图，允许反向传播
    feat = feat.clone().requires_grad_(True)
    text_feat = text_feat.clone().detach()

    # print(f"text_feat:{text_feat}")
    B, C, H, W = feat.shape  # [1,256,64,64]
    #train
    with torch.enable_grad():
        for epoch in range(15):
            weights=model(text_feat)

            weights = weights.view(B, C, 1, 1)

            weighted_feats=feat*weights
            loss = -torch.cosine_similarity(
                weighted_feats.flatten(1),  # 展平成向量
                feat.flatten(1),
                dim=-1
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    model.eval()
    with torch.no_grad():
        weighted_feat=feat*model(text_feat).view(B, C, 1, 1)
        return weighted_feat