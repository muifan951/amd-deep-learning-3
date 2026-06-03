import torch, torch.nn as nn
class TST(nn.Module):
    def __init__(s, inp=7, d=128, heads=8, layers=4, pred=24):
        super().__init__()
        s.enc = nn.Linear(inp, d)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d, nhead=heads, batch_first=True), num_layers=layers)
        s.dec = nn.Linear(d, pred)
    def forward(s, x): return s.dec(s.tf(s.enc(x))[:, -1, :])
if __name__ == "__main__":
    d = "cuda" if torch.cuda.is_available() else "cpu"
    m = TST().to(d)
    print(f"TST on {torch.cuda.get_device_name(0)}, {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    print(f"Output: {m(torch.randn(64,168,7,device=d)).shape}")