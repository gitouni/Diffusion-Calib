from models.tools.scn import SparseFeatureEncoder, sparse_tensor_collate
import torch

if __name__ == "__main__":
    model = SparseFeatureEncoder(in_chan=1).cuda()
    coords = torch.rand(2, 1000, 3).cuda()
    feats = torch.ones(2*1000, 1).cuda()
    feats = model([sparse_tensor_collate(coords, 100, 0), feats, 2])
    for feat in feats:
        feat = feat.features
        print(feat.shape)
        print(feat[:2,:5])