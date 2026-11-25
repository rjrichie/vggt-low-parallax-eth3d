import glob, sys, time, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

imgs = sorted(glob.glob(sys.argv[1] + "/*"))[: int(sys.argv[2])]
dev = "mps"
t = time.time()
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
print(f"model loaded in {time.time()-t:.0f}s")
x = load_and_preprocess_images(imgs).to(dev)
t = time.time()
with torch.no_grad():
    pred = model(x)
ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
print(f"{len(imgs)} images, inference {time.time()-t:.1f}s, extrinsics {tuple(ext.shape)}")
print("finite:", torch.isfinite(ext).all().item())
