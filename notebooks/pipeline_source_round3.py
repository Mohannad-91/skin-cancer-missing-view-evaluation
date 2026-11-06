# %% [markdown]
# # Know When to Look Closer — MILK10k pipeline (round 3: reviewer experiments)
#
# **Round 3.** Set `RUN` in the CONFIG cell: `"A"` = PanDerm-Base (two extra training seeds of the joint model,
# a modality-dropout baseline, a mean-imputation baseline); `"B"` = ConvNeXt-Tiny with the PanDerm recipe
# (12 epochs, no distillation) plus external tests. Folds are unchanged (seed 42).
# Availability-robust, calibrated fusion of clinical (C) and dermoscopic (D) images
# with a dermoscopy-referral head.
#
# **What this notebook does**
# 1. Finds MILK10k in `/kaggle/input`, builds one row per lesion (C image, D image, metadata, 11-class label).
# 2. Caches resized images once (fast training).
# 3. Trains one model that predicts in three view states: both (C+D), clinical-only, dermoscopic-only.
#    Loss = CE on each state + unimodal heads + distillation from the paired state + referral head.
# 4. Lesion-level stratified 5-fold CV; per-fold inner validation for model selection and temperature scaling.
# 5. Reports: macro-F1, balanced accuracy, AUROC, malignant sensitivity/specificity, simplified hierarchical
#    error, ECE/NLL/Brier (global vs per-state temperature), skin-tone subgroups, bootstrap 95% CIs,
#    and dermoscopy-referral curves.
# 6. Optional: MILK10k benchmark submission file; external tests (Derm7pt, PAD-UFES-20, DDI, MIDAS).
#
# Run the cells in order. Change settings only in the **CONFIG** cell.

# %% Imports
import os, glob, json, math, random, time, copy, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import timm
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "| torch", torch.__version__, "| timm", timm.__version__)

# %% CONFIG — the only cell you normally edit
CFG = dict(
    data_root="/kaggle/input",          # searched recursively for the MILK10k files
    out_dir="/kaggle/working/outputs",
    cache_dir="/kaggle/working/cache",
    cache_short_side=288,               # images resized once to this short side
    backbone="panderm_base",            # "panderm_base" (dermatology ViT-B/16) or any timm name, e.g. "convnext_tiny.fb_in22k_ft_in1k"
    pretrained=True,                    # timm backbones need Kaggle "Internet on"; PanDerm loads from the attached dataset
    panderm_ckpt=None,                  # None = search /kaggle/input for panderm_bb_data6_checkpoint-499.pth
    drop_path=0.1,                      # stochastic depth (ViT)
    layer_decay=0.75,                   # layer-wise LR decay for ViT backbones (1.0 = off)
    img_size=224,
    epochs=12,
    batch_size=32,
    lr_backbone=3e-5,
    lr_head=1e-3,
    weight_decay=0.05,
    warmup_epochs=1,
    n_folds=5,
    folds_to_run=[0, 1, 2, 3, 4],
    inner_val_frac=0.10,
    seed=42,
    use_meta=True,                      # age, sex, site
    tone_as_input=True,                 # skin tone class as an input feature
    meta_dropout=0.2,                   # prob. of hiding all metadata in training (for external sets)
    use_monet=False,                    # MONET concept scores (per view, masked with the view)
    lambda_uni=0.5,
    lambda_kd=0.0,                      # round 1: distillation gave no significant gain -> off
    kd_T=2.0,
    lambda_ref=0.5,
    tta=True,                           # horizontal-flip test-time augmentation
    num_workers=4,
    amp=True,
    # experiments to run, in order. 'full' is the proposed model.
    experiments=["full", "paired_only", "clin_only", "derm_only"],
    save_models=True,                   # needed for the benchmark submission and external tests
    referral_sens_tolerance=0.01,       # allowed drop in sensitivity/specificity vs. always-dermoscopy
    target_sens=0.95,                   # screening operating point: threshold set on inner validation per state
    n_boot=1000,
)
# ---- round 3 -------------------------------------------------------------------
RUN = "A"           # "A" = PanDerm-B reviewer experiments, "B" = ConvNeXt-T matched recipe
if RUN == "A":
    CFG.update(experiments=["moddrop", "paired_meanimp", "full_s1", "full_s2"],
               external_experiments=["moddrop", "full_s1", "full_s2"], submission_experiments=["moddrop"])
elif RUN == "B":
    CFG.update(backbone="convnext_tiny.fb_in22k_ft_in1k", layer_decay=1.0, lr_backbone=1e-4, drop_path=0.1,
               epochs=12, experiments=["full"], external_experiments=["full"], submission_experiments=["full"])
QUICK_RUN = False   # True = 1 fold, 2 epochs, first experiment only: a sanity check (~15 min)
if QUICK_RUN:
    CFG.update(epochs=2, folds_to_run=[0], experiments=CFG["experiments"][:2], external_experiments=[],
               submission_experiments=[])

# local smoke test hook (ignored on Kaggle)
if os.environ.get("SKINREF_TEST"):
    CFG.update(json.loads(os.environ["SKINREF_TEST"]))

os.makedirs(CFG["out_dir"], exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])

# %% Classes, malignancy map, simplified hierarchy
CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
N_CLS = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}
# AKIEC (actinic keratosis / intraepidermal carcinoma) is counted as malignant here; state this in the paper.
MALIGNANT = ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"]
MAL_IDX = np.array([C2I[c] for c in MALIGNANT])
IS_MAL = np.isin(np.arange(N_CLS), MAL_IDX)

def hier_distance(y_true, y_pred):
    """Simplified hierarchical error: 0 correct, 1 wrong class within the same malignancy group,
    2 wrong malignancy group. (The official MILK10k metric uses the ISIC-DX tree.)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    d = np.where(y_true == y_pred, 0, np.where(IS_MAL[y_true] == IS_MAL[y_pred], 1, 2))
    return d.mean()

# %% Locate MILK10k files
def _read_header(p):
    try:
        return list(pd.read_csv(p, nrows=2).columns)
    except Exception:
        return []

def locate_milk10k(root):
    csvs = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    gt, per_image, test_meta = None, [], None
    for p in csvs:
        cols = _read_header(p)
        name = os.path.basename(p).lower()
        if "lesion_id" not in cols:
            continue
        n_cls_cols = sum(c in cols for c in CLASSES)
        if n_cls_cols >= 8 and "test" not in name:
            gt = gt or p
        elif "image_type" in cols or "isic_id" in cols:
            if "test" in name:
                test_meta = test_meta or p
            else:
                per_image.append(p)
    if gt is None:
        raise FileNotFoundError(
            "MILK10k ground-truth CSV not found under %s. Add the dataset with 'Add Input' and check the path." % root)
    milk_dir = os.path.dirname(gt)
    # go up until the folder holds the images too
    for _ in range(3):
        if glob.glob(os.path.join(milk_dir, "**", "*.jpg"), recursive=True):
            break
        milk_dir = os.path.dirname(milk_dir)
    return dict(gt=gt, per_image=per_image, test_meta=test_meta, milk_dir=milk_dir)

FILES = locate_milk10k(CFG["data_root"])
print(json.dumps(FILES, indent=1))

def image_index(root):
    idx = {}
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        for p in glob.glob(os.path.join(root, "**", "*." + ext), recursive=True):
            idx[os.path.splitext(os.path.basename(p))[0]] = p
    return idx

IMG_INDEX = image_index(FILES["milk_dir"])
print("images found:", len(IMG_INDEX))

# %% Build the lesion table
def load_per_image(paths):
    df = None
    for p in paths:
        d = pd.read_csv(p)
        key = "isic_id" if "isic_id" in d.columns else None
        if df is None:
            df = d
        elif key and "isic_id" in df.columns:
            new = [c for c in d.columns if c not in df.columns or c == "isic_id"]
            df = df.merge(d[new], on="isic_id", how="left")
    return df

def view_of(s):
    s = str(s).lower()
    if s.startswith("clinical"):
        return "C"
    if s.startswith("dermoscop") or s.startswith("dermatoscop"):
        return "D"
    return None

def build_lesion_table(per_image_paths, gt_path=None, img_index=None):
    im = load_per_image(per_image_paths)
    im["view"] = im["image_type"].map(view_of)
    im = im[im["view"].notna()].copy()
    im["path"] = im["isic_id"].map(img_index)
    site_col = "site" if "site" in im.columns else ("anatom_site_general" if "anatom_site_general" in im.columns else None)
    monet_cols = sorted([c for c in im.columns if c.upper().startswith("MONET")])
    rows = []
    for lid, g in im.groupby("lesion_id"):
        c, d = g[g.view == "C"], g[g.view == "D"]
        if len(c) == 0 or len(d) == 0:
            continue
        c, d = c.iloc[0], d.iloc[0]
        r = dict(lesion_id=lid, isic_c=c["isic_id"], isic_d=d["isic_id"], path_c=c["path"], path_d=d["path"],
                 age=c.get("age_approx", np.nan), sex=c.get("sex", np.nan),
                 site=c.get(site_col, np.nan) if site_col else np.nan,
                 tone=c.get("skin_tone_class", np.nan))
        for m in monet_cols:
            r["c_" + m] = c.get(m, np.nan)
            r["d_" + m] = d.get(m, np.nan)
        rows.append(r)
    les = pd.DataFrame(rows)
    if gt_path is not None:
        gt = pd.read_csv(gt_path)
        gt["label"] = gt[CLASSES].values.argmax(1)
        les = les.merge(gt[["lesion_id", "label"]], on="lesion_id", how="inner")
    missing = les["path_c"].isna() | les["path_d"].isna()
    if missing.any():
        print("WARNING: %d lesions without an image file, dropped" % missing.sum())
        les = les[~missing]
    return les.reset_index(drop=True), monet_cols

LES, MONET_COLS = build_lesion_table(FILES["per_image"], FILES["gt"], IMG_INDEX)
if CFG.get("max_lesions"):   # smoke tests only
    LES = LES.sample(CFG["max_lesions"], random_state=0).sort_values("lesion_id").reset_index(drop=True)
print("lesions:", len(LES))
print(LES["label"].map(dict(enumerate(CLASSES))).value_counts())
print("skin tone:", LES["tone"].value_counts(dropna=False).sort_index().to_dict())

# %% Metadata encoder
class MetaEncoder:
    def __init__(self, df, use_meta=True, tone_as_input=True):
        self.use_meta, self.tone = use_meta, tone_as_input
        self.sites = sorted([s for s in df["site"].dropna().astype(str).unique()])
        self.dim = (2 + 3 + len(self.sites) + 1 + (2 if tone_as_input else 0)) if use_meta else 0

    def encode(self, age=np.nan, sex=np.nan, site=np.nan, tone=np.nan):
        if not self.use_meta:
            return np.zeros(0, np.float32)
        v = []
        a_ok = not pd.isna(age)
        v += [float(age) / 100.0 if a_ok else 0.0, 0.0 if a_ok else 1.0]
        s = str(sex).lower() if not pd.isna(sex) else ""
        v += [s == "male", s == "female", s not in ("male", "female")]
        st = str(site) if not pd.isna(site) else None
        v += [st == x for x in self.sites] + [st not in self.sites]
        if self.tone:
            t_ok = not pd.isna(tone)
            v += [float(tone) / 5.0 if t_ok else 0.0, 0.0 if t_ok else 1.0]
        return np.asarray(v, np.float32)

    def unknown(self):
        return self.encode()

META = MetaEncoder(LES, CFG["use_meta"], CFG["tone_as_input"])
print("meta dim:", META.dim)

# %% Image cache (resize once)
def _cache_one(args):
    src, dst, short = args
    if os.path.exists(dst):
        return
    try:
        im = Image.open(src)
        im.draft("RGB", (short * 2, short * 2))
        im = im.convert("RGB")
        w, h = im.size
        s = short / min(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im.save(dst, quality=95)
    except Exception as e:
        print("cache failed:", src, e)

def build_cache(paths, cache_dir, short):
    os.makedirs(cache_dir, exist_ok=True)
    jobs, out = [], {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(cache_dir, stem + ".jpg")
        jobs.append((p, dst, short)); out[p] = dst
    t0 = time.time()
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        pool.map(_cache_one, jobs, chunksize=16)
    print("cached %d images in %.0fs" % (len(jobs), time.time() - t0))
    return out

_map = build_cache(list(LES["path_c"]) + list(LES["path_d"]), CFG["cache_dir"], CFG["cache_short_side"])
LES["cpath_c"] = LES["path_c"].map(_map)
LES["cpath_d"] = LES["path_d"].map(_map)

# %% Dataset and transforms
def make_transforms(img_size, mean, std):
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(0.2, 0.2, 0.1, 0.02),
        T.ToTensor(), T.Normalize(mean, std)])
    test = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean, std)])
    return train, test

class LesionDS(Dataset):
    def __init__(self, df, meta_enc, tf, train=False, meta_dropout=0.0, use_monet=False,
                 has_c=True, has_d=True):
        self.df, self.meta, self.tf, self.train = df.reset_index(drop=True), meta_enc, tf, train
        self.meta_dropout, self.use_monet = meta_dropout, use_monet
        self.has_c, self.has_d = has_c, has_d
        self.meta_vecs = np.stack([meta_enc.encode(r.age, r.sex, r.site, r.tone) for r in self.df.itertuples()]) \
            if meta_enc.dim else np.zeros((len(self.df), 0), np.float32)
        if use_monet and MONET_COLS:
            # external sets have no MONET scores -> zeros (keep use_monet=False for external claims)
            self.mc = self.df.reindex(columns=["c_" + m for m in MONET_COLS]).fillna(0).values.astype(np.float32)
            self.md = self.df.reindex(columns=["d_" + m for m in MONET_COLS]).fillna(0).values.astype(np.float32)
        else:
            self.mc = self.md = np.zeros((len(self.df), 0), np.float32)
        self.blank = None

    def __len__(self):
        return len(self.df)

    def _img(self, p):
        return self.tf(Image.open(p).convert("RGB"))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        xc = self._img(r["cpath_c"]) if self.has_c else torch.zeros(3, CFG["img_size"], CFG["img_size"])
        xd = self._img(r["cpath_d"]) if self.has_d else torch.zeros(3, CFG["img_size"], CFG["img_size"])
        m = self.meta_vecs[i]
        if self.train and self.meta.dim and random.random() < self.meta_dropout:
            m = self.meta.unknown()
        y = int(r["label"]) if "label" in r and not pd.isna(r["label"]) else -1
        return xc, xd, torch.from_numpy(m), torch.from_numpy(self.mc[i]), torch.from_numpy(self.md[i]), y

# %% Model
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def find_panderm(cfg):
    if cfg.get("panderm_ckpt"):
        return cfg["panderm_ckpt"]
    hits = glob.glob(os.path.join(cfg["data_root"], "**", "panderm_bb_data6_checkpoint-499.pth"), recursive=True)
    if not hits:
        raise FileNotFoundError("PanDerm-Base checkpoint not found: attach the 'panderm-base-weights' dataset.")
    return hits[0]

def convert_panderm(sd, depth=12):
    """PanDerm-Base (BEiT/CAE-style ViT-B/16) keys -> timm VisionTransformer keys.
    gamma_1/2 -> ls1/2.gamma; q_bias + v_bias -> qkv.bias (k bias = 0); final norm -> fc_norm (mean pooling)."""
    for k in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("encoder.", "")
        if k.endswith("attn.q_bias") or k.endswith("attn.v_bias") or k.endswith("relative_position_index"):
            continue
        k2 = k.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma")
        if k2.startswith("norm."):
            k2 = "fc_norm." + k2[len("norm."):]
        out[k2] = v
    for i in range(depth):
        q, v = sd.get(f"blocks.{i}.attn.q_bias"), sd.get(f"blocks.{i}.attn.v_bias")
        if q is not None and v is not None:
            out[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q, torch.zeros_like(q), v])
    return out

def make_encoder(cfg):
    if cfg["backbone"] == "panderm_base":
        enc = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="avg",
                                init_values=1e-5, drop_path_rate=cfg.get("drop_path", 0.0))
        enc.panderm = True
        if cfg["pretrained"]:
            sd = convert_panderm(torch.load(find_panderm(cfg), map_location="cpu", weights_only=False))
            msg = enc.load_state_dict(sd, strict=False)
            bad = [k for k in msg.missing_keys if not k.startswith("head")]
            print("PanDerm load: missing %d %s | unexpected %d %s" % (len(bad), bad[:4], len(msg.unexpected_keys),
                                                                      msg.unexpected_keys[:4]))
            if bad:
                raise RuntimeError("PanDerm weights did not map onto the ViT - stop and report this message.")
        return enc
    enc = timm.create_model(cfg["backbone"], pretrained=cfg["pretrained"], num_classes=0,
                            drop_path_rate=cfg.get("drop_path", 0.0))
    enc.panderm = False
    return enc

def enc_norm(enc):
    if getattr(enc, "panderm", False):
        return IMAGENET_MEAN, IMAGENET_STD
    d = timm.data.resolve_data_config({}, model=enc)
    return d["mean"], d["std"]

def backbone_param_groups(enc, lr, decay, wd):
    """Layer-wise LR decay for ViT (blocks.N); one group otherwise."""
    if not hasattr(enc, "blocks") or decay >= 1.0:
        return [{"params": list(enc.parameters()), "lr": lr, "weight_decay": wd}]
    n = len(enc.blocks) + 1
    groups = {}
    for name, p in enc.named_parameters():
        if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
            lid = 0
        elif name.startswith("blocks."):
            lid = int(name.split(".")[1]) + 1
        else:
            lid = n
        no_wd = p.ndim == 1 or name.endswith(".bias") or name in ("cls_token", "pos_embed")
        key = (lid, no_wd)
        groups.setdefault(key, {"params": [], "lr": lr * decay ** (n - lid), "weight_decay": 0.0 if no_wd else wd})
        groups[key]["params"].append(p)
    return list(groups.values())

def parallel(model):
    return nn.DataParallel(model) if DEVICE == "cuda" and torch.cuda.device_count() > 1 else model

class RefNet(nn.Module):
    """Two encoders, learned 'missing' tokens, a fusion head used in all three view states,
    unimodal heads, and a referral head that scores the expected value of adding dermoscopy."""
    def __init__(self, cfg, meta_dim, monet_dim, d=512):
        super().__init__()
        self.enc_c, self.enc_d = make_encoder(cfg), make_encoder(cfg)
        f = self.enc_c.num_features
        self.proj_c = nn.Sequential(nn.Linear(f + monet_dim, d), nn.GELU(), nn.Dropout(0.2))
        self.proj_d = nn.Sequential(nn.Linear(f + monet_dim, d), nn.GELU(), nn.Dropout(0.2))
        self.miss_c = nn.Parameter(torch.zeros(d))
        self.miss_d = nn.Parameter(torch.zeros(d))
        self.meta_dim = meta_dim
        md = 64 if meta_dim else 0
        self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.GELU(), nn.Dropout(0.1)) if meta_dim else None
        self.fuse = nn.Sequential(nn.LayerNorm(2 * d + md + 2), nn.Linear(2 * d + md + 2, d), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(d, N_CLS))
        self.head_c, self.head_d = nn.Linear(d, N_CLS), nn.Linear(d, N_CLS)
        self.ref = nn.Sequential(nn.Linear(d + md, 128), nn.GELU(), nn.Linear(128, 1))

    def backbone_params(self):
        return list(self.enc_c.parameters()) + list(self.enc_d.parameters())

    def head_params(self):
        ids = {id(p) for p in self.backbone_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def _meta(self, m):
        return self.meta(m) if self.meta is not None else m[:, :0]

    def encode_c(self, xc, mc):
        return self.proj_c(torch.cat([self.enc_c(xc), mc], 1))

    def encode_d(self, xd, md):
        return self.proj_d(torch.cat([self.enc_d(xd), md], 1))

    def fuse_logits(self, zc, zd, hm, has_c, has_d):
        B = hm.shape[0]
        zc = zc if has_c else self.miss_c.expand(B, -1)
        zd = zd if has_d else self.miss_d.expand(B, -1)
        if getattr(self, "flag_both", False):     # mean-imputation baseline: the head is never told a view is missing
            has_c = has_d = True
        flags = torch.tensor([float(has_c), float(has_d)], device=hm.device).expand(B, 2)
        return self.fuse(torch.cat([zc, zd, hm, flags], 1))

    def forward(self, xc, xd, m, mc, md, states=("both", "c", "d")):
        hm = self._meta(m)
        need_c = any(s in ("both", "c") for s in states)
        need_d = any(s in ("both", "d") for s in states)
        zc = self.encode_c(xc, mc) if need_c else None
        zd = self.encode_d(xd, md) if need_d else None
        out = {}
        if "both" in states: out["both"] = self.fuse_logits(zc, zd, hm, True, True)
        if "c" in states:    out["c"] = self.fuse_logits(zc, None, hm, True, False)
        if "d" in states:    out["d"] = self.fuse_logits(None, zd, hm, False, True)
        if zc is not None:
            out["uni_c"] = self.head_c(zc)
            out["ref"] = self.ref(torch.cat([zc.detach(), hm.detach()], 1)).squeeze(1)
        if zd is not None:
            out["uni_d"] = self.head_d(zd)
        return out

EXPERIMENTS = {
    # name: (states trained, states evaluated, kd, referral)
    "full":        (("both", "c", "d"), ("both", "c", "d"), True,  True),
    "no_kd":       (("both", "c", "d"), ("both", "c", "d"), False, True),
    "paired_only": (("both",),          ("both", "c", "d"), False, False),  # naive: missing view -> untrained token
    "clin_only":   (("c",),             ("c",),             False, False),
    "derm_only":   (("d",),             ("d",),             False, False),
    # round 3 (reviewer requests)
    "moddrop":        (("both", "c", "d"), ("both", "c", "d"), False, False),  # one random view state per lesion
    "paired_meanimp": (("both",),          ("both", "c", "d"), False, False),  # missing view -> mean training embedding
    "full_s1":        (("both", "c", "d"), ("both", "c", "d"), False, True),   # joint model, training seed +1000
    "full_s2":        (("both", "c", "d"), ("both", "c", "d"), False, True),   # joint model, training seed +2000
}
MODDROP = {"moddrop": (0.5, 0.25, 0.25)}          # P(both), P(clinical only), P(dermoscopic only) per lesion
MEANIMP = {"paired_meanimp"}
SEED_OFFSET = {"full_s1": 1000, "full_s2": 2000}

# %% Training utilities
def p_malignant(logits):
    p = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), 1).numpy()
    return p[:, MAL_IDX].sum(1)

def class_weights(labels):
    n = np.bincount(labels, minlength=N_CLS).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(n, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)

def kd_loss(student, teacher, T_):
    return F.kl_div(F.log_softmax(student / T_, 1), F.softmax(teacher.detach() / T_, 1),
                    reduction="batchmean") * T_ * T_

def moddrop_loss(out, y, w, probs, cfg):
    """Modality dropout: each lesion is trained in one randomly drawn view state."""
    B = y.shape[0]
    r = torch.rand(B, device=y.device)
    st = torch.where(r < probs[0], 0, torch.where(r < probs[0] + probs[1], 1, 2))
    ce_n = lambda lg: F.cross_entropy(lg, y, weight=w, reduction="none")
    wy = w[y]
    ce_all = torch.stack([ce_n(out["both"]), ce_n(out["c"]), ce_n(out["d"])])
    loss = ce_all[st, torch.arange(B, device=y.device)].sum() / wy.sum()
    mc, md = st != 2, st != 1
    if mc.any(): loss = loss + cfg["lambda_uni"] * ce_n(out["uni_c"])[mc].sum() / wy[mc].sum()
    if md.any(): loss = loss + cfg["lambda_uni"] * ce_n(out["uni_d"])[md].sum() / wy[md].sum()
    return loss

def compute_loss(out, y, w, exp_cfg, cfg, exp_name=None):
    if exp_name in MODDROP:
        return moddrop_loss(out, y, w, MODDROP[exp_name], cfg)
    states, _, use_kd, use_ref = exp_cfg
    ce = lambda lg: F.cross_entropy(lg, y, weight=w)
    loss = sum(ce(out[s]) for s in states)
    if "uni_c" in out and "c" in states:
        loss = loss + cfg["lambda_uni"] * ce(out["uni_c"])
    if "uni_d" in out and "d" in states:
        loss = loss + cfg["lambda_uni"] * ce(out["uni_d"])
    if use_kd and cfg["lambda_kd"] > 0:
        loss = loss + cfg["lambda_kd"] * (kd_loss(out["c"], out["both"], cfg["kd_T"]) +
                                          kd_loss(out["d"], out["both"], cfg["kd_T"]))
    if use_ref:
        with torch.no_grad():
            pb = torch.softmax(out["both"].float(), 1)[:, MAL_IDX].sum(1)
            pc = torch.softmax(out["c"].float(), 1)[:, MAL_IDX].sum(1)
            target = (pb - pc).abs().clamp(0, 1)
        loss = loss + cfg["lambda_ref"] * F.binary_cross_entropy_with_logits(out["ref"].float(), target)
    return loss

@torch.no_grad()
def predict(model, loader, eval_states, tta=False):
    model.eval()
    acc = {}
    for xc, xd, m, mc, md, _ in loader:
        xc, xd, m, mc, md = [t.to(DEVICE, non_blocking=True) for t in (xc, xd, m, mc, md)]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["amp"] and DEVICE == "cuda"):
            out = model(xc, xd, m, mc, md, states=eval_states)
            if tta:
                o2 = model(xc.flip(3), xd.flip(3), m, mc, md, states=eval_states)
                out = {k: (v + o2[k]) / 2 for k, v in out.items()}
        for k, v in out.items():
            acc.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}

def macro_f1(y, logits):
    return f1_score(y, logits.argmax(1), average="macro", labels=list(range(N_CLS)), zero_division=0)

def train_fold(exp_name, tr_df, va_df, fold):
    exp_cfg = EXPERIMENTS[exp_name]
    states, eval_states = exp_cfg[0], exp_cfg[1]
    model = RefNet(CFG, META.dim, len(MONET_COLS) if CFG["use_monet"] else 0).to(DEVICE)
    mean, std = enc_norm(model.enc_c)
    tf_tr, tf_te = make_transforms(CFG["img_size"], mean, std)
    net = parallel(model)
    need_c = any(s in ("both", "c") for s in states)
    need_d = any(s in ("both", "d") for s in states)
    ds_tr = LesionDS(tr_df, META, tf_tr, True, CFG["meta_dropout"], CFG["use_monet"], need_c, need_d)
    ds_va = LesionDS(va_df, META, tf_te, False, 0, CFG["use_monet"])
    dl_tr = DataLoader(ds_tr, CFG["batch_size"], shuffle=True, num_workers=CFG["num_workers"],
                       pin_memory=DEVICE == "cuda", drop_last=len(ds_tr) > CFG["batch_size"])
    dl_va = DataLoader(ds_va, CFG["batch_size"] * 2, shuffle=False, num_workers=CFG["num_workers"])
    w = class_weights(tr_df["label"].values).to(DEVICE)
    opt = torch.optim.AdamW(
        backbone_param_groups(model.enc_c, CFG["lr_backbone"], CFG["layer_decay"], CFG["weight_decay"]) +
        backbone_param_groups(model.enc_d, CFG["lr_backbone"], CFG["layer_decay"], CFG["weight_decay"]) +
        [{"params": model.head_params(), "lr": CFG["lr_head"], "weight_decay": CFG["weight_decay"]}])
    steps = CFG["epochs"] * max(1, len(dl_tr)); warm = CFG["warmup_epochs"] * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=CFG["amp"] and DEVICE == "cuda")
    best, best_state, log = -1, None, []
    for ep in range(CFG["epochs"]):
        model.train(); t0 = time.time(); tot = 0.0
        for xc, xd, m, mc, md, y in dl_tr:
            xc, xd, m, mc, md, y = [t.to(DEVICE, non_blocking=True) for t in (xc, xd, m, mc, md, y)]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["amp"] and DEVICE == "cuda"):
                out = net(xc, xd, m, mc, md, states=states)
            loss = compute_loss({k: v.float() for k, v in out.items()}, y, w, exp_cfg, CFG, exp_name)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        pv = predict(net, dl_va, eval_states)
        yv = va_df["label"].values
        sel_states = [s for s in eval_states if s in states] or list(eval_states)
        score = float(np.mean([macro_f1(yv, pv[s]) for s in sel_states]))
        log.append(dict(epoch=ep, loss=tot / max(1, len(dl_tr)), val_macro_f1=score, sec=time.time() - t0))
        print("[%s f%d] ep %d loss %.3f val mF1 %.4f (%.0fs)" % (exp_name, fold, ep, log[-1]["loss"], score, log[-1]["sec"]))
        if score > best:
            best, best_state = score, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model, tf_te, log

# %% Calibration and metrics
def fit_temperature(logits, y):
    """Grid search on NLL (robust; LBFGS can diverge on small validation sets)."""
    L = torch.tensor(logits, dtype=torch.float32); Y = torch.tensor(y, dtype=torch.long)
    grid = torch.exp(torch.linspace(math.log(0.05), math.log(20.0), 400))
    with torch.no_grad():
        nll = torch.stack([F.cross_entropy(L / t, Y) for t in grid])
    nll = torch.nan_to_num(nll, nan=float("inf"))
    return float(grid[int(nll.argmin())])

def softmax_np(lg):
    return torch.softmax(torch.tensor(lg, dtype=torch.float32), 1).numpy()

def ece_score(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.any():
            e += msk.mean() * abs(acc[msk].mean() - conf[msk].mean())
    return float(e)

def thr_for_sens(pm, ym, target):
    """Largest threshold whose sensitivity on (pm, ym) is >= target."""
    pos = np.sort(pm[ym == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])

def metrics_block(y, p, thr=0.5, thr_op=None):
    """y: int labels; p: probabilities (n, 11); thr_op: per-lesion operating-point thresholds (target sensitivity)."""
    pred = p.argmax(1)
    ym = IS_MAL[y].astype(int); pm = p[:, MAL_IDX].sum(1)
    out = dict(n=int(len(y)), accuracy=float((pred == y).mean()),
               balanced_acc=float(balanced_accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average="macro", labels=list(range(N_CLS)), zero_division=0)),
               hier_error=float(hier_distance(y, pred)),
               ece=ece_score(p, y),
               nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
               brier=float(((p - np.eye(N_CLS)[y]) ** 2).sum(1).mean()))
    present = [k for k in range(N_CLS) if 0 < (y == k).sum() < len(y)]
    try:
        out["macro_auroc"] = float(np.mean([roc_auc_score(y == k, p[:, k]) for k in present]))
    except ValueError:
        out["macro_auroc"] = float("nan")
    if 0 < ym.sum() < len(ym):
        out["mal_auroc"] = float(roc_auc_score(ym, pm))
    out["mal_sens"] = float(((pm >= thr) & (ym == 1)).sum() / max(1, ym.sum()))
    out["mal_spec"] = float(((pm < thr) & (ym == 0)).sum() / max(1, (ym == 0).sum()))
    if thr_op is not None:
        out["mal_sens_op"] = float(((pm >= thr_op) & (ym == 1)).sum() / max(1, ym.sum()))
        out["mal_spec_op"] = float(((pm < thr_op) & (ym == 0)).sum() / max(1, (ym == 0).sum()))
    return out

def bootstrap_ci(y, p, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(y[i], p[i]))
        except ValueError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2

# %% Referral analysis
def referral_curves(y, p_c, p_b, scores, qs=np.linspace(0, 1, 21), thr_c=0.5, thr_b=0.5, n_rand=20, y_mal=None):
    """Hybrid policy: referred lesions use the paired prediction and the paired threshold,
    the rest use the clinical-only prediction and threshold (thresholds = target-sensitivity operating points)."""
    ym = IS_MAL[y].astype(int) if y_mal is None else np.asarray(y_mal).astype(int)
    def at(refer):
        P = np.where(refer[:, None], p_b, p_c)
        pm = P[:, MAL_IDX].sum(1)
        thr = np.where(refer, thr_b, thr_c)
        return dict(sens=((pm >= thr) & (ym == 1)).sum() / max(1, ym.sum()),
                    spec=((pm < thr) & (ym == 0)).sum() / max(1, (ym == 0).sum()),
                    macro_f1=f1_score(y, P.argmax(1), average="macro", labels=list(range(N_CLS)), zero_division=0))
    n = len(y); rows = []
    for name, sc in scores.items():
        order = np.argsort(-sc)
        for q in qs:
            refer = np.zeros(n, bool); refer[order[:int(round(q * n))]] = True
            rows.append(dict(policy=name, refer_frac=q, **at(refer)))
    rng = np.random.default_rng(0)
    for q in qs:
        acc = []
        for _ in range(n_rand):
            refer = np.zeros(n, bool); refer[rng.permutation(n)[:int(round(q * n))]] = True
            acc.append(at(refer))
        rows.append(dict(policy="random", refer_frac=q, **{k: float(np.mean([a[k] for a in acc])) for k in acc[0]}))
    return pd.DataFrame(rows)

def dermoscopy_avoided(curves, tol):
    out = {}
    full = curves[curves.refer_frac == 1.0].iloc[0]
    for pol, g in curves.groupby("policy"):
        # both sensitivity AND specificity must stay within `tol` of always sending for dermoscopy
        ok = g[(g.sens >= full.sens - tol) & (g.spec >= full.spec - tol)].sort_values("refer_frac")
        q = float(ok.refer_frac.iloc[0]) if len(ok) else 1.0
        out[pol] = dict(min_refer_frac=q, dermoscopy_avoided=1 - q,
                        auc_sens=float(np.trapezoid(g.sort_values("refer_frac").sens, g.sort_values("refer_frac").refer_frac))
                        if hasattr(np, "trapezoid") else float(np.trapz(g.sort_values("refer_frac").sens, g.sort_values("refer_frac").refer_frac)))
    return out

# %% Cross-validation driver
def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)

@torch.no_grad()
def set_mean_imputation(model, df, tf):
    """Replace the missing-view tokens with the mean training embedding of each view."""
    dl = DataLoader(LesionDS(df, META, tf, False, 0, CFG["use_monet"]), CFG["batch_size"] * 2,
                    shuffle=False, num_workers=CFG["num_workers"])
    model.eval(); sc, sd, n = 0, 0, 0
    for xc, xd, m, mc, md, _ in dl:
        xc, xd, mc, md = [t.to(DEVICE) for t in (xc, xd, mc, md)]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=CFG["amp"] and DEVICE == "cuda"):
            zc, zd = model.encode_c(xc, mc).float(), model.encode_d(xd, md).float()
        sc, sd, n = sc + zc.sum(0), sd + zd.sum(0), n + len(zc)
    model.miss_c.data.copy_(sc / n); model.miss_d.data.copy_(sd / n); model.flag_both = True

def run_experiment(exp_name):
    exp_cfg = EXPERIMENTS[exp_name]
    eval_states = exp_cfg[1]
    exp_dir = os.path.join(CFG["out_dir"], exp_name); os.makedirs(exp_dir, exist_ok=True)
    skf = StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    folds = list(skf.split(LES, LES["label"]))
    oof = {s: np.full((len(LES), N_CLS), np.nan, np.float32) for s in eval_states}
    oof_T = {s: np.full((len(LES), N_CLS), np.nan, np.float32) for s in eval_states}   # per-state temperature
    oof_G = {s: np.full((len(LES), N_CLS), np.nan, np.float32) for s in eval_states}   # one global temperature
    oof_thr = {s: np.full(len(LES), np.nan, np.float32) for s in eval_states}         # operating-point thresholds
    oof_ref = np.full(len(LES), np.nan, np.float32)
    fold_id = np.full(len(LES), -1)
    temps = {}
    for f in CFG["folds_to_run"]:
        tr_idx, te_idx = folds[f]
        tr_df = LES.iloc[tr_idx]
        strat = tr_df["label"] if tr_df["label"].value_counts().min() >= 2 else None
        tr_in, va_in = train_test_split(tr_df, test_size=CFG["inner_val_frac"], stratify=strat, random_state=CFG["seed"])
        seed_all(CFG["seed"] + SEED_OFFSET.get(exp_name, 0) + f)
        model, tf_te, log = train_fold(exp_name, tr_in, va_in, f)
        if exp_name in MEANIMP:
            set_mean_imputation(model, tr_in, tf_te)
        pd.DataFrame(log).to_csv(os.path.join(exp_dir, "trainlog_fold%d.csv" % f), index=False)
        dl = lambda d: DataLoader(LesionDS(d, META, tf_te, False, 0, CFG["use_monet"]), CFG["batch_size"] * 2,
                                  shuffle=False, num_workers=CFG["num_workers"])
        pnet = parallel(model)
        pv = predict(pnet, dl(va_in), eval_states, CFG["tta"])
        pt = predict(pnet, dl(LES.iloc[te_idx]), eval_states, CFG["tta"])
        yv = va_in["label"].values
        Ts = {s: fit_temperature(pv[s], yv) for s in eval_states}
        Tg = fit_temperature(np.concatenate([pv[s] for s in eval_states]), np.tile(yv, len(eval_states)))
        # operating point: threshold reaching target sensitivity on the inner validation set, per state
        thr = {s: thr_for_sens(softmax_np(pv[s] / Ts[s])[:, MAL_IDX].sum(1), IS_MAL[yv].astype(int),
                               CFG["target_sens"]) for s in eval_states}
        temps[f] = dict(per_state=Ts, global_T=Tg, thr=thr)
        for s in eval_states:
            oof[s][te_idx] = softmax_np(pt[s])
            oof_T[s][te_idx] = softmax_np(pt[s] / Ts[s])
            oof_G[s][te_idx] = softmax_np(pt[s] / Tg)
            oof_thr[s][te_idx] = thr[s]
        if "ref" in pt and exp_cfg[3]:
            oof_ref[te_idx] = 1 / (1 + np.exp(-pt["ref"]))
        fold_id[te_idx] = f
        if CFG["save_models"]:
            torch.save({"state_dict": model.state_dict(), "temps": temps[f], "meta_sites": META.sites,
                        "cfg": CFG, "experiment": exp_name, "flag_both": getattr(model, "flag_both", False)},
                       os.path.join(exp_dir, "model_fold%d.pt" % f))
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(exp_dir, "oof.npz"), fold=fold_id, y=LES["label"].values, ref=oof_ref,
                        lesion_id=np.asarray(LES["lesion_id"].astype(str).tolist(), dtype="U32"), tone=LES["tone"].values.astype(float),
                        **{"raw_" + s: oof[s] for s in eval_states},
                        **{"T_" + s: oof_T[s] for s in eval_states},
                        **{"G_" + s: oof_G[s] for s in eval_states},
                        **{"thr_" + s: oof_thr[s] for s in eval_states})
    json.dump(temps, open(os.path.join(exp_dir, "temperatures.json"), "w"), indent=1)
    return evaluate_experiment(exp_name)

# %% Evaluation of one experiment (re-runnable from saved oof.npz)
def tone_group(t):
    if pd.isna(t): return "unknown"
    t = int(t)
    return "dark (0-2)" if t <= 2 else ("medium (3)" if t == 3 else "light (4-5)")

def evaluate_experiment(exp_name):
    exp_dir = os.path.join(CFG["out_dir"], exp_name)
    z = np.load(os.path.join(exp_dir, "oof.npz"))
    done = z["fold"] >= 0
    y = z["y"][done]
    states = [k[2:] for k in z.files if k.startswith("T_")]
    res = dict(experiment=exp_name, n_lesions=int(done.sum()), states={})
    rows = []
    THR = {s: (z["thr_" + s][done] if "thr_" + s in z.files else None) for s in states}
    for s in states:
        res["states"][s] = {}
        for cal in ("raw", "G", "T"):
            p = z["%s_%s" % (cal, s)][done]
            mb = metrics_block(y, p, thr_op=THR[s] if cal == "T" else None)
            res["states"][s][cal] = mb
            rows.append(dict(state=s, calibration={"raw": "none", "G": "global T", "T": "per-state T"}[cal], **mb))
        p = z["T_" + s][done]
        res["states"][s]["CI95"] = dict(
            macro_f1=bootstrap_ci(y, p, lambda a, b: f1_score(a, b.argmax(1), average="macro",
                                  labels=list(range(N_CLS)), zero_division=0), CFG["n_boot"]),
            mal_auroc=bootstrap_ci(y, p, lambda a, b: roc_auc_score(IS_MAL[a], b[:, MAL_IDX].sum(1)), CFG["n_boot"]))
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
    print(tab.round(4).to_string(index=False))

    # skin tone subgroups (per-state temperature)
    tg = LES["tone"].map(tone_group).values[done]
    trows = []
    for s in states:
        p = z["T_" + s][done]
        for g in sorted(set(tg)):
            mk = tg == g
            if mk.sum() < 20: continue
            mb = metrics_block(y[mk], p[mk], thr_op=THR[s][mk] if THR[s] is not None else None)
            trows.append(dict(state=s, tone_group=g, n=int(mk.sum()), accuracy=mb["accuracy"],
                              macro_f1=mb["macro_f1"], mal_auroc=mb.get("mal_auroc", np.nan),
                              mal_sens=mb["mal_sens"], mal_spec=mb["mal_spec"],
                              mal_sens_op=mb.get("mal_sens_op", np.nan), mal_spec_op=mb.get("mal_spec_op", np.nan),
                              ece=mb["ece"]))
    pd.DataFrame(trows).to_csv(os.path.join(exp_dir, "skin_tone.csv"), index=False)

    # referral analysis (needs both and clinical-only states)
    if "both" in states and "c" in states:
        p_c, p_b = z["T_c"][done], z["T_both"][done]
        pm_c = p_c[:, MAL_IDX].sum(1)
        scores = {"clin_entropy": entropy(p_c), "clin_mal_margin": -np.abs(pm_c - 0.5)}
        if not np.all(np.isnan(z["ref"][done])):
            scores["referral_head"] = z["ref"][done]
        tc = THR["c"] if THR["c"] is not None else 0.5
        tb = THR["both"] if THR["both"] is not None else 0.5
        cur = referral_curves(y, p_c, p_b, scores, thr_c=tc, thr_b=tb)
        cur.to_csv(os.path.join(exp_dir, "referral_curves.csv"), index=False)
        res["referral"] = dermoscopy_avoided(cur, CFG["referral_sens_tolerance"])
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for pol, g in cur.groupby("policy"):
            ax[0].plot(g.refer_frac, g.spec, label=pol); ax[1].plot(g.refer_frac, g.sens, label=pol)
            ax[2].plot(g.refer_frac, g.macro_f1, label=pol)
        for a, lab in zip(ax, ["malignant specificity", "malignant sensitivity", "macro-F1 (11 classes)"]):
            a.set_xlabel("fraction sent for dermoscopy"); a.set_ylabel(lab)
        ax[0].set_title("operating point: %.0f%% target sensitivity" % (100 * CFG["target_sens"]), fontsize=9)
        ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(exp_dir, "referral_curves.png"), dpi=150)
        plt.close(fig)
        print("Operating point: thresholds for %.0f%% sensitivity (set on inner validation)." % (100 * CFG["target_sens"]))
        print("Always-dermoscopy vs. clinical-only at that point:",
              cur[(cur.policy == "random") & (cur.refer_frac.isin([0.0, 1.0]))][["refer_frac", "sens", "spec", "macro_f1"]]
              .round(3).to_string(index=False))
        print("Dermoscopy avoided while sensitivity AND specificity stay within %.2f of always-dermoscopy:"
              % CFG["referral_sens_tolerance"])
        print(pd.DataFrame(res["referral"]).T.round(3))
    json.dump(res, open(os.path.join(exp_dir, "summary.json"), "w"), indent=1, default=float)
    return res

# %% Run all experiments
ALL = {}
for name in CFG["experiments"]:
    print("=" * 30, name, "=" * 30)
    t0 = time.time()
    ALL[name] = run_experiment(name)
    print("%s done in %.1f min" % (name, (time.time() - t0) / 60))

# %% Comparison table across experiments (per-state temperature)
def comparison_table(out_dir, names):
    rows = []
    for n in names:
        p = os.path.join(out_dir, n, "summary.json")
        if not os.path.exists(p): continue
        r = json.load(open(p))
        for s, d in r["states"].items():
            m = d["T"]
            rows.append(dict(experiment=n, state=s, macro_f1=m["macro_f1"], ci_f1=d["CI95"]["macro_f1"],
                             bal_acc=m["balanced_acc"], macro_auroc=m["macro_auroc"],
                             mal_auroc=m.get("mal_auroc"), mal_sens=m["mal_sens"], mal_spec=m["mal_spec"],
                             mal_sens_at_op=m.get("mal_sens_op"), mal_spec_at_op=m.get("mal_spec_op"),
                             hier_error=m["hier_error"], ece_raw=d["raw"]["ece"], ece_globalT=d["G"]["ece"],
                             ece_stateT=m["ece"]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    return t

COMP = comparison_table(CFG["out_dir"], CFG["experiments"])
print(COMP.round(4).to_string(index=False))

# %% [markdown]
# ## Optional: MILK10k benchmark submission (ensemble of the 'full' fold models, paired state)
# Upload `submission.csv` on the MILK10k challenge site to get the official blind-test score.

# %% Benchmark submission
def load_fold_models(exp_name):
    models = []
    for p in sorted(glob.glob(os.path.join(CFG["out_dir"], exp_name, "model_fold*.pt"))):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"]); cfg["pretrained"] = False; cfg["panderm_ckpt"] = None
        m = RefNet(cfg, META.dim, len(MONET_COLS) if CFG["use_monet"] else 0).to(DEVICE)
        m.load_state_dict(ck["state_dict"]); m.eval(); m.flag_both = ck.get("flag_both", False)
        models.append((m, ck["temps"]))
    return models

def predict_ensemble(models, df, state, has_c=True, has_d=True):
    mean, std = enc_norm(models[0][0].enc_c)
    _, tf_te = make_transforms(CFG["img_size"], mean, std)
    dl = DataLoader(LesionDS(df, META, tf_te, False, 0, CFG["use_monet"], has_c, has_d),
                    CFG["batch_size"] * 2, shuffle=False, num_workers=CFG["num_workers"])
    ps, refs = [], []
    for m, temps in models:
        out = predict(parallel(m), dl, (state,) if state != "both" else ("both", "c"), CFG["tta"])
        ps.append(softmax_np(out[state] / temps["per_state"][state]))
        if "ref" in out: refs.append(1 / (1 + np.exp(-out["ref"])))
    return np.mean(ps, 0), (np.mean(refs, 0) if refs else None)

def ensemble_threshold(models, state):
    """Mean of the per-fold operating-point thresholds (target sensitivity), 0.5 if not stored."""
    t = [tm.get("thr", {}).get(state) for _, tm in models]
    t = [x for x in t if x is not None]
    return float(np.mean(t)) if t else 0.5

MAKE_SUBMISSION = True
if MAKE_SUBMISSION and FILES["test_meta"] and CFG["save_models"] and CFG.get("submission_experiments"):
    TEST_LES, _ = build_lesion_table([FILES["test_meta"]], None, IMG_INDEX)
    tm = build_cache(list(TEST_LES["path_c"]) + list(TEST_LES["path_d"]), CFG["cache_dir"], CFG["cache_short_side"])
    TEST_LES["cpath_c"] = TEST_LES["path_c"].map(tm); TEST_LES["cpath_d"] = TEST_LES["path_d"].map(tm)
    for e in CFG["submission_experiments"]:
        p_test, _ = predict_ensemble(load_fold_models(e), TEST_LES, "both")
        sub = pd.DataFrame(p_test, columns=CLASSES); sub.insert(0, "lesion_id", TEST_LES["lesion_id"].values)   # scorer requires "lesion_id"
        sub.to_csv(os.path.join(CFG["out_dir"], "submission_%s.csv" % e), index=False)
        print("submission_%s.csv written:" % e, sub.shape)
else:
    print("Submission skipped (no test metadata, no saved models, or no submission experiments).")

# %% [markdown]
# ## Optional: external validation
# Add each dataset with **Add Input**, set its folder below, and run. Labels are mapped to the overlapping
# MILK10k classes; the binary malignant task is reported for all. Metadata is passed as "unknown" unless mapped.
# * Derm7pt (paired C+D): evaluated in all three states + referral.
# * PAD-UFES-20 (smartphone clinical): clinical-only state.
# * DDI (clinical, Fitzpatrick 12/34/56): clinical-only state, by skin-tone group.
# * MIDAS (paired, prospective): template — set the column names after looking at its metadata file.

# %% External datasets: loaders
def _auto_dir(pattern, check=None):
    for p in glob.glob(os.path.join(CFG["data_root"], "**", pattern), recursive=True):
        if check is None or check(p):
            return os.path.dirname(p)
    return None

_d7 = _auto_dir("meta.csv", lambda p: os.path.basename(os.path.dirname(p)) == "meta" and "clinic" in _read_header(p))
_pad = _auto_dir("metadata.csv", lambda p: "diagnostic" in _read_header(p))
EXTERNAL = dict(
    derm7pt_dir=os.path.dirname(_d7) if _d7 else None,   # auto-detected (folder holding images/ and meta/)
    pad_dir=_pad,                                         # auto-detected PAD-UFES-20 folder
    ddi_dir=None,        # e.g. "/kaggle/input/ddi-diverse-dermatology-images"
    midas_dir=None,      # e.g. "/kaggle/input/mra-midas"
    midas_cols=dict(file="midas_file_name", lesion="midas_record_id", view="midas_distance",
                    derm_value="dscope", label="midas_path", malignant_values=["malignant"]),
)
if os.environ.get("SKINREF_TEST_EXT"):
    EXTERNAL.update(json.loads(os.environ["SKINREF_TEST_EXT"]))

def _find(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return hits[0] if hits else None

def load_derm7pt(root):
    meta_p = _find(root, "meta.csv")
    img_dir = os.path.join(os.path.dirname(os.path.dirname(meta_p)), "images")
    meta = pd.read_csv(meta_p)
    def resolve(rel):
        p = os.path.join(img_dir, str(rel))
        return p if os.path.exists(p) else None
    def lab(d):
        d = str(d).lower()
        if "melanoma" in d: return "MEL"
        if "basal cell" in d: return "BCC"
        if "seborrheic" in d: return "BKL"
        if "nevus" in d: return "NV"
        if "dermatofibroma" in d: return "DF"
        if "vascular" in d: return "VASC"
        return None
    rows = []
    for r in meta.itertuples():
        pc, pd_ = resolve(r.clinic), resolve(r.derm)
        L = lab(r.diagnosis)
        mal = 1 if L in ("MEL", "BCC") else 0
        sex = getattr(r, "sex", np.nan)
        rows.append(dict(lesion_id="d7_%s" % r.case_num, path_c=pc, path_d=pd_, label=C2I[L] if L else -1,
                         malignant=mal, age=np.nan, sex=sex, site=np.nan, tone=np.nan, diagnosis=str(r.diagnosis)))
    return pd.DataFrame(rows).dropna(subset=["path_c", "path_d"])

def load_pad(root):
    meta = pd.read_csv(_find(root, "metadata.csv"))
    idx = image_index(root)
    mp = {"ACK": "AKIEC", "BCC": "BCC", "MEL": "MEL", "NEV": "NV", "SCC": "SCCKA", "SEK": "BKL"}
    rows = []
    for r in meta.itertuples():
        L = mp.get(str(r.diagnostic))
        rows.append(dict(lesion_id="pad_%s" % r.img_id, path_c=idx.get(os.path.splitext(r.img_id)[0]), path_d=None,
                         label=C2I[L] if L else -1, malignant=int(L in MALIGNANT) if L else 0,
                         age=getattr(r, "age", np.nan), sex=str(getattr(r, "gender", "")).lower() or np.nan,
                         site=np.nan, tone=np.nan, fitz=getattr(r, "fitspatrick", np.nan),
                         patient_id=getattr(r, "patient_id", np.nan), diagnosis=str(r.diagnostic)))
    return pd.DataFrame(rows).dropna(subset=["path_c"])

def load_ddi(root):
    meta = pd.read_csv(_find(root, "ddi_metadata.csv"))
    idx = image_index(root)
    rows = []
    for r in meta.itertuples():
        rows.append(dict(lesion_id="ddi_%s" % r.DDI_file, path_c=idx.get(os.path.splitext(r.DDI_file)[0]), path_d=None,
                         label=-1, malignant=int(str(r.malignant).lower() in ("true", "1")),
                         age=np.nan, sex=np.nan, site=np.nan, tone=np.nan, fitz=r.skin_tone))
    return pd.DataFrame(rows).dropna(subset=["path_c"])

def load_midas(root, c):
    p = _find(root, "*.xlsx") or _find(root, "*.csv")
    meta = pd.read_excel(p) if p.endswith(".xlsx") else pd.read_csv(p)
    print("MIDAS columns:", list(meta.columns))
    idx = image_index(root)
    rows = []
    for lid, g in meta.groupby(c["lesion"]):
        d = g[g[c["view"]].astype(str) == c["derm_value"]]
        cl = g[g[c["view"]].astype(str) != c["derm_value"]]
        if len(d) == 0 or len(cl) == 0: continue
        lab = str(d.iloc[0][c["label"]]).lower()
        rows.append(dict(lesion_id="midas_%s" % lid,
                         path_c=idx.get(os.path.splitext(str(cl.iloc[0][c["file"]]))[0]),
                         path_d=idx.get(os.path.splitext(str(d.iloc[0][c["file"]]))[0]),
                         label=-1, malignant=int(any(v in lab for v in c["malignant_values"])),
                         age=np.nan, sex=np.nan, site=np.nan, tone=np.nan))
    return pd.DataFrame(rows).dropna(subset=["path_c", "path_d"])

# %% External datasets: evaluation
def external_eval(name, df, states, exp_name="full", group_col=None):
    if len(df) == 0:
        print(name, ": no usable rows"); return None
    paths = [p for p in list(df["path_c"]) + list(df.get("path_d", pd.Series(dtype=str))) if isinstance(p, str)]
    cm = build_cache(paths, CFG["cache_dir"], CFG["cache_short_side"])
    df = df.copy(); df["cpath_c"] = df["path_c"].map(cm)
    df["cpath_d"] = df["path_d"].map(cm) if "path_d" in df else None
    df["cpath_d"] = df["cpath_d"].fillna(df["cpath_c"])           # placeholder, never used when has_d=False
    models = load_fold_models(exp_name)
    res, preds = {}, {}
    for s in states:
        p, ref = predict_ensemble(models, df, s, has_c=s in ("both", "c"), has_d=s in ("both", "d"))
        preds[s] = p
        ym = df["malignant"].values; pm = p[:, MAL_IDX].sum(1)
        t_op = ensemble_threshold(models, s)      # fixed on MILK10k, never tuned on the external set
        r = dict(n=len(df), mal_auroc=float(roc_auc_score(ym, pm)) if 0 < ym.sum() < len(ym) else np.nan,
                 mal_sens=float(((pm >= .5) & (ym == 1)).sum() / max(1, ym.sum())),
                 mal_spec=float(((pm < .5) & (ym == 0)).sum() / max(1, (ym == 0).sum())),
                 thr_op=t_op,
                 mal_sens_at_op=float(((pm >= t_op) & (ym == 1)).sum() / max(1, ym.sum())),
                 mal_spec_at_op=float(((pm < t_op) & (ym == 0)).sum() / max(1, (ym == 0).sum())))
        lab = df["label"].values; mk = lab >= 0
        if mk.sum() > 0:
            keep = sorted(set(lab[mk]))
            sub = p[mk][:, keep]
            pred = np.array(keep)[sub.argmax(1)]
            r["macro_f1_overlap"] = float(f1_score(lab[mk], pred, average="macro", labels=keep, zero_division=0))
        if group_col:
            for g, gi in df.groupby(group_col).groups.items():
                gi = df.index.get_indexer(gi)
                if 0 < ym[gi].sum() < len(gi):
                    r["mal_auroc_%s_%s" % (group_col, g)] = float(roc_auc_score(ym[gi], pm[gi]))
        if s == "c" and ref is not None:
            preds["ref"] = ref
        res[s] = r
    # per-lesion predictions for offline analysis (site recalibration, subgroups, patient-level)
    keep = [c for c in ("lesion_id", "patient_id", "diagnosis", "malignant", "label", "fitz", "age", "sex", "path_c")
            if c in df.columns]
    out_df = df[keep].copy()
    for s_, p_ in preds.items():
        if s_ == "ref":
            out_df["ref_score"] = p_
        else:
            out_df["pmal_" + s_] = p_[:, MAL_IDX].sum(1)
            for k, cname in enumerate(CLASSES):
                out_df["p_%s_%s" % (s_, cname)] = p_[:, k]
            out_df["thr_op_" + s_] = ensemble_threshold(models, s_)
    out_df.to_csv(os.path.join(CFG["out_dir"], "external_%s_predictions.csv" % name), index=False)
    if "both" in preds and "c" in preds:
        y_bin = df["malignant"].values
        ym_lab = np.where(y_bin == 1, C2I["MEL"], C2I["NV"])   # binary proxy labels for the curve
        sc = {"clin_entropy": entropy(preds["c"]), "clin_mal_margin": -np.abs(preds["c"][:, MAL_IDX].sum(1) - .5)}
        if "ref" in preds: sc["referral_head"] = preds["ref"]
        cur = referral_curves(ym_lab, preds["c"], preds["both"], sc, y_mal=y_bin,
                              thr_c=ensemble_threshold(models, "c"), thr_b=ensemble_threshold(models, "both"))
        res["referral"] = dermoscopy_avoided(cur, CFG["referral_sens_tolerance"])
        cur.to_csv(os.path.join(CFG["out_dir"], "external_%s_referral.csv" % name), index=False)
    json.dump(res, open(os.path.join(CFG["out_dir"], "external_%s.json" % name), "w"), indent=1, default=float)
    print(name, json.dumps(res, indent=1, default=float))
    return res

print("External datasets found:", {k: v for k, v in EXTERNAL.items() if k.endswith("_dir")})
for e in CFG.get("external_experiments", []):
    if EXTERNAL["derm7pt_dir"]:
        external_eval("derm7pt_%s" % e, load_derm7pt(EXTERNAL["derm7pt_dir"]), ("both", "c", "d"), exp_name=e)
    if EXTERNAL["pad_dir"]:
        external_eval("pad_ufes20_%s" % e, load_pad(EXTERNAL["pad_dir"]), ("c",), exp_name=e, group_col="fitz")
if EXTERNAL["ddi_dir"]:
    external_eval("ddi", load_ddi(EXTERNAL["ddi_dir"]), ("c",), group_col="fitz")
if EXTERNAL["midas_dir"]:
    external_eval("midas", load_midas(EXTERNAL["midas_dir"], EXTERNAL["midas_cols"]), ("both", "c", "d"))

# %% Pack results for download
import zipfile
ZIP_PATH = os.path.join(os.path.dirname(CFG["out_dir"].rstrip("/")), "results_r3%s.zip" % RUN)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in glob.glob(os.path.join(CFG["out_dir"], "**", "*"), recursive=True):
        if os.path.isfile(p) and not p.endswith(".pt"):          # model weights stay out of the zip
            zf.write(p, os.path.relpath(p, CFG["out_dir"]))
print("Done. Download", ZIP_PATH, "from the Output panel and send it to me.")
