"""BreakHis indexing, patient-level splits, stain augmentation and the PyTorch dataset."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

SUBTYPES = {
    "A": ("benign", "Adenosis"),
    "F": ("benign", "Fibroadenoma"),
    "PT": ("benign", "Phyllodes tumor"),
    "TA": ("benign", "Tubular adenoma"),
    "DC": ("malignant", "Ductal carcinoma"),
    "LC": ("malignant", "Lobular carcinoma"),
    "MC": ("malignant", "Mucinous carcinoma"),
    "PC": ("malignant", "Papillary carcinoma"),
}

# SOB_B_TA-14-4659-40-001 -> procedure_class_subtype-year-slide-magnification-sequence
NAME_RE = re.compile(
    r"^(?P<procedure>[A-Z]+)_(?P<cls>[BM])_(?P<subtype>[A-Z]{1,2})"
    r"-(?P<year>\d{2})-(?P<slide>[A-Za-z0-9]+)-(?P<mag>\d+)-(?P<seq>\d+)$"
)


def build_index(root):
    """Walk the dataset folder and return one row per image.

    label: 0 = benign, 1 = malignant. patient: the slide ID encoded in the file name.
    Duplicate files (some Kaggle mirrors ship the images twice) are dropped by file name.
    """
    rows = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        m = NAME_RE.match(p.stem)
        if not m:
            continue
        g = m.groupdict()
        rows.append({
            "path": str(p),
            "filename": p.name,
            "label": 0 if g["cls"] == "B" else 1,
            "subtype": g["subtype"],
            "patient": f'{g["year"]}-{g["slide"]}',
            "magnification": int(g["mag"]),
        })
    if not rows:
        raise FileNotFoundError(f"No BreakHis images found under {root}")
    df = pd.DataFrame(rows).drop_duplicates("filename").reset_index(drop=True)
    return df


def find_official_folds(root):
    """Return the official Folds.csv as a DataFrame if the download includes it, else None."""
    hits = list(Path(root).rglob("Folds.csv"))
    return pd.read_csv(hits[0]) if hits else None


def _patient_labels(df):
    return df.groupby("patient")["label"].agg(lambda s: int(round(s.mean())))


def split_by_patient(df, val_frac=0.15, test_frac=0.15, seed=0):
    """Stratified split at the patient level: no patient appears in two sets."""
    pl = _patient_labels(df)
    patients, labels = pl.index.to_numpy(), pl.to_numpy()
    trval, test = train_test_split(patients, test_size=test_frac, stratify=labels, random_state=seed)
    trval_labels = pl.loc[trval].to_numpy()
    train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac),
                                  stratify=trval_labels, random_state=seed)
    split = pd.Series("train", index=df.index)
    split[df["patient"].isin(val)] = "val"
    split[df["patient"].isin(test)] = "test"
    return split


def split_by_image(df, val_frac=0.15, test_frac=0.15, seed=0):
    """Random image-level split, stratified by label. Used only for RQ3."""
    idx = df.index.to_numpy()
    trval, test = train_test_split(idx, test_size=test_frac, stratify=df["label"], random_state=seed)
    train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac),
                                  stratify=df.loc[trval, "label"], random_state=seed)
    split = pd.Series("train", index=df.index)
    split[val] = "val"
    split[test] = "test"
    return split


def split_official(df, folds, fold, val_frac=0.15, seed=0):
    """Use the official patient-level train/test fold, then carve a validation set out of
    the training patients so the test set is never used for model selection."""
    f = folds[folds["fold"] == fold]
    grp = dict(zip(f["filename"].map(lambda s: Path(s).name), f["grp"].str.lower()))
    split = df["filename"].map(grp)
    keep = split.notna()
    split = split.where(keep, "unused")
    train_patients = df.loc[split == "train", "patient"].unique()
    pl = _patient_labels(df[df["patient"].isin(train_patients)])
    _, val = train_test_split(pl.index.to_numpy(), test_size=val_frac, stratify=pl.to_numpy(), random_state=seed)
    split[(split == "train") & df["patient"].isin(val)] = "val"
    return split


def make_split(df, mode="patient", seed=0, fold=None, folds=None):
    if mode == "image":
        return split_by_image(df, seed=seed)
    if fold is not None:
        if folds is None:
            raise ValueError("--fold was given but no Folds.csv was found in the dataset")
        return split_official(df, folds, fold, seed=seed)
    return split_by_patient(df, seed=seed)


class HEDJitter:
    """Stain augmentation: randomly scale and shift the hematoxylin, eosin and DAB channels.

    Follows the HED color augmentation idea of Tellez et al. (2018). sigma controls strength.
    """

    def __init__(self, sigma=0.05, p=0.8):
        self.sigma, self.p = sigma, p

    def __call__(self, img):
        from skimage.color import hed2rgb, rgb2hed
        if np.random.rand() > self.p:
            return img
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        hed = rgb2hed(rgb)
        alpha = np.random.uniform(1 - self.sigma, 1 + self.sigma, size=3)
        beta = np.random.uniform(-self.sigma, self.sigma, size=3)
        out = hed2rgb(hed * alpha + beta * hed.std(axis=(0, 1)))
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def build_transforms(img_size=224, train=True, stain_aug=False):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not train:
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    ops = [T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
           T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(90)]
    if stain_aug:
        ops.append(HEDJitter())
    else:
        ops.append(T.ColorJitter(brightness=0.1, contrast=0.1))
    return T.Compose(ops + [T.ToTensor(), norm])


class BreakHisDataset:
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["path"]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(r["label"]), i
