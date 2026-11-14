"""Functional groups a chemist would point at, found with SMARTS patterns.

SMARTS is a pattern language for molecules (like "regular expressions for
chemistry"). Each group has a short chemistry reason used in the
plain-language explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class Group:
    """A functional group and why it matters for solubility."""

    key: str
    label: str          # singular, e.g. "–OH group"
    plural: str         # e.g. "–OH groups"
    smarts: str
    polar: bool         # True = we expect it to help solubility
    why_up: str         # explanation when it increases solubility
    why_down: str       # explanation when it decreases solubility


GROUPS: list[Group] = [
    Group("acid", "–COOH (carboxylic acid) group", "–COOH groups", "[CX3](=O)[OX2H1]", True,
          "it forms strong hydrogen bonds with water", "it is outweighed by the rest of the molecule here"),
    Group("hydroxyl", "–OH group", "–OH groups", "[OX2H1][#6;!$(C=O)]", True,
          "it forms hydrogen bonds with water", "it also helps molecules stick to each other in the solid"),
    Group("amine", "–NH₂/–NH– (amine) group", "amine groups", "[NX3;H2,H1;!$(NC=O);!$(N-a);!$(N-S(=O)=O)]", True,
          "it forms hydrogen bonds with water", "its effect is small in this molecule"),
    Group("aniline", "aromatic –NH₂ group", "aromatic –NH₂ groups", "[NX3;H2,H1;$(N-a);!$(NC=O)]", True,
          "it forms hydrogen bonds with water", "its effect is small in this molecule"),
    Group("amide", "amide (–CO–N–) group", "amide groups", "[NX3][CX3](=[OX1])", True,
          "it is very polar and can both give and accept hydrogen bonds", "amides also hold crystals together strongly"),
    Group("carbonyl", "C=O (carbonyl) group", "C=O groups", "[CX3;!$(C-[OX2H1]);!$(C-[NX3]);!$(C-O-[#6])]=[OX1]", True,
          "its oxygen accepts hydrogen bonds from water", "its effect is small in this molecule"),
    Group("ester", "ester group", "ester groups", "[#6][CX3](=O)[OX2][#6]", True,
          "its oxygens accept hydrogen bonds from water", "it adds size without adding hydrogen-bond donors"),
    Group("ether", "ether oxygen (–O–)", "ether oxygens", "[OD2;!$(OC=O)]([#6])[#6]", True,
          "its oxygen accepts hydrogen bonds from water", "its effect is small in this molecule"),
    Group("nitro", "nitro (–NO₂) group", "nitro groups", "[N+](=O)[O-]", True,
          "it is polar", "it makes the molecule flat and tightly packed in crystals"),
    Group("nitrile", "nitrile (–C≡N) group", "nitrile groups", "C#N", True,
          "it is polar", "its effect is small in this molecule"),
    Group("sulfonyl", "sulfonyl/sulfonamide group", "sulfonyl groups", "S(=O)(=O)", True,
          "it is very polar", "it also strengthens the crystal"),
    Group("aromatic_n", "ring nitrogen", "ring nitrogens", "[n;H0]", True,
          "its lone pair accepts hydrogen bonds from water", "its effect is small in this molecule"),
    Group("halogen", "halogen atom (Cl, Br, I)", "halogen atoms (Cl, Br, I)", "[Cl,Br,I]", False,
          "it adds some polarity", "it makes the molecule bigger and more oily (hydrophobic)"),
    Group("fluorine", "fluorine atom", "fluorine atoms", "[F]", False,
          "it adds some polarity", "it makes the molecule more hydrophobic"),
    Group("benzene", "benzene ring", "benzene rings", "c1ccccc1", False,
          "its effect is small in this molecule", "it is non-polar and flat, so it avoids water and packs tightly in crystals"),
    Group("chain", "long carbon chain", "long carbon chains", "[CH2][CH2][CH2][CH2,CH3]", False,
          "its effect is small in this molecule", "it is non-polar - water cannot hydrogen-bond to it"),
]

#: Persian text for each group: (name, why it raises solubility, why it lowers it).
GROUPS_FA: dict[str, tuple[str, str, str]] = {
    "acid": ("گروه ‎–COOH‎ (کربوکسیلیک اسید)", "با آب پیوند هیدروژنی قوی تشکیل می‌دهد", "بقیهٔ مولکول اثر آن را خنثی کرده است"),
    "hydroxyl": ("گروه ‎–OH‎", "با آب پیوند هیدروژنی تشکیل می‌دهد", "به چسبیدن مولکول‌ها به یکدیگر در جامد هم کمک می‌کند"),
    "amine": ("گروه آمین (‎–NH₂‎/‎–NH–‎)", "با آب پیوند هیدروژنی تشکیل می‌دهد", "اثر آن در این مولکول کوچک است"),
    "aniline": ("گروه ‎–NH₂‎ آروماتیک", "با آب پیوند هیدروژنی تشکیل می‌دهد", "اثر آن در این مولکول کوچک است"),
    "amide": ("گروه آمید (‎–CO–N–‎)", "بسیار قطبی است و هم پیوند هیدروژنی می‌دهد و هم می‌پذیرد", "آمیدها بلور را هم محکم نگه می‌دارند"),
    "carbonyl": ("گروه کربونیل (‎C=O‎)", "اکسیژن آن از آب پیوند هیدروژنی می‌پذیرد", "اثر آن در این مولکول کوچک است"),
    "ester": ("گروه استری", "اکسیژن‌های آن از آب پیوند هیدروژنی می‌پذیرند", "مولکول را بزرگ‌تر می‌کند بدون آنکه دهندهٔ پیوند هیدروژنی اضافه کند"),
    "ether": ("اکسیژن اتری (‎–O–‎)", "اکسیژن آن از آب پیوند هیدروژنی می‌پذیرد", "اثر آن در این مولکول کوچک است"),
    "nitro": ("گروه نیترو (‎–NO₂‎)", "قطبی است", "مولکول را تخت می‌کند و در بلور فشرده‌تر جا می‌گیرد"),
    "nitrile": ("گروه نیتریل (‎–C≡N‎)", "قطبی است", "اثر آن در این مولکول کوچک است"),
    "sulfonyl": ("گروه سولفونیل/سولفونامید", "بسیار قطبی است", "بلور را هم محکم‌تر می‌کند"),
    "aromatic_n": ("نیتروژن حلقه", "جفت‌الکترون ناپیوندی آن از آب پیوند هیدروژنی می‌پذیرد", "اثر آن در این مولکول کوچک است"),
    "halogen": ("اتم هالوژن (‎Cl، Br، I‎)", "کمی قطبیت اضافه می‌کند", "مولکول را بزرگ‌تر و چرب‌تر (آب‌گریزتر) می‌کند"),
    "fluorine": ("اتم فلوئور", "کمی قطبیت اضافه می‌کند", "مولکول را آب‌گریزتر می‌کند"),
    "benzene": ("حلقهٔ بنزنی", "اثر آن در این مولکول کوچک است", "ناقطبی و تخت است؛ از آب دوری می‌کند و در بلور فشرده جا می‌گیرد"),
    "chain": ("زنجیر کربنی بلند", "اثر آن در این مولکول کوچک است", "ناقطبی است و آب نمی‌تواند با آن پیوند هیدروژنی بدهد"),
}

_PATTERNS = {g.key: Chem.MolFromSmarts(g.smarts) for g in GROUPS}


def find_groups(mol: Chem.Mol) -> list[tuple[Group, list[set[int]]]]:
    """All groups present in ``mol`` with the atom indices of every match.

    Overlapping matches of the same group are merged (e.g. a long chain is
    counted once, a benzene ring is counted once per ring).
    """
    found = []
    for g in GROUPS:
        matches = [set(m) for m in mol.GetSubstructMatches(_PATTERNS[g.key], uniquify=True)]
        if not matches:
            continue
        if g.key in ("chain", "benzene"):
            matches = _merge_overlapping(matches)
        if g.key == "chain" and max(len(m) for m in matches) < 4:
            continue
        found.append((g, matches))
    # An -OH that belongs to a -COOH is not also reported as a plain -OH.
    acid_atoms = set().union(*[set().union(*m) for g, m in found if g.key == "acid"]) if any(
        g.key == "acid" for g, _ in found) else set()
    return [(g, [m for m in ms if not (g.key in ("hydroxyl", "carbonyl") and m & acid_atoms)])
            for g, ms in found if not (g.key in ("hydroxyl", "carbonyl") and all(m & acid_atoms for m in ms))]


def _merge_overlapping(sets: list[set[int]]) -> list[set[int]]:
    merged: list[set[int]] = []
    for s in sets:
        for m in merged:
            if m & s:
                m |= s
                break
        else:
            merged.append(set(s))
    return merged
