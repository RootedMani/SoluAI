# The chemistry of solubility, for high school students

## 1. What is solubility?

**Solubility** is the most of a substance (the *solute*) that can dissolve in a solvent
(here, water) at a given temperature. Add more than that and the extra stays undissolved:
the solution is **saturated**.

We measure solubility in **moles per litre (mol/L)**. Real molecules cover an enormous
range, from about 10 mol/L down to 0.000000000001 mol/L. So chemists use the logarithm:

> **logS = log₁₀(solubility in mol/L)**

| logS | Solubility | Means | Example (measured, AqSolDB) |
|---|---|---|---|
| +1 | 10 mol/L | mixes almost completely | ethanol ≈ +1.2 |
| 0 | 1 mol/L | very soluble | 1-butanol ≈ 0.0, phenol ≈ 0.0 |
| −1 | 0.1 mol/L | soluble | caffeine ≈ −0.9, benzene ≈ −1.6 |
| −3 | 0.001 mol/L | slightly soluble | 1-octanol ≈ −2.6, hexane ≈ −3.9 |
| −7 | 0.0000001 mol/L | practically insoluble | naphthalene ≈ −4.3, 2,4′-DDT ≈ −7.1 |

**Every step of 1 in logS is a factor of 10.** A model error of 1.0 log unit therefore
means "off by a factor of 10". That sounds large, but the range of logS covers a factor
of 10¹⁵.

To convert to grams per litre: **g/L = 10^logS × molar mass**.

## 2. What happens when something dissolves?

Dissolving a solid molecule in water takes three steps:

1. **Break the solid apart.** The molecule must leave its crystal (this costs energy).
2. **Make a hole in water.** Water molecules must move apart to make a cavity (this costs
   energy, because water's hydrogen bonds are broken).
3. **Let the molecule and water interact.** New attractions form (this releases energy).

A substance dissolves well when step 3 pays back steps 1 and 2. Each of the ideas below
is about one of these steps.

## 3. "Like dissolves like": polarity

In a water molecule, oxygen is much more **electronegative** than hydrogen. It pulls the
shared electrons towards itself, so the O end is slightly negative (δ−) and the H ends are
slightly positive (δ+). Water is **polar**.

* **Polar** solutes (with O, N, –OH, C=O groups) attract the δ+/δ− ends of water, so step 3
  releases a lot of energy and they dissolve well.
* **Non-polar** solutes (hydrocarbons such as hexane, oils, fats) attract water only weakly.
  Water molecules prefer each other, so the hydrocarbon is "pushed out". This is the
  **hydrophobic effect**: oil and water don't mix.

Chemists measure oiliness with **logP**, the log of the ratio [concentration in octanol] /
[concentration in water]. High logP means oily (hydrophobic); low or negative logP means
water-loving (hydrophilic).

## 4. Hydrogen bonds

A **hydrogen bond** is a strong attraction between:

* an H atom bonded to O, N or F (the **donor**), and
* a lone pair on another O or N atom (the **acceptor**).

Water is both an excellent donor and an excellent acceptor. Molecules that also have
donors and acceptors can join water's hydrogen-bond network:

| Group | Donor? | Acceptor? | Effect on solubility |
|---|---|---|---|
| –OH (alcohol) | ✔ | ✔ | strong increase |
| –COOH (acid) | ✔ | ✔ | strong increase |
| –NH₂ (amine) | ✔ | ✔ | increase |
| C=O (ketone) | ✘ | ✔ | moderate increase |
| –O– (ether) | ✘ | ✔ | small increase |
| C–H, C–C | ✘ | ✘ | decrease (hydrophobic) |

Glucose (C₆H₁₂O₆) has five –OH groups, so it is very soluble even though it is fairly large.

## 5. Size

Step 2 (making a hole in water) costs more for bigger molecules. Within a family of similar
molecules, solubility therefore falls as they get bigger. The textbook example is the
alcohols:

| Alcohol | Formula | logS (measured, approx.) |
|---|---|---|
| ethanol | C₂H₅OH | fully miscible |
| 1-butanol | C₄H₉OH | ≈ −0.05 |
| 1-hexanol | C₆H₁₃OH | ≈ −1.4 |
| 1-octanol | C₈H₁₇OH | ≈ −2.6 |

(Values from AqSolDB.) All four have one –OH group, but the non-polar chain grows. Each extra
CH₂ lowers logS by about 0.65, making the alcohol roughly 4–5 times less soluble.

## 6. The crystal: melting point

Step 1 (leaving the crystal) depends on how tightly molecules pack. **Flat, symmetric
molecules** (benzene rings stacked like plates) and molecules that hydrogen-bond strongly
*to each other* form very stable crystals with **high melting points**, and they dissolve
less than you would expect.

Yalkowsky's **General Solubility Equation** captures both main effects in one line:

> logS ≈ 0.5 − 0.01 × (melting point in °C − 25) − logP

The first part is the crystal effect; the second is the polarity effect.

## 7. Other factors

* **Temperature:** most solids dissolve better in hot water; gases dissolve worse.
* **pH:** acids such as aspirin dissolve much better in basic solutions, because they turn
  into charged ions. Bases dissolve better in acidic solutions.
* **Salts** (NaCl, CaCO₃) dissolve by splitting into ions. Their solubility depends on lattice
  energy and on how strongly water surrounds each ion (hydration). These are different
  rules, which is why SoluAI does not try to predict inorganic salts.

## 8. How this connects to SoluAI

The model never saw this page: it only saw ~8,700 molecules and their measured logS.
Did it rediscover the chemistry anyway? On the website's "How the model works" page you can
check that:

* **logP** is the single most important feature, and high logP **lowers** the prediction
  (polarity, section 3);
* **polar surface area** and **H-bond donors** **raise** it (hydrogen bonding, section 4);
* **molecular weight** lowers it (size, section 5);
* **aromatic rings** lower it. This is partly a stand-in for crystal packing (section 6),
  which the model cannot see directly. It is also where the model makes its biggest
  mistakes.
* **One surprise:** more **H-bond acceptors** slightly *lower* the prediction (correlation −0.25),
  although chemistry says acceptors help. The likely reason: the acceptor count grows with
  molecule size, and the polar effect is already counted by polar surface area and donors.
  So the model uses "many acceptors" partly as a sign of "big molecule". This is a good example
  of why a model's reasoning must be checked: features are correlated, and a model can pick up
  a shortcut instead of the real cause.

The atom colours on the Predict page show the same thing for one molecule at a time: O and N
atoms in –OH, –NH₂ and –COOH groups are usually blue (they help), and carbon chains,
aromatic rings and halogens are usually red (they hurt).
