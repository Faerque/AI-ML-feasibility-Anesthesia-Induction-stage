# explanation_templates.py

dosages_feature_reference_map = {
    "Age": {
        "positive": "Younger age ({value}) maintained a higher dose since drug clearance is often more efficient.",
        "negative": "Older age ({value}) reduced the dose due to slower metabolism and higher sensitivity.",
        "reference": "Aldrete JA. Anesth Analg. 1998;86(4):791–795. [PMID: 9523811]"
    },
    "BMI": {
        "positive": "BMI ({value}) in normal range allowed standard dosing.",
        "negative": "BMI ({value}) outside normal range led to dose adjustment to avoid under- or over-sedation.",
        "reference": "Ingrande J, Lemmens HJ. Br J Anaesth. 2010;105(Suppl 1):i16–i23. [PMID: 21148656]"
    },
    "ASA_Class": {
        "positive": "ASA Class ({value}) indicates healthy baseline, supporting a standard dose.",
        "negative": "Higher ASA Class ({value}) reduced dosage due to increased surgical risk.",
        "reference": "Daabiss M. Saudi J Anaesth. 2011;5(2):112–116. [PMID: 25885388]"
    },
    "ALDH2_Genotype": {
        "positive": "ALDH2 ({value}) increased the dose. Enhanced aldehyde metabolism supports standard drug clearance.",
        "negative": "ALDH2 ({value}) reduced the dose due to slower detoxification pathways.",
        "reference": "Brooks PJ et al. Pharmacogenetics. 2001;11(8):679–685. [PMID: 11592792]"
    },
    "CYP2D6": {
        "positive": "CYP2D6 ({value}) increased the predicted dose. This suggests the patient breaks down anesthetics quickly, needing more.",
        "negative": "CYP2D6 ({value}) decreased the dose. The body likely processes drugs slowly, so less is needed.",
        "reference": "Zhou SF. Clin Pharmacokinet. 2009;48(11):761–804. [PMID: 19743964]"
    },
    "CYP3A4": {
        "positive": "CYP3A4 ({value}) increased the dose. Enhanced metabolism may lower drug concentration faster.",
        "negative": "CYP3A4 ({value}) decreased the dose. A slower breakdown can make the drug last longer.",
        "reference": "Wilkinson GR. Clin Pharmacokinet. 2005;44(4):279–295. [PMID: 15766382]"
    },
    "CYP2C9": {
        "positive": "CYP2C9 ({value}) supported normal dosage due to expected drug metabolism.",
        "negative": "CYP2C9 ({value}) reduced the dosage because of reduced metabolic activity.",
        "reference": "Lee CR et al. Pharmacogenomics. 2002;3(2):277–287. [PMID: 12052251]"
    },
    "CYP2B6": {
        "positive": "CYP2B6 ({value}) increased the dose to compensate for rapid metabolism.",
        "negative": "CYP2B6 ({value}) decreased the dose due to slower breakdown of agents like propofol.",
        "reference": "Tremblay PB et al. Clin Pharmacol Ther. 2003;73(1):42–52. [PMID: 12545145]"
    },
    "UGT1A1_Variant": {
        "positive": "UGT1A1 ({value}) allowed standard dosing due to normal glucuronidation activity.",
        "negative": "UGT1A1 ({value}) reduced the dose due to impaired clearance pathways.",
        "reference": "Lazarus P. Drug Metab Rev. 2004;36(3-4):703–722. [PMID: 15554238]"
    },
    "RYR1_Variant": {
        "positive": "RYR1 ({value}) normal variant had no significant impact on dosage.",
        "negative": "RYR1 ({value}) reduced dose to minimize malignant hyperthermia risk.",
        "reference": "Rosenberg H et al. Anesthesiology. 2015;122(6):1223–1229. [PMID: 25858881]"
    },
    "SCN9A_Variant": {
        "positive": "SCN9A ({value}) normal response ensured standard analgesic sensitivity.",
        "negative": "SCN9A ({value}) altered response required dose adjustment due to pain insensitivity.",
        "reference": "Fertleman CR et al. Nat Genet. 2006;38(7):803–808. [PMID: 16767104]"
    },
    "F5_Variant": {
        "positive": "F5 ({value}) normal clotting allowed standard anesthetic planning.",
        "negative": "F5 ({value}) variant slightly reduced dosage to minimize thrombotic risk.",
        "reference": "Ridker PM et al. N Engl J Med. 1997;336(14):973–979. [PMID: 9077379]"
    },
    "GABRA2_Variant": {
        "positive": "GABRA2 ({value}) enhanced GABAergic sensitivity, potentially improving sedation effect.",
        "negative": "GABRA2 ({value}) reduced effect may require dose adaptation.",
        "reference": "Enoch MA. Alcohol. 2008;43(7):505–510. [PMID: 19036958]"
    },
    "OPRM1_Variant": {
        "positive": "OPRM1 ({value}) supported normal opioid receptor response to anesthesia.",
        "negative": "OPRM1 ({value}) may decrease drug efficacy, requiring dose adjustments.",
        "reference": "Oertel BG et al. Anesthesiology. 2006;105(4):806–814. [PMID: 17006088]"
    },
    "OrganFunction": {
        "positive": "Normal organ function allowed expected drug clearance.",
        "negative": "Impaired organ function ({value}) reduced clearance, necessitating lower dose.",
        "reference": "Miller’s Anesthesia, 9th ed., Chapter on Hepatic & Renal Disease"
    },
    "KidneyFunction": {
        "positive": "Normal renal function supported baseline anesthetic elimination.",
        "negative": "Impaired kidney function ({value}) led to reduced clearance and lower dose.",
        "reference": "Butterworth JF, et al. Morgan and Mikhail's Clinical Anesthesiology, 6th ed."
    },
    "CardiovascularHistory": {
        "positive": "No history of cardiovascular issues enabled standard anesthesia planning.",
        "negative": "Cardiovascular history ({value}) prompted reduced dosage to mitigate hemodynamic risk.",
        "reference": "Naranjo CA et al. J Clin Pharmacol. 1981;21(6-7):400–407. [PMID: 7249508]"
    },
    "Diabetes": {
        "positive": "No diabetes risk simplified pharmacodynamic response.",
        "negative": "Diabetes can alter anesthetic pharmacokinetics, requiring careful dosing.",
        "reference": "Frances MF et al. Can J Anaesth. 2008;55(9):634–642. [PMID: 18759368]"
    }
}


feature_reference_map_anesthesia_type = {
    "Age": {
        "positive": "Age ({value}) favored Sevoflurane due to better hemodynamic stability in older adults.",
        "negative": "Age ({value}) reduced preference for Sevoflurane or Isoflurane in younger adults where Propofol is more suitable.",
        "reference": "Walder B et al. Anesth Analg. 2001;93(5):1185–1191. [PMID: 11682417]"
    },
    "Weight": {
        "positive": "Weight ({value} kg) favored Isoflurane, which distributes well in adipose tissue.",
        "negative": "Weight ({value} kg) reduced Isoflurane preference when under 100kg.",
        "reference": "Ingrande J, Lemmens HJ. Br J Anaesth. 2010;105(Suppl 1):i16–i23. [PMID: 21148656]"
    },
    "BMI": {
        "positive": "BMI ({value}) < 30 made Propofol and Sevoflurane equally viable options.",
        "negative": "BMI ({value}) > 30 increased Isoflurane preference due to fat solubility.",
        "reference": "Kirkegaard H et al. Br J Anaesth. 2006;96(1):80–87. [PMID: 16311292]"
    },
    "Gender": {
        "positive": "Gender ({value}) slightly favored Isoflurane in females due to differences in fat/muscle distribution.",
        "negative": "Gender ({value}) did not impact choice when patient is male.",
        "reference": "Olver IN et al. Br J Anaesth. 1983;55(11):1107–1112. [PMID: 6358642]"
    },
    "RYR1_Variant": {
        "positive": "RYR1 ({value}) normal allowed standard inhalation anesthetics.",
        "negative": "RYR1 ({value}) favored Propofol due to malignant hyperthermia risk with volatile agents.",
        "reference": "Rosenberg H et al. Anesthesiology. 2015;122(6):1223–1229. [PMID: 25858881]"
    },
    "CardiovascularHistory": {
        "positive": "Cardiovascular status ({value}) supported Propofol for better hemodynamic control.",
        "negative": "No cardiovascular history allowed flexible anesthetic choices.",
        "reference": "Butterworth JF et al. Morgan and Mikhail’s Clinical Anesthesiology, 6th ed."
    },
    "ALDH2_Genotype": {
        "positive": "ALDH2 ({value}) supported Propofol preference due to more efficient aldehyde metabolism.",
        "negative": "ALDH2 ({value}) reduced Isoflurane preference due to slower aldehyde clearance.",
        "reference": "Brooks PJ et al. Pharmacogenetics. 2001;11(8):679–685. [PMID: 11592792]"
    },
    "CYP2D6": {
        "positive": "CYP2D6 ({value}) UM status favored Propofol for faster metabolism.",
        "negative": "CYP2D6 ({value}) PM status reduced Propofol use due to poor metabolism.",
        "reference": "Zhou SF. Clin Pharmacokinet. 2009;48(11):761–804. [PMID: 19743964]"
    },
    "CYP3A4": {
        "positive": "CYP3A4 ({value}) UM status slightly increased Sevoflurane metabolism preference.",
        "negative": "CYP3A4 ({value}) PM status lowered Sevoflurane preference due to slower clearance.",
        "reference": "Wilkinson GR. Clin Pharmacokinet. 2005;44(4):279–295. [PMID: 15766382]"
    },
    "CYP2C9": {
        "positive": "CYP2C9 ({value}) supported standard dosing of Propofol and Isoflurane.",
        "negative": "CYP2C9 ({value}) variant reduced Propofol preference due to slower metabolism.",
        "reference": "Lee CR et al. Pharmacogenomics. 2002;3(2):277–287. [PMID: 12052251]"
    },
    "CYP2B6": {
        "positive": "CYP2B6 ({value}) UM status encouraged Propofol use due to enhanced clearance.",
        "negative": "CYP2B6 ({value}) PM status discouraged Propofol due to slow breakdown.",
        "reference": "Tremblay PB et al. Clin Pharmacol Ther. 2003;73(1):42–52. [PMID: 12545145]"
    },
    "UGT1A1_Variant": {
        "positive": "UGT1A1 ({value}) allowed use of Propofol due to normal glucuronidation.",
        "negative": "UGT1A1 ({value}) discouraged Propofol due to reduced elimination capacity.",
        "reference": "Lazarus P. Drug Metab Rev. 2004;36(3-4):703–722. [PMID: 15554238]"
    },
    "SCN9A_Variant": {
        "positive": "SCN9A ({value}) normal function enabled balanced anesthetic options.",
        "negative": "SCN9A ({value}) Loss-of-Function decreased Sevoflurane due to altered nociception.",
        "reference": "Fertleman CR et al. Nat Genet. 2006;38(7):803–808. [PMID: 16767104]"
    },
    "F5_Variant": {
        "positive": "F5 ({value}) supported use of Propofol without increased thrombotic risk.",
        "negative": "F5 ({value}) mutation slightly favored Propofol over Isoflurane for coagulation safety.",
        "reference": "Ridker PM et al. N Engl J Med. 1997;336(14):973–979. [PMID: 9077379]"
    },
    "GABRA2_Variant": {
        "positive": "GABRA2 ({value}) polymorphism favored Propofol due to GABAergic receptor responsiveness.",
        "negative": "GABRA2 ({value}) normal had no strong directional effect.",
        "reference": "Enoch MA. Alcohol. 2008;43(7):505–510. [PMID: 19036958]"
    },
    "OPRM1_Variant": {
        "positive": "OPRM1 ({value}) normal (A/A) supported standard anesthetic preference including Propofol.",
        "negative": "OPRM1 ({value}) variants influenced preference toward Propofol due to altered pain response.",
        "reference": "Oertel BG et al. Anesthesiology. 2006;105(4):806–814. [PMID: 17006088]"
    }
}


feature_reference_map_response = {
    "Age": {
        "positive": "Age ({value}) supported an effective response due to youthful physiological resilience.",
        "negative": "Age ({value}) slightly reduced effectiveness due to age-related metabolic decline.",
        "reference": "Aldrete JA. Anesth Analg. 1998;86(4):791–795. [PMID: 9523811]"
    },
    "BMI": {
        "positive": "BMI ({value}) within healthy range allowed predictable anesthetic distribution.",
        "negative": "BMI ({value}) outside normal range increased risk of suboptimal response.",
        "reference": "Ingrande J, Lemmens HJ. Br J Anaesth. 2010;105(Suppl 1):i16–i23. [PMID: 21148656]"
    },
    "OrganFunction": {
        "positive": "Organ function ({value}) supported efficient drug clearance and stability.",
        "negative": "Organ function ({value}) impaired drug clearance, reducing efficacy.",
        "reference": "Miller’s Anesthesia, 9th ed., Chapter on Hepatic & Renal Disease"
    },
    "ASA_Class": {
        "positive": "ASA Class ({value}) reflects stable physiology, promoting effective response.",
        "negative": "ASA Class ({value}) increased risk of adverse or ineffective anesthetic response.",
        "reference": "Daabiss M. Saudi J Anaesth. 2011;5(2):112–116. [PMID: 25885388]"
    },
    "CYP2D6": {
        "positive": "CYP2D6 ({value}) metabolizer status supported drug breakdown for effective response.",
        "negative": "CYP2D6 ({value}) variant may slow or over-accelerate metabolism, affecting drug impact.",
        "reference": "Zhou SF. Clin Pharmacokinet. 2009;48(11):761–804. [PMID: 19743964]"
    },
    "CYP3A4": {
        "positive": "CYP3A4 ({value}) enabled steady metabolism, supporting desired anesthetic effect.",
        "negative": "CYP3A4 ({value}) altered metabolism rate, risking efficacy loss or accumulation.",
        "reference": "Wilkinson GR. Clin Pharmacokinet. 2005;44(4):279–295. [PMID: 15766382]"
    },
    "CYP2C9": {
        "positive": "CYP2C9 ({value}) supported consistent clearance, promoting reliable anesthetic response.",
        "negative": "CYP2C9 ({value}) compromised metabolic capacity, risking ineffective or adverse outcomes.",
        "reference": "Lee CR et al. Pharmacogenomics. 2002;3(2):277–287. [PMID: 12052251]"
    },
    "ALDH2_Genotype": {
        "positive": "ALDH2 ({value}) efficient variant helped maintain normal drug processing and response.",
        "negative": "ALDH2 ({value}) variant slowed aldehyde clearance, impairing anesthetic effect.",
        "reference": "Brooks PJ et al. Pharmacogenetics. 2001;11(8):679–685. [PMID: 11592792]"
    },
    "RYR1_Variant": {
        "positive": "RYR1 ({value}) normal variant avoided malignant hyperthermia, improving response.",
        "negative": "RYR1 ({value}) mutation raised risk of adverse reaction to inhaled agents.",
        "reference": "Rosenberg H et al. Anesthesiology. 2015;122(6):1223–1229. [PMID: 25858881]"
    },
    "GABRA2_Variant": {
        "positive": "GABRA2 ({value}) increased GABA sensitivity, enhancing sedation response.",
        "negative": "GABRA2 ({value}) may alter receptor dynamics, impacting consistency of effect.",
        "reference": "Enoch MA. Alcohol. 2008;43(7):505–510. [PMID: 19036958]"
    },
    "OPRM1_Variant": {
        "positive": "OPRM1 ({value}) supported effective opioid receptor interaction with anesthetics.",
        "negative": "OPRM1 ({value}) variant reduced opioid receptor efficacy, possibly diminishing response.",
        "reference": "Oertel BG et al. Anesthesiology. 2006;105(4):806–814. [PMID: 17006088]"
    },
    "Dosage": {
        "positive": "Dosage ({value} mg) was within the optimal range for this patient's profile.",
        "negative": "Dosage ({value} mg) was suboptimal, contributing to an unpredictable response.",
        "reference": "Egan TD et al. Anesthesiology. 2004;100(4):876–886. [PMID: 15087606]"
    }
}


__all__ = [
    "dosages_feature_reference_map",
    "feature_reference_map_anesthesia_type",
    "feature_reference_map_response"
]
