import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
#  1) Patient Count & Setup
# ----------------------------------------------------------------------

N = 40000  # number of synthetic records
np.random.seed(42)  # Fix seed for reproducibility

# ----------------------------------------------------------------------
#  2) Patient Demographics
# ----------------------------------------------------------------------


def calculate_bmi(weight, height_cm):
    """
    Compute BMI from weight (kg) and height (cm).
    Range typically clamped to avoid extreme outliers.
    """
    height_m = height_cm / 100.0
    return weight / (height_m * height_m)


def calculate_ibw_abw(gender, height_cm, tbw_kg):
    """
    Calculate Ideal Body Weight (IBW) and Adjusted Body Weight (ABW).
    Using Devine formula + 120% rule for ABW.

    If TBW >120% of IBW => ABW = IBW + 0.4*(TBW-IBW),
    else ABW = TBW.
    """
    inches = height_cm * 0.3937008
    if gender == 'Male':
        ibw = 50.0 + 2.3 * (inches - 60.0)
    else:
        ibw = 45.5 + 2.3 * (inches - 60.0)
    ibw = max(45.0, ibw)

    if tbw_kg > 1.2 * ibw:
        abw = ibw + 0.4 * (tbw_kg - ibw)
    else:
        abw = tbw_kg
    return ibw, abw


# Generate random distributions for age, height, weight
N_half = N // 2
age_group1 = np.random.normal(35, 8, N_half).astype(int)  # Younger half
age_group2 = np.random.normal(70, 10, N_half).astype(int)  # Elderly half
ages = np.concatenate([age_group1, age_group2])
ages = np.clip(ages, 18, 95)

weights = np.random.normal(75, 15, N).astype(int)
weights = np.clip(weights, 40, 150)

male_heights = np.random.normal(175, 7, int(N * 0.49)).astype(int)
female_heights = np.random.normal(162, 6, int(N * 0.51)).astype(int)
heights = np.concatenate([male_heights, female_heights])
heights = np.clip(heights, 140, 200)

genders = np.random.choice(['Male', 'Female'], N, p=[0.49, 0.51])

bmis, ibws, abws = [], [], []
for i in range(N):
    bmi_val = calculate_bmi(weights[i], heights[i])
    bmi_val = max(min(bmi_val, 50), 15)  # clamp BMI to [15..50]
    bmis.append(bmi_val)
    ibw_val, abw_val = calculate_ibw_abw(genders[i], heights[i], weights[i])
    ibws.append(ibw_val)
    abws.append(abw_val)

# ----------------------------------------------------------------------
#  3) Diet Column
# ----------------------------------------------------------------------
# 25% Vegetarian, 5% Vegan, 70% Non-vegetarian
diets = np.random.choice(
    ['Vegetarian', 'Vegan', 'Non-vegetarian'],
    size=N,
    p=[0.25, 0.05, 0.70]
)

# ----------------------------------------------------------------------
#  4) Genetic Variants with Realistic Frequencies
# ----------------------------------------------------------------------

aldh2_genotypes = np.random.choice(
    ['G/G', 'G/A', 'A/A'], N, p=[0.55, 0.35, 0.10])
cyp2d6_types = np.random.choice(
    ['PM', 'IM', 'NM', 'UM'], N, p=[0.05, 0.20, 0.65, 0.10])
cyp3a4_types = np.random.choice(['PM', 'UM'], N, p=[0.10, 0.90])
cyp2c9_types = np.random.choice(['*1/*1', '*1/*2', '*2/*2', '*1/*3', '*2/*3', '*3/*3'],
                                N, p=[0.50, 0.20, 0.05, 0.15, 0.05, 0.05])
cyp2b6_types = np.random.choice(['PM', 'UM', 'NM'], N, p=[0.10, 0.10, 0.80])
ugt1a1_variants = np.random.choice(
    ['Normal Function', 'Reduced Function'], N, p=[0.75, 0.25])
ryr1_variants = np.random.choice(
    ['Normal', 'Variant'], N, p=[0.99999, 0.00001])
scn9a_variants = np.random.choice(
    ['Normal', 'Loss-of-Function'], N, p=[0.9999, 0.0001])
f5_variants = np.random.choice(
    ['No Mutation', 'Factor V Leiden'], N, p=[0.95, 0.05])
gabra2_variants = np.random.choice(
    ['Normal', 'Polymorphism'], N, p=[0.90, 0.10])
oprm1_variants = np.random.choice(
    ['A/A', 'A/G', 'G/G'], N, p=[0.50, 0.40, 0.10])

# ----------------------------------------------------------------------
#  5) Medical History & ASA Classification
# ----------------------------------------------------------------------


def generate_medical_history(age, bmi, diet):
    """
    - Veg/Vegan => +0.05 organ impairment probability from B12 deficiency
    - Non-veg   => +5% cardiovascular risk
    - No direct +0.1 on ASA => let ASA rise naturally if organ/cardiac issues
    """
    prob_imp = np.clip(0.05 + (age-40)*0.005 + max(0, bmi-30)*0.01, 0.05, 0.50)

    # Extra impairment if Veg/Vegan
    if diet in ['Vegetarian', 'Vegan']:
        prob_imp += 0.05

    organ_fn = np.random.choice(
        ['Normal', 'Impaired'], p=[1-prob_imp, prob_imp])
    kidney_fn = np.random.choice(
        ['Normal', 'Impaired'], p=[1-prob_imp, prob_imp])

    # Cardiovascular risk
    base_cardiovascular = np.clip(
        0.10 + (age-50)*0.01 + max(0, bmi-25)*0.02, 0.10, 0.60
    )
    if diet == 'Non-vegetarian':
        base_cardiovascular *= 1.05

    p_htn = base_cardiovascular * 0.7
    p_cvd = base_cardiovascular * 0.3
    p_none = 1 - (p_htn + p_cvd)
    cardio = np.random.choice(
        ['No History', 'Hypertension', 'Cardiovascular Disease'],
        p=[p_none, p_htn, p_cvd]
    )

    prob_diab = np.clip(0.05 + (age-45)*0.008 +
                        max(0, bmi-30)*0.02, 0.05, 0.40)
    diab = np.random.choice(['No', 'Yes'], p=[1-prob_diab, prob_diab])

    # ASA
    asa_score = 1
    if age > 65:
        asa_score += 1
    if cardio != 'No History' or diab == 'Yes':
        asa_score += 1
    if organ_fn == 'Impaired' or kidney_fn == 'Impaired':
        asa_score += 1
    asa_class = min(asa_score, 4)

    return organ_fn, kidney_fn, cardio, diab, asa_class


medical_data = [generate_medical_history(
    a, b, d) for a, b, d in zip(ages, bmis, diets)]
organ_functions, kidney_functions, cardiovascular_history, diabetes, asa_classes = zip(
    *medical_data)


def determine_medications(cardiovascular, diabetes):
    if cardiovascular == 'No History' and diabetes == 'No':
        return np.random.choice(['No History', 'Other'], p=[0.75, 0.25])
    elif cardiovascular == 'Hypertension':
        return np.random.choice(['Beta-Blocker', 'ACE Inhibitor'], p=[0.6, 0.4])
    else:
        return np.random.choice(['Beta-Blocker', 'ACE Inhibitor', 'Statin'], p=[0.3, 0.4, 0.3])


current_meds = [
    determine_medications(c, d) for c, d in zip(cardiovascular_history, diabetes)
]

# A few procedure types
procedure_types = np.random.choice(
    ['Minor', 'Major'], N, p=[0.6, 0.4])


# ----------------------------------------------------------------------
#  6) Adjusted Metabolism
# ----------------------------------------------------------------------
VMAX = {'Propofol': {'CYP2B6': 100.0, 'UGT1A1': 80.0}}
KM = {'Propofol': {'CYP2B6':  20.0,  'UGT1A1': 15.0}}
cyp2b6_multiplier = {'PM': 0.4, 'NM': 1.0, 'UM': 1.5}
ugt1a1_multiplier = {'Normal Function': 1.0, 'Reduced Function': 0.6}


def michaelis_menten(dose, vmax, km):
    return (vmax*dose)/(km + dose)


def adjust_dose_for_metabolism(agent, dose, cyp2b6_type, ugt1a1_type):
    """
    Adjust final dose based on the patient's metabolism capacity.
    Poor metabolizers => lower final dose. Ultra-rapid => raise dose.
    """
    if agent != 'Propofol':
        return dose

    cyp_factor = 0.6
    ugt_factor = 0.4
    adj_vmax_cyp = VMAX['Propofol']['CYP2B6'] * \
        cyp2b6_multiplier.get(cyp2b6_type, 1.0)
    adj_vmax_ugt = VMAX['Propofol']['UGT1A1'] * \
        ugt1a1_multiplier.get(ugt1a1_type, 1.0)

    cyp_metab = michaelis_menten(dose, adj_vmax_cyp, KM['Propofol']['CYP2B6'])
    ugt_metab = michaelis_menten(dose, adj_vmax_ugt, KM['Propofol']['UGT1A1'])
    total_metab = cyp_factor*cyp_metab + ugt_factor*ugt_metab

    normal_cap = 120.0
    if total_metab < normal_cap:
        dose_factor = 1.0 - (normal_cap - total_metab)/normal_cap
    else:
        dose_factor = 1.0 + (total_metab - normal_cap)/normal_cap

    dose_factor = np.clip(dose_factor, 0.5, 1.5)
    return dose * dose_factor

# ----------------------------------------------------------------------
#  7) General Anesthesia Selection & Dosage
# ----------------------------------------------------------------------


def determine_general_anesthesia_type(proc_type, age, ryr1_variant, bmi, gender,
                                      cardiovascular_history, aldh2_genotype,
                                      cyp2d6, cyp3a4, cyp2c9, cyp2b6, ugt1a1,
                                      scn9a, f5, gabra2, oprm1):
    """
    Weighted approach.
    """

    agents = {
        'Propofol': 1.0,
        'Sevoflurane': 1.0,
        'Isoflurane': 1.0,
        'Ketamine': 1.0,
        'Thiopental': 1.0,
        'Desflurane': 1.0,
        'Halothane': 1.0
    }

    # Some weighting examples
    if age > 65:
        agents['Sevoflurane'] *= 1.5
        agents['Isoflurane'] *= 1.2
        agents['Halothane'] *= 0.7
    if ryr1_variant == 'Variant':
        agents['Propofol'] *= 1.5
        for a in ['Sevoflurane', 'Isoflurane', 'Desflurane', 'Halothane']:
            agents[a] *= 0.5
        agents['Ketamine'] *= 1.4
        agents['Thiopental'] *= 0.8
    if bmi > 30:
        agents['Isoflurane'] *= 1.3
        agents['Desflurane'] *= 1.3
        agents['Halothane'] *= 0.8
    if cardiovascular_history != 'No History':
        agents['Propofol'] *= 1.3
        agents['Sevoflurane'] *= 0.8
        agents['Isoflurane'] *= 0.9
        agents['Halothane'] *= 0.7
    if aldh2_genotype == 'A/A':
        agents['Propofol'] *= 1.2
        agents['Isoflurane'] *= 0.8

    # cyp2d6
    if cyp2d6 == 'PM':
        agents['Propofol'] *= 0.7
        agents['Ketamine'] *= 1.1
    elif cyp2d6 == 'UM':
        agents['Propofol'] *= 1.3

    # cyp3a4 => ±15% on Sevo
    if cyp3a4 == 'PM':
        agents['Sevoflurane'] *= 0.85
    else:
        agents['Sevoflurane'] *= 1.15

    if cyp2c9 in ['*2/*2', '*3/*3', '*2/*3']:
        agents['Propofol'] *= 0.9
        agents['Isoflurane'] *= 0.95
        agents['Thiopental'] *= 0.9
    if cyp2b6 == 'PM':
        agents['Propofol'] *= 0.9
    elif cyp2b6 == 'UM':
        agents['Propofol'] *= 1.1
    if ugt1a1 == 'Reduced Function':
        agents['Propofol'] *= 0.85
    if scn9a == 'Loss-of-Function':
        agents['Propofol'] *= 1.1
        agents['Sevoflurane'] *= 0.9
    if f5 == 'Factor V Leiden':
        agents['Propofol'] *= 1.1
    if gabra2 == 'Polymorphism':
        agents['Propofol'] *= 1.2
        agents['Sevoflurane'] *= 0.9
    if oprm1 == 'G/G':
        agents['Propofol'] *= 1.1
    elif oprm1 == 'A/G':
        agents['Propofol'] *= 1.05

    if proc_type == 'Major':
        agents['Desflurane'] *= 1.3
        agents['Propofol'] *= 0.8

    return max(agents, key=agents.get)


def pick_dosing_weight(agent, ibw, abw, tbw):
    """
    For lipophilic agents, prefer ABW if TBW >120% of IBW.
    """
    lipophilic = {'Propofol', 'Ketamine', 'Thiopental'}
    if agent in lipophilic:
        if tbw > 1.2*ibw:
            return abw
        else:
            return tbw
    else:
        return tbw


def determine_general_dosage(
    age, tbw, ibw, abw, chosen_agent,
    organ_fn, kidney_fn, cardio_hist, diabetes, asa_class,
    aldh2_genotype, cyp2d6, cyp3a4, cyp2c9, cyp2b6, ugt1a1,
    ryr1, scn9a, f5, gabra2, oprm1,
    diet, bmi
):
    """
    Includes refined diet-BMI logic:
     - Veg/Vegan => base -3% if BMI<25, else -5%
       BUT if vegan AND BMI<20 => extra -2%
    """

    dosing_wt = pick_dosing_weight(chosen_agent, ibw, abw, tbw)
    # baseline ~1.5 mg/kg for demonstration
    base_dosage = dosing_wt * 1.5

    # ASA factor
    asa_factors = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7}
    base_dosage *= asa_factors[asa_class]

    # Age, organ function
    if age > 70:
        base_dosage *= 0.9
    if organ_fn == 'Impaired':
        base_dosage *= 0.8
    if kidney_fn == 'Impaired':
        base_dosage *= 0.85
    if cardio_hist != 'No History':
        base_dosage *= 0.9
    if diabetes == 'Yes':
        base_dosage *= 0.95

    # Metabolizer adjustments
    c2d6map = {'PM': 0.7, 'IM': 0.85, 'NM': 1.0, 'UM': 1.2}
    base_dosage *= c2d6map.get(cyp2d6, 1.0)

    if cyp3a4 == 'PM':
        base_dosage *= 0.8
    elif cyp3a4 == 'UM':
        base_dosage *= 1.15

    if cyp2c9 in ['*2/*2', '*3/*3']:
        base_dosage *= 0.75
    elif cyp2c9 in ['*1/*2', '*1/*3']:
        base_dosage *= 0.9
    elif cyp2c9 == '*2/*3':
        base_dosage *= 0.8

    if cyp2b6 == 'PM':
        base_dosage *= 0.85
    elif cyp2b6 == 'UM':
        base_dosage *= 1.1

    if ugt1a1 == 'Reduced Function':
        base_dosage *= 0.9
    if ryr1 == 'Variant':
        base_dosage *= 0.75
    if scn9a == 'Loss-of-Function':
        base_dosage *= 0.95
    if f5 == 'Factor V Leiden':
        base_dosage *= 0.95
    if gabra2 == 'Polymorphism':
        base_dosage *= 1.1
    if oprm1 == 'G/G':
        base_dosage *= 1.15
    elif oprm1 == 'A/G':
        base_dosage *= 1.05

    # Refined diet-based adjustments
    if diet == 'Vegan':
        # if BMI <20 => an extra -2%
        # first handle the usual -3 or -5
        if bmi < 25:
            base_dosage *= 0.97
        else:
            base_dosage *= 0.95
        if bmi < 20:
            base_dosage *= 0.98  # extra -2%
    elif diet == 'Vegetarian':
        # -3% if BMI<25, else -5%
        if bmi < 25:
            base_dosage *= 0.97
        else:
            base_dosage *= 0.95
    else:
        # Non-veg => +5%
        base_dosage *= 1.05

    # Random final variation
    base_dosage *= np.random.normal(1, 0.05)

    # Adjust for metabolism
    final_dose = adjust_dose_for_metabolism(
        chosen_agent, base_dosage, cyp2b6, ugt1a1)

    return round(np.clip(final_dose, 0, 300), 2)


# ----------------------------------------------------------------------
#  8) Logistic-Based Response for General Anesthesia
# ----------------------------------------------------------------------
ADVERSE_REACTIONS = ['Hypotension', 'Arrhythmia',
                     'Respiratory', 'Nausea/Vomiting']

COEF_EFFECTIVE = {
    'intercept': 1.0,
    'dose': 0.01,   # Positive => higher dose => higher chance of "Effective"
    'ASA3_4': -0.3,
    'cardio': -0.2
}
COEF_ADVERSE = {
    'intercept': -2.5,
    'dose': 0.05,   # Higher dose => more adverse
    'ASA3_4': 0.5,
    'cardio': 0.5
}


def logistic(z):
    return 1.0/(1.0 + np.exp(-z))


def pick_specific_adverse(agent):
    """
    If agent is Ketamine or Halothane => 4 possible adverse events equally.
    Else => 3 events (exclude 'Nausea/Vomiting').
    """
    if agent in ['Ketamine', 'Halothane']:
        return np.random.choice(['Hypotension', 'Arrhythmia', 'Respiratory', 'Nausea/Vomiting'])
    else:
        return np.random.choice(['Hypotension', 'Arrhythmia', 'Respiratory'])


def determine_general_response(age, asa_class, cardio, dose, agent):

    asa34 = 1 if asa_class >= 3 else 0
    card = 1 if cardio != 'No History' else 0

    z_eff = (COEF_EFFECTIVE['intercept']
             + COEF_EFFECTIVE['dose'] * (dose/100.0)
             + COEF_EFFECTIVE['ASA3_4']*asa34
             + COEF_EFFECTIVE['cardio']*card)
    p_eff = logistic(z_eff)

    z_adv = (COEF_ADVERSE['intercept']
             + COEF_ADVERSE['dose'] * (dose/100.0)
             + COEF_ADVERSE['ASA3_4']*asa34
             + COEF_ADVERSE['cardio']*card)
    p_adv = logistic(z_adv)

    total = p_eff + p_adv
    if total > 1.0:
        # normalize
        p_eff /= total
        p_adv /= total
        total = 1.0
    p_ineff = 1.0 - total

    r = np.random.rand()
    if r < p_eff:
        return 'Effective'
    elif r < (p_eff + p_adv):
        # pick an event, with extra chance of 'Nausea/Vomiting' for Ketamine/Halothane
        return 'Adverse: ' + pick_specific_adverse(agent)
    else:
        return 'Ineffective'


# ----------------------------------------------------------------------
# Main: Build final dataset
# ----------------------------------------------------------------------
general_agents = []
general_dosages = []
general_responses = []

for i in range(N):
    proc_type = procedure_types[i]

    # 1) General anesthesia agent (some sedation for Dental)
    g_agent = determine_general_anesthesia_type(
        proc_type,
        age=ages[i],
        ryr1_variant=ryr1_variants[i],
        bmi=bmis[i],
        gender=genders[i],
        cardiovascular_history=cardiovascular_history[i],
        aldh2_genotype=aldh2_genotypes[i],
        cyp2d6=cyp2d6_types[i],
        cyp3a4=cyp3a4_types[i],
        cyp2c9=cyp2c9_types[i],
        cyp2b6=cyp2b6_types[i],
        ugt1a1=ugt1a1_variants[i],
        scn9a=scn9a_variants[i],
        f5=f5_variants[i],
        gabra2=gabra2_variants[i],
        oprm1=oprm1_variants[i]
    )

    # 2) General dosage
    g_dose = determine_general_dosage(
        age=ages[i],
        tbw=weights[i],
        ibw=ibws[i],
        abw=abws[i],
        chosen_agent=g_agent,
        organ_fn=organ_functions[i],
        kidney_fn=kidney_functions[i],
        cardio_hist=cardiovascular_history[i],
        diabetes=diabetes[i],
        asa_class=asa_classes[i],
        aldh2_genotype=aldh2_genotypes[i],
        cyp2d6=cyp2d6_types[i],
        cyp3a4=cyp3a4_types[i],
        cyp2c9=cyp2c9_types[i],
        cyp2b6=cyp2b6_types[i],
        ugt1a1=ugt1a1_variants[i],
        ryr1=ryr1_variants[i],
        scn9a=scn9a_variants[i],
        f5=f5_variants[i],
        gabra2=gabra2_variants[i],
        oprm1=oprm1_variants[i],
        diet=diets[i],
        bmi=bmis[i]
    )

    # 3) General response
    g_resp = determine_general_response(
        age=ages[i],
        asa_class=asa_classes[i],
        cardio=cardiovascular_history[i],
        dose=g_dose,
        agent=g_agent
    )

    general_agents.append(g_agent)
    general_dosages.append(g_dose)
    general_responses.append(g_resp)

df = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Height_cm': heights,
    'Weight_kg': weights,
    'BMI': bmis,
    'IBW': ibws,
    'ABW': abws,
    'Diet': diets,
    'OrganFunction': organ_functions,
    'KidneyFunction': kidney_functions,
    'CardiovascularHistory': cardiovascular_history,
    'Diabetes': diabetes,
    'CurrentMedications': current_meds,
    'ProcedureType': procedure_types,
    'ASA_Class': asa_classes,
    'ALDH2_Genotype': aldh2_genotypes,
    'CYP2D6_Type': cyp2d6_types,
    'CYP3A4_Type': cyp3a4_types,
    'CYP2C9_Type': cyp2c9_types,
    'CYP2B6_Type': cyp2b6_types,
    'UGT1A1_Variant': ugt1a1_variants,
    'RYR1_Variant': ryr1_variants,
    'SCN9A_Variant': scn9a_variants,
    'F5_Variant': f5_variants,
    'GABRA2_Variant': gabra2_variants,
    'OPRM1_Variant': oprm1_variants,
    'General_AnesthesiaType': general_agents,
    'General_Dosage': general_dosages,
    'General_Response': general_responses
})

df.to_csv('enhanced_anesthesia_dataset_v4.csv', index=False)

# Quick checks
print("Dataset generation complete. Sample records:")
print(df.head(10))

print("\nASA class distribution:\n",
      df['ASA_Class'].value_counts(normalize=True))
print("\nDiet distribution:\n", df['Diet'].value_counts(normalize=True))
print("\nGeneral dosage stats by agent:\n",
      df.groupby('General_AnesthesiaType')['General_Dosage'].describe())
