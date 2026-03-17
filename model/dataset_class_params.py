from utils import intersect_dicts

def get_dropped_moa_classes(dropped_moa,
                            negative_controls=['Ciprofloxacin', 'Cefsulodin', 'Relebactam', 'Ceftriaxone', 'Doxycycline']):
    dropped_classes = []
    doses = ['0.125xIC50', '0.25xIC50', '0.5xIC50', '1xIC50']
    for moa in dropped_moa:
        if moa == 'PBP1':
            class_names = ['Cefsulodin', 'PenicillinG', 'Sulbactam']
        elif moa == 'PBP2':
            class_names = ['Avibactam', 'Mecillinam', 'Meropenem', 'Relebactam', 'Clavulanate']
        elif moa == 'PBP3':
            class_names = ['Aztreonam', 'Ceftriaxone', 'Cefepime']
        elif moa == 'Gyrase':
            class_names = ['Ciprofloxacin', 'Levofloxacin', 'Norfloxacin']
        elif moa == 'Ribosome':
            class_names = ['Doxycycline', 'Kanamycin', 'Chloramphenicol', 'Clarithromycin']
        elif moa == 'Membrane':
            class_names = ['Colistin', 'PolymyxinB']

        class_names_ = [c for c in negative_controls if c not in class_names]
        class_names += class_names_
        classes_ = [f'{x}_{d}' for d in doses for x in class_names]
        dropped_classes += classes_

    return dropped_classes

def get_e_coli_moa_dict():
    e_coli_moa_dict = {'Avibactam_0.125xIC50': 'Cell wall (PBP 2)',
                      'Avibactam_0.25xIC50': 'Cell wall (PBP 2)',
                      'Avibactam_0.5xIC50': 'Cell wall (PBP 2)',
                      'Avibactam_1xIC50': 'Cell wall (PBP 2)',
                      'Aztreonam_0.125xIC50': 'Cell wall (PBP 3)',
                      'Aztreonam_0.25xIC50': 'Cell wall (PBP 3)',
                      'Aztreonam_0.5xIC50': 'Cell wall (PBP 3)',
                      'Aztreonam_1xIC50': 'Cell wall (PBP 3)',
                      'Cefepime_0.125xIC50': 'Cell wall (PBP 3)',
                      'Cefepime_0.25xIC50': 'Cell wall (PBP 3)',
                      'Cefepime_0.5xIC50': 'Cell wall (PBP 3)',
                      'Cefepime_1xIC50': 'Cell wall (PBP 3)',
                      'Cefsulodin_0.125xIC50': 'Cell wall (PBP 1)',
                      'Cefsulodin_0.25xIC50': 'Cell wall (PBP 1)',
                      'Cefsulodin_0.5xIC50': 'Cell wall (PBP 1)',
                      'Cefsulodin_1xIC50': 'Cell wall (PBP 1)',
                      'Ceftriaxone_0.125xIC50': 'Cell wall (PBP 3)',
                      'Ceftriaxone_0.25xIC50': 'Cell wall (PBP 3)',
                      'Ceftriaxone_0.5xIC50': 'Cell wall (PBP 3)',
                      'Ceftriaxone_1xIC50': 'Cell wall (PBP 3)',
                      'Chloramphenicol_0.125xIC50': 'Ribosome',
                      'Chloramphenicol_0.25xIC50': 'Ribosome',
                      'Chloramphenicol_0.5xIC50': 'Ribosome',
                      'Chloramphenicol_1xIC50': 'Ribosome',
                      'Ciprofloxacin_0.125xIC50': 'Gyrase',
                      'Ciprofloxacin_0.25xIC50': 'Gyrase',
                      'Ciprofloxacin_0.5xIC50': 'Gyrase',
                      'Ciprofloxacin_1xIC50': 'Gyrase',
                      'Clarithromycin_0.125xIC50': 'Ribosome',
                      'Clarithromycin_0.25xIC50': 'Ribosome',
                      'Clarithromycin_0.5xIC50': 'Ribosome',
                      'Clarithromycin_1xIC50': 'Ribosome',
                      'Clavulanate_0.125xIC50': 'Cell wall (PBP 2)',
                      'Clavulanate_0.25xIC50': 'Cell wall (PBP 2)',
                      'Clavulanate_0.5xIC50': 'Cell wall (PBP 2)',
                      'Clavulanate_1xIC50': 'Cell wall (PBP 2)',
                      'Colistin_0.125xIC50': 'Membrane integrity',
                      'Colistin_0.25xIC50': 'Membrane integrity',
                      'Colistin_0.5xIC50': 'Membrane integrity',
                      'Colistin_1xIC50': 'Membrane integrity',
                      'DMSO_0.125xIC50': 'Control',
                      'DMSO_0.25xIC50': 'Control',
                      'DMSO_0.5xIC50': 'Control',
                      'DMSO_1xIC50': 'Control',
                      'Doxycycline_0.125xIC50': 'Ribosome',
                      'Doxycycline_0.25xIC50': 'Ribosome',
                      'Doxycycline_0.5xIC50': 'Ribosome',
                      'Doxycycline_1xIC50': 'Ribosome',
                      'Kanamycin_0.125xIC50': 'Ribosome',
                      'Kanamycin_0.25xIC50': 'Ribosome',
                      'Kanamycin_0.5xIC50': 'Ribosome',
                      'Kanamycin_1xIC50': 'Ribosome',
                      'Levofloxacin_0.125xIC50': 'Gyrase',
                      'Levofloxacin_0.25xIC50': 'Gyrase',
                      'Levofloxacin_0.5xIC50': 'Gyrase',
                      'Levofloxacin_1xIC50': 'Gyrase',
                      'Mecillinam_0.125xIC50': 'Cell wall (PBP 2)',
                      'Mecillinam_0.25xIC50': 'Cell wall (PBP 2)',
                      'Mecillinam_0.5xIC50': 'Cell wall (PBP 2)',
                      'Mecillinam_1xIC50': 'Cell wall (PBP 2)',
                      'Meropenem_0.125xIC50': 'Cell wall (PBP 2)',
                      'Meropenem_0.25xIC50': 'Cell wall (PBP 2)',
                      'Meropenem_0.5xIC50': 'Cell wall (PBP 2)',
                      'Meropenem_1xIC50': 'Cell wall (PBP 2)',
                      'Norfloxacin_0.125xIC50': 'Gyrase',
                      'Norfloxacin_0.25xIC50': 'Gyrase',
                      'Norfloxacin_0.5xIC50': 'Gyrase',
                      'Norfloxacin_1xIC50': 'Gyrase',
                      'PenicillinG_0.125xIC50': 'Cell wall (PBP 1)',
                      'PenicillinG_0.25xIC50': 'Cell wall (PBP 1)',
                      'PenicillinG_0.5xIC50': 'Cell wall (PBP 1)',
                      'PenicillinG_1xIC50': 'Cell wall (PBP 1)',
                      'PolymyxinB_0.125xIC50': 'Membrane integrity',
                      'PolymyxinB_0.25xIC50': 'Membrane integrity',
                      'PolymyxinB_0.5xIC50': 'Membrane integrity',
                      'PolymyxinB_1xIC50': 'Membrane integrity',
                      'Relebactam_0.125xIC50': 'Cell wall (PBP 2)',
                      'Relebactam_0.25xIC50': 'Cell wall (PBP 2)',
                      'Relebactam_0.5xIC50': 'Cell wall (PBP 2)',
                      'Relebactam_1xIC50': 'Cell wall (PBP 2)',
                      'Rifampicin_0.125xIC50': 'RNA polymerase',
                      'Rifampicin_0.25xIC50': 'RNA polymerase',
                      'Rifampicin_0.5xIC50': 'RNA polymerase',
                      'Rifampicin_1xIC50': 'RNA polymerase',
                      'Sulbactam_0.125xIC50': 'Cell wall (PBP 1)',
                      'Sulbactam_0.25xIC50': 'Cell wall (PBP 1)',
                      'Sulbactam_0.5xIC50': 'Cell wall (PBP 1)',
                      'Sulbactam_1xIC50': 'Cell wall (PBP 1)',
                      'Trimethoprim_0.125xIC50': 'DNA synthesis',
                      'Trimethoprim_0.25xIC50': 'DNA synthesis',
                      'Trimethoprim_0.5xIC50': 'DNA synthesis',
                      'Trimethoprim_1xIC50': 'DNA synthesis'}

    return e_coli_moa_dict

def get_class_merge_dict(use_e_coli_moa=False):
    class_merge_dict = {'DMSO_0.125xIC50': 'Control',
                        'DMSO_0.25xIC50': 'Control',
                        'DMSO_0.5xIC50': 'Control',
                        'DMSO_1xIC50': 'Control',
                        'DMSO_1': 'Control',
                        'DMSO_2': 'Control',
                        'DMSO_3': 'Control',
                        'DMSO_4': 'Control',
                        'DMSO_5': 'Control',
                        'DMSO_6': 'Control'}

    if use_e_coli_moa:
        e_coli_moa_dict = get_e_coli_moa_dict()

        class_merge_dict = intersect_dicts(class_merge_dict, e_coli_moa_dict)

    return class_merge_dict

def get_moa_class_weights(classes):
    e_coli_moa_dict = get_e_coli_moa_dict()

    return [1 - (list(e_coli_moa_dict.values()).count(x) / len(e_coli_moa_dict.values())) for x in classes]
