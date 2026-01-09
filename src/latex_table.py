import json
import os
import argparse
from utils.file_management import list_files

if __name__ == '__main__':

    description = 'Script to compute the metrics of the test dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Metrics-computation')
    parser.add_argument('eval_dir', type=str,
                        help='The evaluation directory where metrics are saved')
    args = parser.parse_args()

    metrics_files = list_files(args.eval_dir, pattern=r'metrics.*\.json')

    all_scores = {}
    for file in metrics_files:
        with open(file) as f:
            metrics = json.load(f)
        name = os.path.basename(os.path.dirname(file)) + os.path.basename(file).replace('metrics', '').replace('.json', '')
        name = name.replace('generated_', '').replace('_', ' ').replace('post processed', '')
        try:
            all_scores[name] = {
                'v2v_dist': metrics['v2v_dist'],
                'chamfer': metrics['chamfer'],
                'normal':  metrics['normal'],
                'penetration': metrics['penetration'],
                'curvature': metrics['curvature'],
                'wrinkling': metrics['wrinkling'],
                'edge_strain': metrics['edge_strain'],
                'area_strain': metrics['area_strain'],
                'height_diff': metrics['height_diff'],
                'mass_center': metrics['mass_center'],
                'mass_center_height': metrics['mass_center_height'],
            }
        except KeyError:
            print(f"Warning: Missing metrics in file {file}, skipping.")

    def sort_key(item):
        _, scores = item
        return (scores['v2v_dist'])
    all_scores = dict(sorted(all_scores.items(), key=sort_key, reverse=True))

    best_scores = {}
    for metric in ['v2v_dist', 'chamfer', 'normal', 'penetration', 'curvature', 'wrinkling', 'edge_strain', 'area_strain', 'height_diff', 'mass_center', 'mass_center_height']:
        best_name = min(all_scores.keys(), key=lambda name: all_scores[name][metric])
        best_scores[metric] = (best_name, all_scores[best_name][metric])

    latex_table = ""
    latex_table += "  & $E_v$ & $E_{CD}$ & $E_n$ & $E_c$ & $E_b$ & $E_s$ & $E_d$ \\\\ \n"
    latex_table += "\\hline\n"

    for name, scores in all_scores.items():
        v2v_dist = f"{scores['v2v_dist']*100:.4f}"
        chamfer = f"{scores['chamfer']*100:.4f}"
        normal = f"{scores['normal']:.4f}"
        penetration = f"{scores['penetration'] * 100:.4f}"
        curvature = f"{scores['curvature']:.4f}"
        wrinkling = f"{scores['wrinkling']:.4f}"
        edge_strain = f"{scores['edge_strain']:.4f}"
        area_strain = f"{scores['area_strain']*1e5:.4f}"
        height_diff = f"{scores['height_diff']*100:.4f}"
        mass_center = f"{scores['mass_center']*100:.4f}"
        mass_center_height = f"{scores['mass_center_height']*100:.4f}"

        if name == best_scores['v2v_dist'][0]:
            v2v_dist = "\\textbf{" + v2v_dist + "}"
        if name == best_scores['chamfer'][0]:
            chamfer = "\\textbf{" + chamfer + "}"
        if name == best_scores['normal'][0]:
            normal = "\\textbf{" + normal + "}"
        if name == best_scores['penetration'][0]:
            penetration = "\\textbf{" + penetration + "}"
        if name == best_scores['curvature'][0]:
            curvature = "\\textbf{" + curvature + "}"
        if name == best_scores['wrinkling'][0]:
            wrinkling = "\\textbf{" + wrinkling + "}"
        if name == best_scores['edge_strain'][0]:
            edge_strain = "\\textbf{" + edge_strain + "}"
        if name == best_scores['area_strain'][0]:
            area_strain = "\\textbf{" + area_strain + "}"
        if name == best_scores['height_diff'][0]:
            height_diff = "\\textbf{" + height_diff + "}"
        if name == best_scores['mass_center'][0]:
            mass_center = "\\textbf{" + mass_center + "}"
        if name == best_scores['mass_center_height'][0]:
            mass_center_height = "\\textbf{" + mass_center_height + "}"

        latex_table += f"{name} & {v2v_dist} & {chamfer} & {normal} & {penetration} & {wrinkling} & {area_strain} & {mass_center} \\\\ \n"

    print(latex_table)
