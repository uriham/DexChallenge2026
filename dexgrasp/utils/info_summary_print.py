import yaml
import numpy as np
import os


def _as_float(value):
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return 0.0


def save_results_summary(results, filename, to_yaml=False):
    # output_data={}

    mean_succ_rate = float(np.mean(results['total_succ_rates'])) if results['total_succ_rates'] else 0.0
    total_success_num = _as_float(results.get('total_success_num', 0.0))
    total_trials = int(results.get('total_trials', 0))
    weighted_success_rate = total_success_num / total_trials if total_trials > 0 else mean_succ_rate
    results['total_succ_rates'] = mean_succ_rate
    results['total_success_num'] = total_success_num
    results['total_trials'] = total_trials
    results['weighted_success_rate'] = float(weighted_success_rate)
    print(f"Mean Success Rate: {mean_succ_rate:.4f}\n")
    print(
        "Weighted Success Rate: {:.4f} ({:.1f}/{})\n".format(
            weighted_success_rate,
            total_success_num,
            total_trials,
        )
    )
    # for key, val in results.items():
    #     if isinstance(val,list) and bool(1-isinstance(val[0],str)):
    #         results[key] = float(np.mean(val))

    # output_data = {
    #     # 'dataset_name': results.get('dataset_name', ''),
    #     'mean_total_succ_rate': round(mean_succ_rate, 4),
    #     'detail': results['detail']
    # }

    if to_yaml:
        os.makedirs("./results", exist_ok=True)
        filename = filename if filename.endswith('.yaml') else filename + '.yaml'
        with open("./results/{}".format(filename), 'w') as f:
            yaml.dump(results, f, allow_unicode=True)
            a=1
    else:
        filename = filename if filename.endswith('.txt') else filename + '.txt'
        with open("./results/{}".format(filename), 'w') as f:
            f.write(f"Dataset: {output_data['dataset_name']}\n")
            f.write(f"Mean Success Rate: {output_data['mean_total_succ_rate']:.4f}\n")
            f.write("Details:\n")
            for line in output_data['detail']:
                if isinstance(line, list):
                    f.write("  " + " | ".join(map(str, line)) + "\n")
                else:
                    f.write("  " + str(line) + "\n")
