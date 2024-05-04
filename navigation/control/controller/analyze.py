import os
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('data_dir', type=str, default='results/navigation/')
    parser.add_argument('--scenarios', action='store_true')
    args = parser.parse_args()

    # Read evaluation CSV file(s)
    frames = []
    for file in os.listdir(args.data_dir):
        if not file.endswith('.csv'):
            continue
        frames.append(pd.read_csv(os.path.join(args.data_dir, file)))
    df = pd.concat(frames).reset_index()

    # Separate data based on controller type
    none = df.loc[~df['Overall']].loc[~df['Regional']]
    overall = df.loc[df['Overall']].loc[~df['Regional']]
    reg_turn = df.loc[~df['Overall']].loc[df['Regional']].loc[~df['Trajectory']]
    reg_traj = df.loc[~df['Overall']].loc[df['Regional']].loc[df['Trajectory']]
    both_turn = df.loc[df['Overall']].loc[df['Regional']].loc[~df['Trajectory']]
    both_traj = df.loc[df['Overall']].loc[df['Regional']].loc[df['Trajectory']]
    ctrl_dfs = [none, overall, reg_turn, reg_traj, both_turn, both_traj]
    controllers = ['None', 'Overall', 'Regional-Turning', 'Regional-Trajectory', 'Both-Turning', 'Both-Trajectory']

    # Create dataframe to combine all statistics of interest
    metrics = ['Successes', 'Timeouts', 'Collisions', 'Nav Time', 'Path Length', 'Lin Vel', 'Ang Vel', 'Lin Accel', 'Ang Accel']
    all_df = pd.DataFrame(columns=metrics, index=controllers)

    # Compute totals/averages across trials
    for df, ctrl in zip(ctrl_dfs, controllers):
        success = len(df[df['Result'] == 'success'])
        timeout = len(df) - success
        collision = len(df[df['Collision']])
        time   = df['Nav Time'].mean()
        path   = df[df['Result'] == 'success']['Path Length'].mean()
        v_lin  = df['Lin Vel'].mean()
        v_ang  = df['Ang Vel'].mean()
        a_lin  = df['Lin Accel'].mean()
        a_ang  = df['Ang Accel'].mean()
        all_df.loc[ctrl] = [success, timeout, collision, time, path, v_lin, v_ang, a_lin, a_ang]

    # Display results
    print(tabulate(all_df, headers='keys', tablefmt='psql'))

    if args.scenarios:
        scenarios = np.unique(df['Scenario'])
        for scenario in scenarios:
            # Compute totals/averages across trials
            for df, ctrl in zip(ctrl_dfs, controllers):
                scenario_df = df[df['Scenario'] == scenario]
                success = len(scenario_df[scenario_df['Result'] == 'success'])
                timeout = len(scenario_df) - success
                collision = len(scenario_df[scenario_df['Collision']])
                time   = scenario_df['Nav Time'].mean()
                path   = scenario_df[scenario_df['Result'] == 'success']['Path Length'].mean()
                v_lin  = scenario_df['Lin Vel'].mean()
                v_ang  = scenario_df['Ang Vel'].mean()
                a_lin  = scenario_df['Lin Accel'].mean()
                a_ang  = scenario_df['Ang Accel'].mean()
                all_df.loc[ctrl] = [success, timeout, collision, time, path, v_lin, v_ang, a_lin, a_ang]

            # Display results
            print(scenario)
            print(tabulate(all_df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    main()