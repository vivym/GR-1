import numpy as np

from evaluate_calvin_rdt import robot_obs_to_state_vec, state_vec_to_action


def main():
    robot_obs = np.load("eval_logs_rdt/obs_0_0.npy")

    print(robot_obs)

    state_vec, state_mask = robot_obs_to_state_vec(robot_obs)

    action = state_vec_to_action(state_vec[None])

    print(action)


if __name__ == "__main__":
    main()
